"""A wrapper for video recording environments by rolling it out, frame by frame."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import os.path
import tempfile
import threading
from typing import List, Optional

from gymnasium import error, logger
import torch


class AsyncTensorVideoRecorder:
    """VideoRecorder renders a nice movie of a rollout, frame by frame.

    It comes with an ``enabled`` option, so you can still use the same code on episodes where you don't want to record video.

    Note:
        You are responsible for calling :meth:`close` on a created VideoRecorder, or else you may leak an encoder process.
    """

    def __init__(
        self,
        env,
        path: Optional[str] = None,
        metadata: Optional[dict] = None,
        enabled: bool = True,
        base_path: Optional[str] = None,
        disable_logger: bool = False,
    ):
        """Video recorder renders a nice movie of a rollout, frame by frame.

        Args:
            env (Env): Environment to take video of.
            path (Optional[str]): Path to the video file; will be randomly chosen if omitted.
            metadata (Optional[dict]): Contents to save to the metadata file.
            enabled (bool): Whether to actually record video, or just no-op (for convenience)
            base_path (Optional[str]): Alternatively, path to the video file without extension, which will be added.
            disable_logger (bool): Whether to disable moviepy logger or not.

        Raises:
            Error: You can pass at most one of `path` or `base_path`
            Error: Invalid path given that must have a particular file extension
        """
        self.enabled = enabled
        self.disable_logger = disable_logger
        self._closed = False

        self.render_history = []
        self.env = env

        self.render_mode = env.render_mode

        try:
            # check that moviepy is now installed
            import moviepy  # noqa: F401
        except ImportError as e:
            raise error.DependencyNotInstalled(
                "moviepy is not installed, run `pip install moviepy`"
            ) from e

        if self.render_mode in {None, "human", "ansi", "ansi_list"}:
            raise ValueError(
                f"Render mode is {self.render_mode}, which is incompatible with"
                f" RecordVideo. Initialize your environment with a render_mode"
                f" that returns an image, such as rgb_array."
            )

        # Don't bother setting anything else if not enabled
        if not self.enabled:
            return

        if path is not None and base_path is not None:
            raise error.Error("You can pass at most one of `path` or `base_path`.")

        required_ext = ".mp4"
        if path is None:
            if base_path is not None:
                # Base path given, append ext
                path = base_path + required_ext
            else:
                # Otherwise, just generate a unique filename
                with tempfile.NamedTemporaryFile(suffix=required_ext) as f:
                    path = f.name
        self.path = path

        path_base, actual_ext = os.path.splitext(self.path)

        if actual_ext != required_ext:
            raise error.Error(
                f"Invalid path given: {self.path} -- must have file extension {required_ext}."
            )

        self.frames_per_sec = env.metadata.get("render_fps", 30)

        self.broken = False

        # Dump metadata
        self.metadata = metadata or {}
        self.metadata["content_type"] = "video/mp4"
        self.metadata_path = f"{path_base}.meta.json"
        self.write_metadata()

        logger.info(f"Starting new video recorder writing to {self.path}")

        self.active_buffer: List[torch.Tensor] = []
        self.background_buffer: List[torch.Tensor] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = asyncio.get_event_loop()
        self.lock = threading.Lock()
        self._flushing_task = None
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

    def _start_loop(self):
        """Run the event loop in a background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _flush_background_buffer(self):
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        except ImportError as e:
            raise error.DependencyNotInstalled(
                "moviepy is not installed, run `pip install moviepy`"
            ) from e
        """Flush the background buffer to a video file."""
        if not self.background_buffer:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True
            self.write_metadata()
            return
        frames = (frame.numpy() for frame in self.background_buffer)
        clip = ImageSequenceClip(frames, fps=self.frames_per_sec)
        moviepy_logger = None if self.disable_logger else "bar"
        clip.write_videofile(self.path, logger=moviepy_logger)
        self.background_buffer = []

    def reset(
        self,
        metadata: Optional[dict] = None,
        base_path: Optional[str] = None,
    ):
        if not self.enabled or self._closed:
            return

        if self.active_buffer:
            self._swap_buffers()

        if self._flushing_task:
            asyncio.run(self._flushing_task)

        # TODO: do something similar to __init__, but not clearing buffers, metadata, etc.
        ...

    @property
    def functional(self):
        """Returns if the video recorder is functional, is enabled and not broken."""
        return self.enabled and not self.broken

    def capture_frame(self):
        """Render the given `env` and add the resulting frame to the video."""
        frame = self.env.render()  #  Here the env will return a gpu tensor.
        if isinstance(frame, List):
            self.render_history += frame
            frame = frame[-1]

        if not self.functional:
            return
        if self._closed:
            logger.warn(
                "The video recorder has been closed and no frames will be captured anymore."
            )
            return
        logger.debug("Capturing video frame: path=%s", self.path)

        if frame is None:
            if self._async:
                return
            else:
                # Indicates a bug in the environment: don't want to raise
                # an error here.
                logger.warn(
                    "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                    f"disabled: path={self.path} metadata_path={self.metadata_path}"
                )
                self.broken = True
        else:
            with self.lock:
                self.active_buffer.append(frame.to("cpu", non_blocking=True))

    def _swap_buffers(self):
        """Swap active and background buffers and process background buffer."""
        with self.lock:
            self.active_buffer, self.background_buffer = (
                self.background_buffer,
                self.active_buffer,
            )
        self._flushing_task = self.loop.run_in_executor(
            self.executor, self._flush_background_buffer
        )

    def close(self):
        """Flush all data to disk and close any open frame encoders."""
        if not self.enabled or self._closed:
            return

        if self.active_buffer:
            self._swap_buffers()

        if self._flushing_task:
            asyncio.run(self._flushing_task)

        self.executor.shutdown(wait=True)
        self._closed = True

    def write_metadata(self):
        """Writes metadata to metadata path."""
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f)

    def __del__(self):
        """Closes the environment correctly when the recorder is deleted."""
        # Make sure we've closed up shop when garbage collecting
        if not self._closed:
            logger.warn("Unable to save last video! Did you call close()?")
