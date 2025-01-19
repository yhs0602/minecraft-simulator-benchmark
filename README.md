# Optimizing Minecraft Simulations: Performance Comparison between Malmö and CraftGround
This experiment compares the performance of [Malmö](https://github.com/microsoft/malmo) and [Craftground](https://github.com/yhs0602/CraftGround) in terms of FPS and memory usage. Here we use [MineRL](https://github.com/minerllabs/minerl) which is based on Malmö, because MineRL is most widely used in the research community.

## Caveats
While CraftGround shows significant performance improvements in certain scenarios, there are some caveats to consider:

1. **Feature Gaps**
   - CraftGround currently does not support some advanced features available in Malmö-based environments, such as multi-agent scenarios, custom mission scripting, or integration with specific competitions like IGLU 2021.
   - The API interface of CraftGround differs significantly from Malmö, requiring users to adapt their codebase to use it effectively.
2. **Experimental Nature**: 
    - CraftGround is a relatively new framework and is still under active development. As a result, some features may be less stable or lack documentation compared to mature Malmö-based environments.
    - Certain optimizations, such as ZeroCopy mode, achieve high performance by bypassing abstractions and safety checks, which might introduce unexpected behavior in complex scenarios.
    - The ecosystem and community support for CraftGround are still growing, and users may encounter fewer readily available resources or troubleshooting guides compared to MineRL or Malmö.
  
> We are exploring the possibility of providing extensions for Malmö-based environments, such as MineRL, to enable the use of CraftGround as a backend with minimal code changes. This could enhance compatibility and performance for users looking for alternative backend solutions.


## Configurations
- Targets: Malmö(MineRL 0.4.4, MineRL 1.0.0) vs Craftground RAW vs Craftground ZeroCopy
- Steps: 100_000
- Image Width:
    - 64 x 64
    - 114 x 64
    - 640 x 360
- Settings
    - (Simulation)
    - (Simulation + Render)
    - (Simulation + PPO)
    - (Simulation + Render + PPO)
    - (Simulation + Optimized Render)
    - (Simulation + Optimized PPO)
    - (Simulation + Optimized Render + Optimized PPO)
- Metrics
    - FPS
    - Memory Usage


## Environment Setup
- Conda, python=3.11
- Ubuntu Ubuntu 18.04.6 LTS, cuda driver version 525.105.17, NVIDIA GeForce RTX 3090 Ti, VirtualGL, RAM 188GB, AMD Ryzen Threadripper 3960X 24-Core Processor
- Apple M1 Pro, single process at once,  single process at once, normal load (not strict setting)

### MineRL 0.4.4 (exp_minerl044, on Ubuntu 18.04.6 LTS)
This may help solving issues such as https://github.com/minerllabs/minerl/issues/788.
- Use this table to get latest gcc version your OS supports: https://askubuntu.com/a/1163021/901082
- To solve MixinGradle issue, follow the steps as mentioned here: https://github.com/MineDojo/MineDojo/issues/113#issuecomment-1908997704
```bash
conda create -n exp_minerl044 python=3.11
conda activate exp_minerl044
conda install conda-forge::openjdk=8 
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install gym==0.19.0
pip install --upgrade pip
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y libxml2-dev libxslt1-dev gfortran libopenblas-dev software-properties-common
# Ensure you have the latest version of gcc:
# To check the version of gcc, run `gcc --version`
sudo apt-get install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 120
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
tar -xvf minerl-0.4.4.tar.gz
cd minerl-0.4.4
# remove gym line from requirements.txt
pip install -r requirements.txt
# Edit minerl-0.4.4/minerl/Malmo/Minecraft/build.gradle:L19 based on
# https://github.com/MineDojo/MineDojo/issues/113#issuecomment-1908997704
# classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
# Add repository maven to the build.gradle
#         maven { url 'file:file:/absolute-path/to/that/repo's/parent' }
pip install .
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
vglrun python experiments/minerl044.py --image_width 64x64 --load simulation
```

### MineRL 1.0.0 (exp_minerl100)
Installing MineRL 1.0.0 is much easier than 0.4.4.
```bash
conda create -n exp_minerl100 python=3.11
conda activate exp_minerl100
conda install conda-forge::openjdk=8
pip install git+https://github.com/minerllabs/minerl
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
conda install -c anaconda cudnn # for ppo
vglrun python experiments/minerl100_exp.py --image_width 64x64 --load simulation
```

### Craftground
Latest cmake is required for Craftground to ensure it find the cuda libraries correctly.
```bash
conda create -n exp_craftground python=3.11
conda activate exp_craftground
conda install conda-forge::openjdk=21 cmake
sudo apt install libglew-dev libpng-dev zlib1g-dev
pip install craftground
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
# Test SBX, install JAX and Jaxlib
pip install jax jaxlib sbx
# On apple, to use metal backend
pip install jax-metal
# On other systems, to use cuda backend
pip install jaxlib

python experiments/craftground_exp.py --mode raw --image_width 64x64 --load simulation
```


## Run options
- environment: minerl044, minerl100, craftground_raw, craftground_zerocopy
- image_width: 64x64, 114x64, 640x360
- load: simulation, render, ppo, render_ppo, optimized_render, optimized_ppo, optimized_render_ppo


## Experiment Results (Frames Per Second, CUDA)
Both used vglrun to run the experiments on headless server.

For PPO, used stable-baselines3. For optimized version which uses tensor directly, we are planning to implement it in the future.
| Configuration            | Malmö | CraftGround RAW | CraftGround ZeroCopy | Speedup |
| ------------------------ | ----- | --------------- | -------------------- | ------- |
| 64x64 Simul              | 57    | 192             | 145                  | 3.36x   |
| 640x360 Simul            | 56    | 140             | 151                  | 2.7x    |
| 64x64 Render             | 58.5  | 175             | 155                  | 2.99x   |
| 640x360 Render           | 55    | 115             | 128                  | 2.33x   |
| 64x64 PPO                | 45    | 103             | 87                   | 2.29x   |
| 640x360 PPO              | 33    | 56.5            | 46                   | 1.71x   |
| 64x64 PPO Render         | 44.5  | 102             | 76                   | 2.29x   |
| 640x360 PPO Render       | 29.5  | 49              | 47                   | 1.66x   |
| 64x64 Render Optim       | *58.5 | *175            | ?                    |         |
| 640x360 Render Optim     | *55   | *115            | ?                    |         |
| 64x64 PPO Optim          | *45   | *103            | ?                    |         |
| 640x360 PPO Optim        | *33   | *56.5           | ?                    |         |
| 64x64 PPO Render Optim   | *44.5 | *102            | ?                    |         |
| 640x360 PPO Render Optim | *29.5 | *49             | ?                    |         |

* Since the optimized version is not implemented on Malmö and CraftGround RAW mode, the results are the same as the non-optimized version.

## Experiment Results (Frames Per Second, Apple M1 Pro)
| Configuration                 | Malmö | CraftGround RAW | CraftGround ZeroCopy |
| ----------------------------- | ----- | --------------- | -------------------- |
| 64x64 Simul                   | -     | 138             | 133.5                |
| 640x360 Simul                 |       | 90.5            | 117.5                |
| 64x64 Render                  |       | 129             | 144                  |
| 640x360 Render                |       | 111.5           | 134.5                |
| 64x64 PPO                     |       | 26              | 26                   |
| 64x64 PPO(SBX, CPU)           |       | 133.5           | -                    |
| 64x64 PPO(SBX, MPS)           |       | 103             | -                    |
| 640x360 PPO                   |       | 13              | 13.5                 |
| 640x360 PPO (SBX, CPU)        |       | 12              | -                    |
| 640x360 PPO (SBX, MPS)        |       | 44              | -                    |
| 64x64 PPO Render              |       | 27.5            | 25                   |
| 64x64 PPO Render (SBX, CPU)   |       | 147             | -                    |
| 64x64 PPO Render (SBX, MPS)   |       | 103             | -                    |
| 640x360 PPO Render            |       | 13              | 13.5                 |
| 640x360 PPO Render (SBX, CPU) |       | 11.5            | -                    |
| 640x360 PPO Render (SBX, MPS) |       | 42.5            | -                    |




# TroubleShooting
## Installing MineRL 0.4.4
### Error because of MixinGradle
Edit minerl-0.4.4/minerl/Malmo/Minecraft/build.gradle:L19 based on https://github.com/MineDojo/MineDojo/issues/113#issuecomment-1908997704
```gradle
classpath('MixinGradle-dcfaf61:MixinGradle:dcfaf61'){ // 0.6
```
Add repository maven to the build.gradle
```
maven { url 'file:file:/absolute-path/to/that/repo's/parent' }
```

### Hangs in installing build dependencies
```bash
pip install --upgrade pip
```

### OpenSSL error
Append `OPENSSL_ROOT_DIR=$CONDA_PREFIX` to the command if you are using conda. Otherwise, you can set the environment variable after installing openssl-dev.
```bash
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
```

### NumPy requires GCC >= 8.4, SciPy requires GCC >= 9.1
- Use this table to get latest gcc version your OS supports: https://askubuntu.com/a/1163021/901082

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y software-properties-common
sudo apt-get install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 120
```

### gfortran not found
```bash
sudo apt install gfortran
```

### OpenBLAS not found
```bash
sudo apt install libopenblas-dev
```


## Installing MineRL 1.0.0
Ensure you have the correct version of JDK. MineRL 1.0.0 requires JDK 8.
```bash
conda install conda-forge::openjdk=8
```
or 
```bash
sudo apt install openjdk-8-jdk
```
or 
```bash
jenv local 1.8
```
You can get the JDK 8 from various sources such as:
- https://www.azul.com/downloads/?version=java-8-lts&package=jdk#zulu

## Installing Craftground
Ensure you have the latest version of cmake. Currently the apt repository has cmake 3.10.2, which is not enough for Craftground. To install the latest version of cmake, you should use pip or conda to install it.
```bash
conda install cmake
```
or 
```bash
pip install --upgrade cmake
```


## Malmö, MineRL, and CraftGround is not using GPU on CUDA devices
You should install VirtualGL and run the experiments. Take a look at this MineRL documentation:

- https://minerl.readthedocs.io/en/latest/notes/performance-tips.html#faster-alternative-to-xvfb

Also you can check this guide:

- https://yhs0602.github.io/CraftGround/headless.html

```bash
echo $WAYLAND_DISPLAY
echo $XDG_SESSION_TYPE
ps aux | grep -E ’weston|sway’
sudo apt install virtualgl
wget https://sourceforge.net/projects/virtualgl/files/3.1/virtualgl_3.1_amd64.
deb/download
mv download vgl3.1.deb
sudo dpkg -i vgl3.1.deb
sudo vglserver_config
# During configuration, select the option to install both GLX and EGL and adjust device permissions as required
# In case you meed the following error, run the following command
# modprobe: FATAL: Module nvidia_drm is in use. You must execute modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia’ with the display manager stopped in order for the new device permission settings to become effective.
sudo systemctl stop gdm
sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
# If you meet modprobe: FATAL: Module nvidia_drm is in use.
sudo lsof /dev/nvidia*
pkill <pid>
# Restart the display manager
sudo modprobe nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo systemctl restart gdm
# Install Xvfb
sudo apt install xvfb
Xvfb :2 -screen 0 1024x768x24 +extension GLX -ac +extension RENDER & 
export DISPLAY=:2
VGL_DISPLAY=:0 vglrun /opt/VirtualGL/bin/glxspheres64
sudo nvidia-xconfig --query-gpu-info
sudo nvidia-xconfig -a --allow-empty-initial-configuration \
--use-display-device=None --virtual=1920x1200 \
--busid PCI:<BusID>
sudo systemctl restart gdm
VGL_DISPLAY=:0 vglrun /opt/VirtualGL/bin/glxspheres64
# OpenGL Renderer: NVIDIA GeForce RTX 3090/PCIe/SSE2.
```


## Running Craftground
### FileExistsError
```
FileExistsError: Socket file /tmp/minecraftrl_8001.sock already exists. Please choose another port.
```
Then
```bash
rm /tmp/minecraftrl_8001.sock 
```
### Zombie Minecraft process
```bash
jps -l # find the pid of something like DevLaunchInjector.Main
kill -9 <pid>
```


# License
This repository is basically licensed under the MIT License. However, the following files follows the original license of [stable-baselines3](https://github.com/DLR-RM/stable-baselines3/), which is [MIT License](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE):
- experiments/tensor_optimized/async_vec_video_recorder.py
- experiments/tensor_optimized/async_video_recorder.py
- experiments/optim_dummy_vec_env.py

We copied and modified a bit to make it work with tensor observation directly.