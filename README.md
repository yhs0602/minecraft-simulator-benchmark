# Malmö vs Craftground Performance Comparison experiment
This experiment compares the performance of Malmö and Craftground in terms of FPS and memory usage. Here we use MineRL which is based on Malmö, because MineRL is most widely used in the research community.

# Performance Comparison
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


# Run options
- environment: minerl044, minerl100, craftground_raw, craftground_zerocopy
- image_width: 64x64, 114x64, 640x360
- load: simulation, render, ppo, render_ppo, optimized_render, optimized_ppo, optimized_render_ppo

# Environment Setup
- Conda, python=3.11
- Ubuntu Ubuntu 18.04.6 LTS, cuda driver version 525.105.17, NVIDIA GeForce RTX 3090 Ti, VirtualGL, RAM 188GB, AMD Ryzen Threadripper 3960X 24-Core Processor
- Apple M1 Pro, single process at once,  single process at once, normal load (not strict setting)

### MineRL 0.4.4 (exp_minerl044)
This may help solving issues such as https://github.com/minerllabs/minerl/issues/788.
```bash
conda create -n exp_minerl044 python=3.11
conda activate exp_minerl044
conda install conda-forge::openjdk=8 
pip install setuptools==65.5.0 pip==21 wheel==0.38.0
pip install minerl==0.4.4 # error because of MixinGradle
pip download --no-binary :all: minerl==0.4.4 # Hangs in installing build dependencies
pip install --upgrade pip
pip download --no-binary :all: minerl==0.4.4 # Does not hang? openssl error
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
# Takes a long long time to install
# ERROR: Problem encountered: NumPy requires GCC >= 8.4
sudo apt update
sudo apt install -y software-properties-common
sudo apt-get install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 100
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
sudo apt install -y libxml2-dev libxslt1-dev
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
# ERROR: Problem encountered: SciPy requires GCC >= 9.1
# Use this table to get latest gcc version your OS supports: https://askubuntu.com/a/1163021/901082
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-10 g++-10
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 120
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 120
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
# ERROR: Problem encountered: gfortran not found
sudo apt install gfortran
OPENSSL_ROOT_DIR=$CONDA_PREFIX pip download --no-binary :all: minerl==0.4.4
#       ../scipy/meson.build:274:9: ERROR: Dependency "OpenBLAS" not found, tried pkgconfig and cmake
sudo apt install libopenblas-dev
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
```bash
conda create -n exp_minerl100 python=3.11
conda activate exp_minerl100
conda install conda-forge::openjdk=8
pip install git+https://github.com/minerllabs/minerl
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
vglrun python experiments/minerl100_exp.py --image_width 64x64 --load simulation
```

### Craftground (craftground_raw | craftground_zerocopy: exp_craftground)
Latest cmake is required for Craftground to ensure it finds the cuda libraries correctly.
```bash
conda create -n exp_craftground python=3.11
conda activate exp_craftground
conda install conda-forge::openjdk=21 cmake
pip install craftground
pip install wandb tensorboard moviepy stable-baselines3
pip install --upgrade git+https://github.com/DLR-RM/stable-baselines3.git # To ensure correct video rendering
python experiments/craftground_exp.py --mode raw --image_width 64x64 --load simulation
```

# Experiment Results (Simulation Speed, CUDA version)
Both used vglrun to run the experiments on headless server.

For PPO, used stable-baselines3. For optimized version which uses tensor directly, we are planning to implement it in the future.
| Configuration            | Malmö | CraftGround RAW | CraftGround ZeroCopy |
| ------------------------ | ----- | --------------- | -------------------- |
| 64x64 Simul              | 57    | 192             | 146                  |
| 640x360 Simul            | 56    | 140             | 151                  |
| 64x64 Render             | ?     | 175             | 155                  |
| 640x360 Render           | ?     | 115             | 120                  |
| 64x64 PPO                | ?     | ?               | ?                    |
| 640x360 PPO              | ?     | ?               | ?                    |
| 64x64 PPO Render         | ?     | ?               | ?                    |
| 640x360 PPO Render       | ?     | ?               | ?                    |
| 64x64 Render Optim       | ?     | ?               | ?                    |
| 640x360 Render Optim     | ?     | ?               | ?                    |
| 64x64 PPO Render Optim   | ?     | ?               | ?                    |
| 640x360 PPO Render Optim | ?     | ?               | ?                    |

# TroubleShooting
```
FileExistsError: Socket file /tmp/minecraftrl_8001.sock already exists. Please choose another port.
```
Then
```bash
rm /tmp/minecraftrl_8001.sock 
```