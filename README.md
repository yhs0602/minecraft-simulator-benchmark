# Performance Comparison
## Configurations
- Targets: MineRL 0.4.4 vs MineRL 1.0.0 vs Craftground RAW vs Craftground ZeroCopy
- Steps: 100_000
- Image Width:
    - 64 x 64
    - 114 x 64
    - 640 x 320
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
- image_width: 64x64, 114x64, 640x320
- load: simulation, render, ppo, render_ppo, optimized_render, optimized_ppo, optimized_render_ppo

# Environment Setup
- Conda
- MULS, single process at once
- APPLE, single process at once, normal load (not strict setting)

### MineRL 0.4.4 (minerl044)
```bash
conda install conda-forge::openjdk=8 
pip install setuptools==65.5.0 pip==21
pip install wheel==0.38.0
pip install minerl==0.4.4

conda activate exp_minerl044
python experiments/minerl044.py --image_width 64x64 --load simulation
```

### MineRL 1.0.0 (minerl100)
```bash
conda activate exp_minerl100
python experiments/minerl100.py --image_width 64x64 --load simulation
```

### Craftground (craftground_raw | craftground_zerocopy)
```bash
conda activate exp_craftground
python experiments/craftground.py --mode=raw|zerocopy --image_width 64x64 --load simulation
```