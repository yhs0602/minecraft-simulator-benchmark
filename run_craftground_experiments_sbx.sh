count=3
for i in $(seq $count); do

# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load sbx-ppo --port 8001
# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load sbx-ppo --port 8002
# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load sbx-ppo --port 8003
python experiments/craftground_exp.py --mode raw --image_width 64x64 --load render_sbx-ppo --port 8004
python experiments/craftground_exp.py --mode raw --image_width 64x64 --load render_sbx-ppo --port 8005
python experiments/craftground_exp.py --mode raw --image_width 64x64 --load render_sbx-ppo --port 8006
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load sbx-ppo --port 8006
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load sbx-ppo --port 8007
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load render_sbx-ppo --port 8008
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load render_sbx-ppo --port 8009

# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load sbx-ppo --device cpu --port 8010
# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load sbx-ppo --device cpu --port 8011
# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load sbx-ppo --device cpu --port 8012
# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load render_sbx-ppo --device cpu --port 8013
# python experiments/craftground_exp.py --mode raw --image_width 64x64 --load render_sbx-ppo --device cpu --port 8014
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load sbx-ppo --device cpu --port 8015
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load sbx-ppo --device cpu --port 8016
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load render_sbx-ppo --device cpu --port 8017
# python experiments/craftground_exp.py --mode raw --image_width 640x360 --load render_sbx-ppo --device cpu --port 8018

done