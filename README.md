# mysim
conda activate donkey

python -m rl_zoo3.enjoy --algo tqc --env donkey-mountain-track-v0 -f logs/
python -m rl_zoo3.train --algo tqc --env donkey-mountain-track-v0 -f logs/

python rl_zoo3_train.py --algo ppo --env donkey-mountain-track-with-sam-v0 --n-timesteps 100000 --conf-file /home/aa/mysim/donkey-mountain-track-with-sam-v0.yml -tb ~/mysim/tensorboard_log

python rl_zoo3_enjoy.py --algo ppo --env donkey-mountain-track-with-sam-v0 --n-timesteps 100000 -f logs