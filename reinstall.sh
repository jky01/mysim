cd ~/donkeycar/
pip uninstall donkeycar -y
pip install .

cd ~/gym-donkeycar
pip uninstall gym-donkeycar -y
pip install .

cd ~/stable-baselines3
pip uninstall stable_baselines3 -y
pip install .

cd ~/rl-baselines3-zoo
pip uninstall rl_zoo3 -y
pip install .

pip install -U albumentations
