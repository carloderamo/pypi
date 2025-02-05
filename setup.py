from setuptools import setup, find_packages
from os import path
import glob

from mushroom_rl import __version__


def glob_data_files(data_package, data_type=None):
    data_type = '*' if data_type is None else data_type
    data_dir = data_package.replace(".", "/")
    data_files = [] 
    directories = glob.glob(data_dir+'/**/', recursive=True) 
    for directory in directories:
        subdir = directory[len(data_dir)+1:]
        if subdir != "":
            files = subdir + data_type
            data_files.append(files)
    return data_files


here = path.abspath(path.dirname(__file__))


extras = {
    'gymnasium': ['gymnasium'],
    'atari': ['ale-py', 'Pillow', 'opencv-python'],
    'box2d': ['box2d-py'],
    'bullet': ['pybullet'],
    'mujoco': ['mujoco>=2.3', 'dm_control>=1.0.9'],
    'plots': ['pyqtgraph']
}

all_deps = []
for group_name in extras:
    if group_name not in ['plots','box2d', 'bullet']:
        all_deps += extras[group_name]
extras['all'] = all_deps

print(extras['all'])

long_description = 'MushroomRL is a Python Reinforcement Learning (RL) library' \
                   ' whose modularity allows to easily use well-known Python' \
                   ' libraries for tensor computation (e.g. PyTorch, Tensorflow)' \
                   ' and RL benchmarks (e.g. OpenAI Gym, PyBullet, Deepmind' \
                   ' Control Suite). It allows to perform RL experiments in a' \
                   ' simple way providing classical RL algorithms' \
                   ' (e.g. Q-Learning, SARSA, FQI), and deep RL algorithms' \
                   ' (e.g. DQN, DDPG, SAC, TD3, TRPO, PPO). Full documentation' \
                   ' available at http://mushroomrl.readthedocs.io/en/latest/.'

mujoco_data_package = 'mushroom_rl.environments.mujoco_envs.data'
pybullet_data_package = 'mushroom_rl.environments.pybullet_envs.data'

setup(
    version=__version__,
    author="Carlo D'Eramo, Davide Tateo",
    url="https://github.com/MushroomRL",
    long_description=long_description,
    packages=[package for package in find_packages()
              if package.startswith('mushroom_rl')],
    extras_require=extras,
    package_data={
        mujoco_data_package: glob_data_files(mujoco_data_package),
        pybullet_data_package: glob_data_files(pybullet_data_package)
    }
)
