import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/torch/ppo_atari.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_ppo_lstm():
    subprocess.run(
        "python cleanrl/torch/ppo_atari_lstm.py --num-envs 4 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn():
    subprocess.run(
        "python cleanrl/torch/dqn_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )

def test_dqn_jax():
    subprocess.run(
        "python cleanrl/jax/dqn_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )



def test_c51():
    subprocess.run(
        "python cleanrl/torch/c51_atari.py --learning-starts 10 --total-timesteps 16 --buffer-size 10 --batch-size 4",
        shell=True,
        check=True,
    )
