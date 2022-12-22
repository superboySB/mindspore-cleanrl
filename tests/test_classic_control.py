import subprocess


def test_ppo():
    subprocess.run(
        "python cleanrl/torch/ppo.py --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )


def test_dqn():
    subprocess.run(
        "python cleanrl/torch/dqn.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_c51():
    subprocess.run(
        "python cleanrl/torch/c51.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )


def test_dqn_jax():
    subprocess.run(
        "python cleanrl/jax/dqn.py --learning-starts 200 --total-timesteps 205",
        shell=True,
        check=True,
    )
