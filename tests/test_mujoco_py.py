import subprocess


def test_mujoco_py():
    """
    Test mujoco_py
    """
    subprocess.run(
        "python cleanrl/torch/ppo_continuous_action.py --env-id Hopper-v2 --num-envs 1 --num-steps 64 --total-timesteps 256",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/torch/ddpg_continuous_action.py --env-id Hopper-v2 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/jax/ddpg_continuous_action.py --env-id Hopper-v2 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/jax/td3_continuous_action.py --env-id Hopper-v2 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/torch/td3_continuous_action.py --env-id Hopper-v2 --learning-starts 100 --batch-size 32 --total-timesteps 105",
        shell=True,
        check=True,
    )
    subprocess.run(
        "python cleanrl/torch/sac_continuous_action.py --env-id Hopper-v2 --batch-size 128 --total-timesteps 135",
        shell=True,
        check=True,
    )
