from pathlib import Path

import ale_py
import gymnasium
gymnasium.register_envs(ale_py)

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.vec_env import is_vecenv_wrapped


def check_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        return
    import importlib.metadata as metadata
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = line.split("==")[0].split("[")[0].strip()
        pinned_version = line.split("==")[1].strip() if "==" in line else None
        if not pinned_version:
            continue
        try:
            installed = metadata.version(pkg_name)
            if installed != pinned_version:
                print(f"WARNING: {pkg_name} installed={installed} != pinned={pinned_version}")
        except metadata.PackageNotFoundError:
            print(f"WARNING: {pkg_name} not installed (pinned={pinned_version})")


def make_env(seed=None, render_mode=None, transpose_image=False):
    try:
        from config_local import ENV_ID, N_ENVS, N_STACK, SEED
    except ImportError:
        from config_colab import ENV_ID, N_ENVS, N_STACK, SEED

    if seed is None:
        seed = SEED

    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = make_atari_env(ENV_ID, n_envs=N_ENVS, seed=seed, env_kwargs=env_kwargs if env_kwargs else None)
    env = VecFrameStack(env, n_stack=N_STACK)
    if transpose_image:
        env = VecTransposeImage(env)
    return env
