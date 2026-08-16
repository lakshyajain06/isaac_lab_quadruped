import gymnasium as gym

from . import walk_env_cfg

gym.register(
    id="pupper-walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": walk_env_cfg.PupperWalkEnvCfg,
        "skrl_cfg_entry_point": "isaac_lab_quadruped.tasks.manager_based.walk.agents:skrl_ppo_cfg.yaml"

    }
)