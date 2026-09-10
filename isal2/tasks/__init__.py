"""Register only implemented tasks; import after starting Isaac Sim."""
import gymnasium as gym

TASK_ID = "ISAL2-RPO-Base-v0"
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isal2.tasks.base.base_env:BaseEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isal2.tasks.base.base_env_cfg:RPOBaseEnvCfg",
            "rsl_rl_cfg_entry_point": "isal2.tasks.base.agents.ppo_cfg:BaseAgentCfg",
        },
    )
