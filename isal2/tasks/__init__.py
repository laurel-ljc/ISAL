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

AME_TASK_ID = "ISAL2-RPO-AME-v0"
if AME_TASK_ID not in gym.registry:
    gym.register(
        id=AME_TASK_ID,
        entry_point="isal2.tasks.ame.ame_env:AMEEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isal2.tasks.ame.ame_env_cfg:RPOAMEEnvCfg",
            "rsl_rl_cfg_entry_point": "isal2.tasks.ame.agents.ppo_cfg:AMEAgentCfg",
        },
    )
