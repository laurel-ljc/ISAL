"""Active tasks. Environment imports remain lazy until Isaac Sim starts."""
import gymnasium as gym

for _stage in (1, 2):
    _id = f"ISAL2-RPO-AME-Stage{_stage}-v0"
    if _id not in gym.registry:
        _module = f"isal2.tasks.ame_stage{_stage}"
        gym.register(
            id=_id, entry_point=f"{_module}.env:AMEStage{_stage}Env",
            disable_env_checker=True, kwargs={
                "env_cfg_entry_point": f"{_module}.env_cfg:RPOAMEStage{_stage}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{_module}.agents.ppo_cfg:AMEStage{_stage}AgentCfg",
            })
