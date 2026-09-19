"""Active tasks. Environment imports remain lazy until Isaac Sim starts."""
import gymnasium as gym

for _family, _class in (('ame', 'AME'), ('affordance', 'Affordance')):
    for _stage in (1, 2):
        _id = f"ISAL2-RPO-{_class}-Stage{_stage}-v0"
        if _id not in gym.registry:
            _module = f"isal2.tasks.{_family}_stage{_stage}"
            gym.register(
                id=_id, entry_point=f"{_module}.env:{_class}Stage{_stage}Env",
                disable_env_checker=True, kwargs={
                    "env_cfg_entry_point": f"{_module}.env_cfg:RPO{_class}Stage{_stage}EnvCfg",
                    "rsl_rl_cfg_entry_point": f"{_module}.agents.ppo_cfg:{_class}Stage{_stage}AgentCfg",
                })
