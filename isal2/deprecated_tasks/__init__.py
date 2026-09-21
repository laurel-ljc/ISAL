"""Register only implemented tasks; import after starting Isaac Sim."""
import gymnasium as gym

for _family, _class in (("ame", "AME"), ("affordance", "Affordance")):
    for _stage in (1, 2):
        _id = f"ISAL2-RPO-{_class}-Endpoint-Stage{_stage}-v0"
        _module = f"isal2.deprecated_tasks.endpoint_course.{_family}_stage{_stage}"
        if _id not in gym.registry:
            gym.register(id=_id, entry_point=f"{_module}.env:{_class}Stage{_stage}Env",
                disable_env_checker=True, kwargs={
                    "env_cfg_entry_point": f"{_module}.env_cfg:RPO{_class}Stage{_stage}EnvCfg",
                    "rsl_rl_cfg_entry_point": f"{_module}.agents.ppo_cfg:{_class}Stage{_stage}AgentCfg"})

for _family, _class in (("ame", "AME"), ("affordance", "Affordance")):
    _id = f"ISAL2-RPO-{_class}-Sparse-v0"
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=f"isal2.deprecated_tasks.{_family}_sparse.env:{_class}SparseEnv",
            disable_env_checker=True, kwargs={
                "env_cfg_entry_point": f"isal2.deprecated_tasks.{_family}_sparse.env_cfg:RPO{_class}SparseEnvCfg",
                "rsl_rl_cfg_entry_point": f"isal2.deprecated_tasks.{_family}_sparse.agents.ppo_cfg:{_class}SparseAgentCfg"})

if "ISAL2-RPO-Affordance-v0" not in gym.registry:
    gym.register(
        id="ISAL2-RPO-Affordance-v0",
        entry_point="isal2.deprecated_tasks.affordance.affordance_env:AffordanceEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": "isal2.deprecated_tasks.affordance.affordance_env_cfg:RPOAffordanceEnvCfg",
                "rsl_rl_cfg_entry_point": "isal2.deprecated_tasks.affordance.agents.ppo_cfg:AffordanceAgentCfg"},
    )

TASK_ID = "ISAL2-RPO-Base-v0"
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isal2.deprecated_tasks.base.base_env:BaseEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isal2.deprecated_tasks.base.base_env_cfg:RPOBaseEnvCfg",
            "rsl_rl_cfg_entry_point": "isal2.deprecated_tasks.base.agents.ppo_cfg:BaseAgentCfg",
        },
    )

AME_TASK_ID = "ISAL2-RPO-AME-v0"
if AME_TASK_ID not in gym.registry:
    gym.register(
        id=AME_TASK_ID,
        entry_point="isal2.deprecated_tasks.ame.ame_env:AMEEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isal2.deprecated_tasks.ame.ame_env_cfg:RPOAMEEnvCfg",
            "rsl_rl_cfg_entry_point": "isal2.deprecated_tasks.ame.agents.ppo_cfg:AMEAgentCfg",
        },
    )
