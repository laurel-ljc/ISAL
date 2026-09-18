"""Runnable base config; subclasses reuse scene, rewards and lifecycle."""
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg, MeshPlaneTerrainCfg
from isal2 import PROJECT_ROOT
from isal2.assets.robots import RPO_CFG
from .base_config import BaseEnvCfg
from .rpo_env_cfg import RPORewardCfg
from .scene_cfg import SceneCfg
from .terrain_generator_cfg import ROUGH_TERRAINS_CFG, ROUGH_HARD_TERRAINS_CFG


@configclass
class RPOBaseEnvCfg(BaseEnvCfg):
    terrain_preset: str = "rough"
    reward = RPORewardCfg()
    seed: int = 42

    def __post_init__(self):
        self.action_space = 23
        self.scene_context.robot = RPO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.robot.spawn.usd_dir = str(PROJECT_ROOT / ".cache" / "rpo")
        self.scene_context.robot.spawn.usd_file_name = "rpo.usd"
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.height_scanner.enable_height_scan_actor = False
        self.robot.terminate_contacts_body_names = ["torso_link", ".*_thigh_yaw_link", ".*_thigh_roll_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["torso_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["torso_link", "base_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.configure(self.terrain_preset)

    def configure(self, terrain=None, num_envs=None, terrain_rows=None, terrain_cols=None):
        """Apply CLI scene overrides before environment creation; dimensions stay consistent."""
        if terrain is not None:
            self.terrain_preset = terrain
        if num_envs is not None:
            if num_envs < 1:
                raise ValueError("num_envs must be positive")
            self.scene_context.num_envs = num_envs
        if self.terrain_preset == "flat":
            # A local mesh avoids the standard plane's remote USD asset dependency.
            self.scene_context.terrain_type = "generator"
            self.scene_context.terrain_generator = TerrainGeneratorCfg(
                size=(1000.0, 1000.0), border_width=0.0, num_rows=1, num_cols=1,
                curriculum=False, use_cache=False, seed=self.seed,
                sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)})
            self.scene_context.height_scanner.enable_height_scan = False
            self.reward.ang_vel_xy_l2.weight = -0.1
            self.reward.lin_vel_z_l2.weight = -0.2
        elif self.terrain_preset in ("rough", "rough_hard"):
            self.scene_context.terrain_type = "generator"
            generator = {"rough": ROUGH_TERRAINS_CFG, "rough_hard": ROUGH_HARD_TERRAINS_CFG}[self.terrain_preset].copy()
            generator.seed = self.seed
            generator.cache_dir = str(PROJECT_ROOT / ".cache" / "terrain")
            if terrain_rows is not None:
                generator.num_rows = terrain_rows
            if terrain_cols is not None:
                generator.num_cols = terrain_cols
            self.scene_context.terrain_generator = generator
            self.scene_context.height_scanner.enable_height_scan = True
            self.reward.ang_vel_xy_l2.weight = -0.05
            self.reward.lin_vel_z_l2.weight = -0.05
        elif not self._configure_custom_terrain(terrain_rows, terrain_cols):
            raise ValueError(f"Unknown terrain preset: {self.terrain_preset}")
        self._configure_observation_sensors()
        self.actor_frame_dim = 9 + 3 * self.action_space
        self.critic_frame_dim = self.actor_frame_dim + 3 + 2 + 6 + 2 + 2 + 2 * self.action_space
        if self.scene_context.height_scanner.enable_height_scan:
            scan = self.scene_context.height_scanner
            self.critic_frame_dim += (round(scan.size[0] / scan.resolution) + 1) * (round(scan.size[1] / scan.resolution) + 1)
        self.observation_space = self.actor_frame_dim * self.robot.actor_obs_history_length
        self.state_space = self.critic_frame_dim * self.robot.critic_obs_history_length
        self.scene = SceneCfg(self.scene_context, self.sim.dt, self.decimation * self.sim.dt)
        self.scene.terrain.use_terrain_origins = self.terrain_preset != "flat"
        self._configure_scene()
        return self

    def _configure_custom_terrain(self, rows, cols):
        return False

    def _configure_scene(self):
        """Optional task-specific scene additions after standard sensors exist."""

    def _configure_observation_sensors(self):
        """Subclasses override before dimensions and the scene are constructed."""
