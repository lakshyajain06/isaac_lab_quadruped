import math

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

from . import mdp


##
# Pre-defined configs
##

from isaac_lab_quadruped.robots.pupper import PUPPER_CFG


##
# Scene definition
##

@configclass
class PupperWalkSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # robot
    robot = PUPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/.*_3",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )

##
# MDP settings
##

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # base velocities
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

        # gravity vector for orientation
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # joint states
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # history
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_positions = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
    )

@configclass
class CommandsCfg:
    """command specification for the env"""
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0,1.0),
        )
    )

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # tracking velocity rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # Penalties for erratic motion / energy waste
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)          # penalizes vertical motion in base
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)       # penalizes pitch and roll change
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5) # penalizes excessive torques
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)         # penalizes excessive acceleration
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)     # penalizes large action achanges

    # Gait shaping
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_3"),
            "command_name": "base_velocity",
            "threshold": 0.3,
        },
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_contact_forces", body_names="base_link"), "threshold": 1.0},
    )

    # terminate if base height is too low
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.1},
    )

    # terminate if robot tipps over too much
    base_angle = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.52}
    )

##
# Environment configuration
##

@configclass
class PupperWalkEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PupperWalkSceneCfg = PupperWalkSceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Randomization Settings
    events: EventCfg = EventCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation