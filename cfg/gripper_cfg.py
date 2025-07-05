import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

YUMI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/inigo/Documents/isaaclab-link/pick_place/assets/yumi_gripper/yumi_gripper_mimic_mu.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        #articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #    enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        #),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "gripper_joint": 0.025,
            "gripper_joint_m": 0.025,
        },
    ),
    actuators={
        "yumi_gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint", "gripper_joint_m"],   # mimic joint follows automatically (gripper_joint_m)
            effort_limit_sim   = 20.0,
            velocity_limit_sim = 2.0,
            stiffness      = 2000.0,
            damping        = 40.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
# CHECK: friction same, damping stiffness, better pick