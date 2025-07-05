import numpy as np
np.set_printoptions(suppress=True)

import os, types, logging
import base64, grequests
import torch


import isaaclab.sim as sim_utils

# Interactive Scene Class
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# CFGs
from cfg.franka_cfg import FRANKA_PANDA_HIGH_PD_CFG
from cfg.gripper_cfg import YUMI_CFG
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg #, Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim.schemas as sim_schemas

### Extensions
from isaacsim.core.utils.extensions import enable_extension

# Robot assembler
enable_extension("isaacsim.robot_setup.assembler")
from isaacsim.robot_setup.assembler import RobotAssembler

# Lula Inverse Kinematics
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.core.prims import SingleArticulation


from constants import Camera, Prims, RobotArm, DexNet
from pick_utils import Grasp, mat_to_quat


dir_ = os.path.dirname(os.path.realpath(__file__))


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    
    # Physics
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    # Prims
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path = os.path.join(dir_, Prims.Table.path)
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=Prims.Table.pos, rot=Prims.Table.rot)
    )

    KLT_pick = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KLT_pick",
        spawn=sim_utils.UsdFileCfg(
            usd_path= os.path.join(dir_, Prims.KLT_Bin.path)
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=Prims.KLT_Bin.pos_pick)
    )

    KLT_place = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KLT_place",
        spawn=sim_utils.UsdFileCfg(
            usd_path= os.path.join(dir_, Prims.KLT_Bin.path)
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=Prims.KLT_Bin.pos_place)
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.03, 0.03, 0.03),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(
                dynamic_friction=1.0, static_friction=1.0, restitution=0.05 # match with gripper
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.64, 0.32, 1.15)),
    )

    # Articulations
    franka: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka")
    franka.init_state.pos = RobotArm.FrankaArm.pos

    yumi_gripper: ArticulationCfg = YUMI_CFG.replace(
                prim_path="{ENV_REGEX_NS}/yumi_gripper",
    )

    # Sensors
    camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/camera",
    update_period=0.1,
    height=Camera.height,
    width=Camera.width,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=Camera.intr_matrix, height=Camera.height, width=Camera.width
    ),
    offset=CameraCfg.OffsetCfg(pos=tuple(Camera.pos), rot=tuple(Camera.rot), convention="ros"),
    )

    contact_forces_L = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_l")
    contact_forces_R = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_r")



class FrankaScene(InteractiveScene):
    def __init__(self, cfg: InteractiveSceneCfg):
        super().__init__(cfg)

        self.logger = logging.getLogger(__name__)

        self._arm_path_loc = "/Franka/franka_instanceable"
        self._gripper_path_loc = "/yumi_gripper/yumi_gripper"

        self._franka_assets = []
        self._kinematics_solver = None
        self._articulation_kinematics_solvers = []

        self.mode = 0

        self.DexNet_request = None
        self.is_request = False

    
    def setup_post_load(self):

        for env_path in self.env_prim_paths:
            arm_path     = env_path + self._arm_path_loc
            gripper_path = env_path + self._gripper_path_loc

            self._attach_gripper(arm_path, gripper_path)
            #sim_schemas.activate_contact_sensors(gripper_path)

        self._load_InverseKinematics()

    
    def _attach_gripper(self, arm_path, gripper_path):
        robot_assembler = RobotAssembler()
        robot_assembler.assemble_articulations(
            base_robot_path = arm_path,
            attach_robot_path = gripper_path,
            base_robot_mount_frame = "/panda_link7",
            attach_robot_mount_frame = "/gripper_base",
            fixed_joint_offset = RobotArm.Gripper.offset_pos,
            fixed_joint_orient = np.array([1.0,0.0,0.0,0.0]),
            mask_all_collisions = True,
            single_robot=False
        )


    def _load_InverseKinematics(self):
        kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        end_effector_name = "right_gripper" #panda_link8 #right_gripper #panda_wrist_end_pt
        for env_path in self.env_prim_paths:
            franka_path = env_path + self._arm_path_loc
            franka_single = SingleArticulation(franka_path)
            self._franka_assets.append(franka_single)

            articulation_IK_solver = ArticulationKinematicsSolver(franka_single, self._kinematics_solver, end_effector_name)
            self._articulation_kinematics_solvers.append(articulation_IK_solver)
        
            ###
            jview = articulation_IK_solver._joints_view
            orig   = jview.get_joint_positions

            def _as_numpy(self, *a, **kw):
                t = orig(*a, **kw)
                return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

            jview.get_joint_positions = types.MethodType(_as_numpy, jview)
            ###
    

    def franka_initialize(self):
        for franka in self._franka_assets:
            franka.initialize()


    def compute_IK(self, target_pos, target_ori):

        q_target = []
        for IK_solver in self._articulation_kinematics_solvers:
            robot_base_translation, robot_base_orientation = np.array([0.0, 0.0, 1.05]), np.array([1.0, 0.0, 0.0, 0.0])
            self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation) #singlearticulation.get_world_pose()

            action, ok = IK_solver.compute_inverse_kinematics(target_pos, target_ori)

            ik_pos = torch.tensor(
                action.joint_positions,
                dtype   = self.articulations["franka"].data.joint_pos.dtype,
                device  = self.articulations["franka"].data.joint_pos.device,
            )

            q_target.append(ik_pos)

        q_target = torch.stack(q_target, dim=0)  

        return q_target
    

    def is_target_reached(self, target_pos, target_rot, atol=5e-3):

        if target_pos is None or target_rot is None:
            return False

        ee_position, ee_rot_mat = self._articulation_kinematics_solvers[0].compute_end_effector_pose()
        ee_rot = mat_to_quat(ee_rot_mat)

        is_target_reached = np.allclose(ee_position, target_pos, atol=atol) \
                        and np.allclose(ee_rot, target_rot, atol=atol)
        print(ee_rot, target_rot)
        # print(np.abs(target_pos-ee_position))

        return is_target_reached


    def is_obj_clamped(self):
        
        force_L = self.sensors["contact_forces_L"].data.net_forces_w.abs().max() # torch.max(scene["contact_forces"].data.net_forces_w).item())
        force_R = self.sensors["contact_forces_R"].data.net_forces_w.abs().max()

        is_obj_clambed = int(force_L > 0.5 and force_R > 0.5)

        return is_obj_clambed

    
    def request_DexNet_pred(self):

        self.logger.info("Requesting grasp predictions from Dex-Net")

        depth_tensor = self.sensors["camera"].data.output["distance_to_image_plane"]
        depth_np = depth_tensor.cpu().numpy()
        depth_np = depth_np[0,:,:,:]

        payload = {
            "shape": list(depth_np.shape),
            "dtype": str(depth_np.dtype),
            "data" : base64.b64encode(depth_np.tobytes()).decode()
        }

        req = grequests.post(DexNet.url, json=payload)
        self.DexNet_request = grequests.send(req)

        self.is_request = True


    def get_DexNet_pred(self):

        if self.DexNet_request is None or not self.DexNet_request.ready():
            return None

        reply = self.DexNet_request.value.response.json()
        grasps_np = (
            np.frombuffer(bytes.fromhex(reply["data"]),
                                dtype=reply["dtype"])
                            .reshape(reply["shape"])
        )

        self.DexNet_request = None
        self.is_request = False
        
        self.logger.info("Successfully received grasp predictions from Dex-Net")

        grasps = [Grasp(row) for row in grasps_np]

        return grasps