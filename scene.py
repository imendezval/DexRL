import numpy as np
np.set_printoptions(suppress=True)

import os, types, logging
import base64, grequests
import torch
import random, math


import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
import isaaclab.utils.math as math_utils

import omni.physics.tensors.impl.api as physx

# Interactive Scene Class
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# CFGs
from cfg.franka_cfg import FRANKA_PANDA_HIGH_PD_CFG
from cfg.gripper_cfg import YUMI_CFG
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg, RigidObjectCollection
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim.schemas as sim_schemas
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg

### Extensions
from isaacsim.core.utils.extensions import enable_extension

# Robot assembler
enable_extension("isaacsim.robot_setup.assembler")
from isaacsim.robot_setup.assembler import RobotAssembler

# Lula Inverse Kinematics
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver, \
                                                    LulaTaskSpaceTrajectoryGenerator, ArticulationTrajectory
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.core.prims import SingleArticulation


from helpers import Grasp, mat_to_quat, quat_diff_mag, offset_target_pos
from constants import Camera, Prims, RobotArm, DexNet, Settings

ObjPool = Prims.ObjPool



dir_ = os.path.dirname(os.path.realpath(__file__))

OBJ_PATTERNS = [
    f"{{ENV_REGEX_NS}}/obj_{i}"
    for i in range(ObjPool.n_obj_pool)      # 0 … 128
]


class ObjectPool(object):

    def __init__(self):

        # TODO: Generate Prims.ObjPool.n_obj_pool random objects from folders only
        pool_path = os.path.join(dir_, ObjPool.path)
        self.object_pool_cfg = {}

        gird_size = round(math.sqrt(ObjPool.n_obj_pool))
        spacing = 0.1

        for i, USD_file in enumerate(os.scandir(pool_path)):

            if not USD_file.is_file() or os.path.splitext(USD_file.name)[1] != ".usd":
                continue
            
            col = i // gird_size
            row = i  % gird_size

            usd_path = os.path.join(pool_path, USD_file)

            cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/obj_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(random.random(), random.random(), random.random()), metallic=0.2),                    
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(row * spacing, 0.65 + col * spacing, 0)),
            )

            self.object_pool_cfg[f"obj_{i}"] = cfg

            i += 1



@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    
    # Physics
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    # Base Scene
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path = os.path.join(dir_, Prims.Table.path),
            scale=Prims.Table.scale
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=Prims.Table.pos, rot=Prims.Table.rot)
    )

    KLT_pick = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KLT_pick",
        spawn=sim_utils.UsdFileCfg(
            usd_path= os.path.join(dir_, Prims.KLT_Bin.path),
            scale=Prims.KLT_Bin.scale
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=Prims.KLT_Bin.pos_pick)
    )

    KLT_place = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/KLT_place",
        spawn=sim_utils.UsdFileCfg(
            usd_path= os.path.join(dir_, Prims.KLT_Bin.path),
            scale=Prims.KLT_Bin.scale
        ),
        init_state = AssetBaseCfg.InitialStateCfg(pos=Prims.KLT_Bin.pos_place)
    )
    
    # Object Pool
    obj_pool = RigidObjectCollectionCfg(
                rigid_objects=ObjectPool().object_pool_cfg
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

    contact_forces_L = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_l", 
                                        filter_prim_paths_expr = OBJ_PATTERNS)
    contact_forces_R = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_r", 
                                        filter_prim_paths_expr = OBJ_PATTERNS)



class FrankaScene(InteractiveScene):
    def __init__(self, cfg: InteractiveSceneCfg):
        super().__init__(cfg)

        self.logger = logging.getLogger(__name__)

        self.env_ids = torch.as_tensor([env_id for env_id in range(self.num_envs)], device=self.device)

        self._arm_path_loc      = "/Franka/franka_instanceable"
        self._gripper_path_loc  = "/yumi_gripper/yumi_gripper"
        self.end_effector_name  = "right_gripper" # panda_link8 panda_wrist_end_pt

        # Kinematics Solver
        self._franka_assets        = []
        self._kinematics_solver    = None
        self._articulation_kinematics_solvers = []
        # Trajectory Generator
        self._franka_articulations = []
        self._taskspace_trajectory_generator  = None
        self.action_sequence    = [None for _ in range(self.num_envs)]
        self.sequence_step_init = [None for _ in range(self.num_envs)]

        # Joint Targets - init in main
        self.q_target_franka   = None
        self.q_target_yumi     = None
        self.yumi_vel_vec      = None

        # Sequence Flags
        self.mode    = torch.full((self.num_envs,), -2, dtype=torch.long, device=self.device)
        self.counter = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        # DexNet Requests
        self.DexNet_request = [None for _ in range(self.num_envs)]
        self.request_state  = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # DexNet Grasps
        self.GRASPS     = torch.zeros(self.num_envs, 3, device=self.device)
        self.PRE_GRASPS = torch.zeros(self.num_envs, 3, device=self.device)
        self.GRASPS_ROT = torch.zeros(self.num_envs, 4, device=self.device)

        # Bookkeeping
        self.gripped_objs      = torch.full((self.num_envs,), -1, 
                                        dtype=torch.long, device=self.device)
        self.gripped_objs_prev = torch.full((self.num_envs,), -1, 
                                        dtype=torch.long, device=self.device)
        self.grip_log          = torch.zeros(ObjPool.n_obj_pool, 2, 
                                        dtype=torch.uint8, device=self.device)
        self.fail_streaks      = torch.zeros(self.num_envs,
                                        dtype=torch.uint8, device=self.device)


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

        self._kinematics_solver              = LulaKinematicsSolver(**kinematics_config)
        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(**kinematics_config)
    
        robot_base_translation, robot_base_orientation = np.array([0.0, 0.0, 1.05]), np.array([1.0, 0.0, 0.0, 0.0])
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)        

        for env_path in self.env_prim_paths:
            franka_path = env_path + self._arm_path_loc

            # franka_articulation = Articulation(franka_path)
            # self._franka_articulations.append(franka_articulation)

            franka_single       = SingleArticulation(franka_path)
            self._franka_assets.append(franka_single)

            articulation_IK_solver = ArticulationKinematicsSolver(franka_single, self._kinematics_solver, self.end_effector_name)
            self._articulation_kinematics_solvers.append(articulation_IK_solver)
        
            ###
            jview = articulation_IK_solver._joints_view
            orig  = jview.get_joint_positions

            def make_wrapper(orig_fn):
                def _as_numpy(self, *a, **kw):
                    t = orig_fn(*a, **kw)
                    return (
                        t.detach().cpu().numpy()
                        if isinstance(t, torch.Tensor) else t
                    )
                return _as_numpy

            jview.get_joint_positions = types.MethodType(
                make_wrapper(orig), jview
            )
            ###
    

    def franka_initialize(self):
        for franka in self._franka_assets:
            franka.initialize()


    def compute_IK(self, target_pos, target_rot, env_ids, is_tensor: bool = False):

        q_target = self.q_target_franka.clone()

        for env_id in env_ids.cpu().tolist():

            if is_tensor:
                pos_np = target_pos[env_id].cpu().numpy()
                rot_np = target_rot[env_id].cpu().numpy()
                action, _ = self._articulation_kinematics_solvers[env_id].compute_inverse_kinematics(pos_np, rot_np)
            else:
                action, _ = self._articulation_kinematics_solvers[env_id].compute_inverse_kinematics(target_pos, target_rot)    

            ik_pos = torch.tensor(
                action.joint_positions,
                dtype   = self.articulations["franka"].data.joint_pos.dtype,
                device  = self.articulations["franka"].data.joint_pos.device,
            )

            q_target[env_id] = ik_pos
        
        return q_target
    

    def compute_IK_trajectory(self, target_pos, current_pos, target_rot, env_ids, step_init):

        for env_id in env_ids.cpu().tolist():

            curr_np = current_pos[env_id].cpu().numpy()
            pos_np  = target_pos[env_id].cpu().numpy()
            rot_np  = target_rot[env_id].cpu().numpy()

            pos_targets  = np.array([curr_np, pos_np])
            pos_targets[:, 2] -= 1.05
            quat_targets = np.tile(rot_np, (2,1))

            trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_points(
                pos_targets, quat_targets, self.end_effector_name
            )

            articulation_trajectory = ArticulationTrajectory(self._franka_assets[0], trajectory, self.physics_dt)
            self.action_sequence[env_id]    = articulation_trajectory.get_action_sequence()
            self.sequence_step_init[env_id] = step_init
    

    def update_IK_trajectory(self, env_ids, time_step):
        q_target = self.q_target_franka.clone()

        for env_id in env_ids.cpu().tolist():
            step = int((time_step - self.sequence_step_init[env_id]) / 2)

            if step > len(self.action_sequence[env_id]) - 1:
                action = self.action_sequence[env_id][-1]
            else:
                action = self.action_sequence[env_id][step]

            ik_pos = torch.tensor(
                action.joint_positions,
                dtype   = self.articulations["franka"].data.joint_pos.dtype,
                device  = self.articulations["franka"].data.joint_pos.device,
            )

            q_target[env_id] = ik_pos

        return q_target
    

    def is_counter_reached(self, time_step: int):

        t = torch.as_tensor(time_step, device=self.counter.device)

        counter_mask = (self.counter == -1) | (self.counter == t)
        self.counter[counter_mask] = -1

        return counter_mask
    

    def is_target_reached(self, target_pos, target_rot, is_tensor: bool = False, atol=5e-3):
        
        if not is_tensor:
            if target_pos is None or target_rot is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
            # (E, 3), (E, 4)
            target_pos = torch.as_tensor(target_pos, device=self.counter.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1)
            target_rot = torch.as_tensor(target_rot, device=self.counter.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1)


        ee_pos, ee_rot = self.get_curr_poses() # (E, 3), (E, 4)

        pos_err = torch.norm(ee_pos - target_pos, dim=-1)
        rot_err = quat_diff_mag(ee_rot, target_rot)

        is_target_reached = (pos_err < atol) & (rot_err < atol)

        if is_tensor:
            is_target_reached = is_target_reached & (self.request_state == 2)
        
        return is_target_reached
    

    def get_curr_poses(self):

        ee_pos_list = []
        ee_rot_list = []
        for env_id in range(self.num_envs):
            ee_position, ee_rot_mat = self._articulation_kinematics_solvers[env_id].compute_end_effector_pose()
            ee_rot_quat = mat_to_quat(ee_rot_mat)

            ee_pos_list.append(torch.as_tensor(ee_position, device=self.device, dtype=torch.float32))
            ee_rot_list.append(torch.as_tensor(ee_rot_quat, device=self.device, dtype=torch.float32))

        ee_pos = torch.stack(ee_pos_list) # (E, 3)
        ee_rot = torch.stack(ee_rot_list) # (E, 4)

        return ee_pos, ee_rot
    

    def is_obj_clamped(self, thresh = 0.5):
        
        force_L = self.sensors["contact_forces_L"].data.net_forces_w # (E, 1, 3)
        force_R = self.sensors["contact_forces_R"].data.net_forces_w

        force_L_mag = torch.linalg.norm(force_L, dim=-1)
        force_R_mag = torch.linalg.norm(force_R, dim=-1)

        obj_clamped_mask = (force_L_mag > thresh) & (force_R_mag > thresh)    

        return obj_clamped_mask.squeeze(-1)
    
    
    def update_grip_stats(self, env_ids: torch.tensor, is_success: bool = True):
        self.gripped_objs_prev[env_ids] = self.gripped_objs[env_ids]

        force_tensor = self.sensors["contact_forces_L"].data.force_matrix_w # (E, 1, N, 3)
        forces = force_tensor.squeeze(1)        # (E, N, 3)
        mag = torch.linalg.norm(forces, dim=-1) # (E, N)

        values, obj_ids = mag[env_ids, :].max(dim=-1)

        self.gripped_objs[env_ids] = obj_ids

        no_id = (values == 0)
        if no_id.any():
            env_ids_no_id = env_ids[no_id]

            obj_ids_pose = self.get_objs_closest_TCP(env_ids_no_id)         

            self.gripped_objs[env_ids_no_id] = obj_ids_pose


        if is_success:
            self.grip_log[obj_ids.long(), 0] += 1
            self.fail_streaks[env_ids] = 0
            return

        self.grip_log[obj_ids.long(), 1] += 1

        same_obj_mask = (self.gripped_objs[env_ids] == self.gripped_objs_prev[env_ids]) # (len(env_ids), )
        same_obj_envs = env_ids[same_obj_mask]
        diff_obj_envs = env_ids[~same_obj_mask] 
         
        self.fail_streaks[same_obj_envs] += 1
        self.fail_streaks[diff_obj_envs] = 0

    
    def get_objs_closest_TCP(self, env_ids):
        TCP_poses, _ = self.get_curr_poses()    # (E, 3)
        TCP_poses    = TCP_poses[env_ids].unsqueeze(1) # (E, 1, 3)

        obj_poses = self.rigid_object_collections["obj_pool"].root_physx_view.get_transforms() # (E * N, 7)
        obj_poses = obj_poses.view(self.num_envs, -1, 7) # (E, N, 7)
        obj_poses = obj_poses[env_ids, :, :3] # (E, N, 3)

        diffs = torch.norm(obj_poses - TCP_poses, dim=-1)

        idxs = diffs.argmin(dim=1)

        return idxs

    
    def request_DexNet_pred(self, env_ids):

        self.logger.info("Requesting grasp predictions from Dex-Net")

        for env_id in env_ids.cpu().tolist():

            depth_tensor = self.sensors["camera"].data.output["distance_to_image_plane"]
            depth_np = depth_tensor.cpu().numpy()
            depth_np = depth_np[env_id,:,:,:]

            payload = {
                "shape": list(depth_np.shape),
                "dtype": str(depth_np.dtype),
                "data" : base64.b64encode(depth_np.tobytes()).decode()
            }

            req = grequests.post(DexNet.url, json=payload)
            self.DexNet_request[env_id] = grequests.send(req)

            self.request_state[env_id] = 1


    def get_DexNet_pred(self, env_ids, grasps_DexNet):

        for env_id in env_ids.cpu().tolist():
            
            DexNet_request = self.DexNet_request[env_id]

            if DexNet_request is None or not DexNet_request.ready():
                continue

            reply = DexNet_request.value.response.json()
            grasps_np = (
                np.frombuffer(bytes.fromhex(reply["data"]),
                                      dtype=reply["dtype"])
                                   .reshape(reply["shape"])
            )

            self.DexNet_request[env_id] = None
            self.request_state[env_id]       = 2
            
            self.logger.info(f"Received grasp predictions from Dex-Net in {env_id}")

            grasps_DexNet[env_id] = [Grasp(row) for row in grasps_np]

        return grasps_DexNet
    

    def grasps_to_tensors(self, grasps_DexNet, env_ids):

        env_ids = env_ids.cpu().tolist()

        grasps = torch.stack(
            [torch.as_tensor(offset_target_pos(
                            grasps_DexNet[env][0].world_pos),
                            device=self.device,
                            dtype=torch.float)
            for env in env_ids]
        )

        pre_grasps = torch.stack(
            [torch.as_tensor(offset_target_pos(
                            grasps_DexNet[env][0].setup_pos),
                            device=self.device,
                            dtype=torch.float)
            for env in env_ids]
        )

        grasps_rot = torch.stack(
            [torch.as_tensor(grasps_DexNet[env][0].rot_global,
                            device=self.device,
                            dtype=torch.float)
            for env in env_ids]
        )

        return grasps, pre_grasps, grasps_rot


class ObjectGenerator(object):
    def __init__(self, obj_pool: RigidObjectCollection, num_envs, env_origins, device):

        self.obj_pool = obj_pool
        self.view     = obj_pool.root_physx_view 
        self.org_pos  = self.view.get_transforms().clone()

        self.device      = device
        self.num_envs    = num_envs
        self.env_origins = env_origins

        self.obj_ids          = torch.full((self.num_envs, ObjPool.n_objs_ep), -1, dtype=torch.long, device=self.device)
        self.is_obj_id_in_bin = torch.ones(self.num_envs, ObjPool.n_objs_ep, dtype=torch.bool, device=self.device)


    def generate_objects(self, env_ids):

        # Remove previous objects
        if (self.obj_ids[env_ids] != -1).all():
            self.restart_pool(env_ids)

        # Generate random list of objects and view_ids
        obj_ids = torch.stack(
            [torch.tensor(sorted(random.sample(range(ObjPool.n_obj_pool), ObjPool.n_objs_ep)),
                          device=self.device)
            for _ in range(len(env_ids))]
        )
        self.obj_ids[env_ids] = obj_ids # (E, N)

        view_ids = self._env_obj_ids_to_view_ids_abstract(env_ids, obj_ids).to(torch.uint32)

        # Randomise Domain
        self._randomise_friction(view_ids)

        self._randomise_mass(view_ids)

        self._randomise_poses(view_ids, env_ids, obj_ids)

        # Visibility???
        ...


    def is_bin_empty(self, env_ids):

        obj_pool_poses = self.view.get_transforms()
        obj_pool_poses = obj_pool_poses.view(self.num_envs, -1, 7)
        obj_pool_poses = obj_pool_poses[env_ids, ...]

        obj_pool_poses_rel = obj_pool_poses[..., :2] - self.env_origins[env_ids, :2].unsqueeze(1) # (E, N, 2) - # (E, 1, 2)

        objs_outside_bin = ( # (E, N)
            (obj_pool_poses_rel[..., 0] <  ObjPool.pos_x)           |   
            (obj_pool_poses_rel[..., 0] > (ObjPool.pos_x + 0.27))   |
            (obj_pool_poses_rel[..., 1] <  ObjPool.pos_y)           |
            (obj_pool_poses_rel[..., 1] > (ObjPool.pos_y + 0.39))
        )

        bin_empty = objs_outside_bin.all(dim=1)
        return bin_empty


    def remove_objs(self, env_ids, obj_ids):

        view_ids = self._env_obj_ids_to_view_ids_abstract(env_ids, obj_ids).to(torch.uint32)
        object_poses = self.org_pos[view_ids.long()]

        self.write_object_pose_to_sim_abstract(object_poses.view(len(env_ids), -1, 7), view_ids, env_ids, obj_ids)
    

    def restart_pool(self, env_ids):

        self.remove_objs(env_ids, self.obj_ids[env_ids])

    
    def clean_pool(self, env_ids):
        # Get Poses of current objects in bin
        obj_ids       = self.obj_ids[env_ids]
        view_ids_bin  = self._env_obj_ids_to_view_ids_abstract(env_ids, obj_ids).to(torch.uint32)

        obj_pool_poses = self.view.get_transforms()
        obj_bin_poses  = obj_pool_poses[view_ids_bin.long()]
        obj_bin_poses  = obj_bin_poses.view(len(env_ids), ObjPool.n_objs_ep, 7)

        obj_bin_poses_rel = obj_bin_poses[..., :2] - self.env_origins[env_ids, :2].unsqueeze(1) # (E, N, 2) - # (E, 1, 2)

        # Mask out if not in bin
        mask_out = ( # (E, N)
            (obj_bin_poses_rel[..., 0] <  ObjPool.pos_x)           |   
            (obj_bin_poses_rel[..., 0] > (ObjPool.pos_x + 0.27))   |
            (obj_bin_poses_rel[..., 1] <  ObjPool.pos_y)           |
            (obj_bin_poses_rel[..., 1] > (ObjPool.pos_y + 0.39))
        )

        # obj_ids_outside_bin = obj_ids[mask_out]
        # Cannot be done in parallel since different N objects per env possible
        # then obj_ids_outside_bin.shape == (K, ), doesn´t work for -> view_ids + object_state_w
        for i, env_id in enumerate(env_ids):

            if not mask_out[i].any():
                continue

            env_id = env_id.view(-1) 

            obj_ids_outside_bin = obj_ids[i, mask_out[i]].unsqueeze(0) # (N, ) -> (1, N)

            # Get original poses of masked out objects
            view_ids_bin_out     = self._env_obj_ids_to_view_ids_abstract(env_id, obj_ids_outside_bin).to(torch.uint32)
            object_poses_bin_out = self.org_pos[view_ids_bin_out.long()]

            self.write_object_pose_to_sim_abstract(object_poses_bin_out.view(1, -1, 7), view_ids_bin_out, env_id, obj_ids_outside_bin)
        

        self.is_obj_id_in_bin[env_ids] = ~mask_out


    def _randomise_friction(self, view_ids: torch.tensor):

        S = self.view.max_shapes
        N = ObjPool.n_obj_pool * Settings.num_envs # == view.count

        mean_fric = torch.tensor([ObjPool.mu_static,
                                  ObjPool.mu_dynamic],
                                  dtype=torch.float32).view(1, 2).expand(N, 2)
        
        std_fric  = torch.full_like(mean_fric, ObjPool.sigma)
        friction  = torch.normal(mean_fric, std_fric)

        mean_rest   = torch.full((N, 1), ObjPool.mu_rest,  dtype=torch.float32)
        std_rest    = torch.full((N, 1), ObjPool.sigma_rest, dtype=torch.float32)
        restitution = torch.normal(mean_rest, std_rest)
        # TODO: Static > Dynamic
        
        properties = torch.cat([friction, restitution], dim=1) # (N, 3)
        properties = properties.unsqueeze(1)                   # (N, 1, 3)
        properties = properties.expand(N, S, 3).clone()        # (N, S, 3)  
        properties = properties.cpu()

        self.view.set_material_properties(properties, indices=view_ids.cpu())


    def _randomise_mass(self, view_ids: torch.tensor):

        N = ObjPool.n_obj_pool * Settings.num_envs # == view.count

        mean_mass   = torch.full((N, 1), ObjPool.mu_rest,  dtype=torch.float32)
        std_mass    = torch.full((N, 1), ObjPool.sigma_rest, dtype=torch.float32)
        mass        = torch.normal(mean_mass, std_mass).cpu()

        self.view.set_masses(mass, indices=view_ids.cpu())


    def _randomise_poses(self, 
                         view_ids: torch.tensor, 
                         env_ids:  torch.tensor, 
                         obj_ids:  torch.tensor):

        N = ObjPool.n_objs_ep

        xy = torch.stack([
            self._sample_discrete_xy_bins(ObjPool.x_bins, 
                                            ObjPool.y_bins, 
                                            ObjPool.spacing,
                                            ObjPool.pos_x,
                                            ObjPool.pos_y,
                                            N, env_id)
            for env_id in env_ids.cpu().tolist()]) # (E, N, 2)

        z  = torch.empty((len(env_ids), N, 1), device=self.device).uniform_(ObjPool.z_min, ObjPool.z_max) # (E, N, 1)
        
        poses_3d    = torch.cat([xy, z], dim=2)                         # (E, N, 3)

        quartenions = self._sample_uniform_quaternions(N, len(env_ids)) # (E, N, 4)

        object_poses = torch.cat([poses_3d, quartenions], dim=2)        # (E, N, 7)
       
        self.write_object_pose_to_sim_abstract(object_poses, view_ids, env_ids, obj_ids)

    
    def _sample_discrete_xy_bins(self,
                                n_bins_x: int,
                                n_bins_y: int,
                                spacing:  float,
                                x0:       float,
                                y0:       float,
                                N:        int,
                                env_id:   int):
        """
        Pick N unique (x,y) bins on a regular grid.

        Parameters
        ----------
        n_bins_x, n_bins_y : int
            Number of bins along each axis.
        N : int
            How many unique positions you need.
        """
        # TODO: Improve to take in center of Bin and its size

        x0_env = x0 + self.env_origins[env_id, 0]
        y0_env = y0 + self.env_origins[env_id, 1]

        # (n_bins_x · n_bins_y, 2) table of (i,j) integer indices
        ij = torch.stack(torch.meshgrid(
                torch.arange(n_bins_x, device=self.device),
                torch.arange(n_bins_y, device=self.device),
                indexing="ij"), dim=-1
            ).reshape(-1, 2)

        # take N different rows without replacement
        chosen = ij[torch.randperm(ij.size(0), device=self.device)[:N]]

        # convert bin indices to metric coordinates
        xy = (chosen.float() + 0.5) * spacing
        xy += torch.tensor([x0_env, y0_env], device=self.device)
        return xy


    def _sample_uniform_quaternions(self, N: int, E: int):
        """
        Marsaglia 1972 method — uniform over SO(3).

        Returns
        -------
        q : torch.Tensor, shape (E, N, 4)  (w, x, y, z)
        """
        u1, u2, u3 = torch.rand(3, E, N, device=self.device)

        q = torch.zeros((E, N, 4), device=self.device)
        q[..., 0] = torch.sqrt(1.0 - u1) * torch.sin(2 * math.pi * u2)   # x
        q[..., 1] = torch.sqrt(1.0 - u1) * torch.cos(2 * math.pi * u2)   # y
        q[..., 2] = torch.sqrt(      u1) * torch.sin(2 * math.pi * u3)   # z
        q[..., 3] = torch.sqrt(      u1) * torch.cos(2 * math.pi * u3)   # w
        return q[..., (3, 0, 1, 2)]
    

    def _env_obj_ids_to_view_ids_abstract(
        self, env_ids: torch.Tensor, object_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        ...

        Parameters
        ----------
        env_ids: Tensor
            (E,)
        object_ids : Tensor
            (E, N)
        """
        view_ids = (object_ids * self.num_envs + env_ids.unsqueeze(1)).flatten()

        return view_ids


    def write_object_pose_to_sim_abstract(
        self,
        object_pose: torch.Tensor,
        view_ids,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):

        # set into internal buffers
        self.obj_pool._data.object_state_w[env_ids[:, None], object_ids, :7] = object_pose.clone()
        
        # convert the quaternion from wxyz to xyzw
        poses_xyzw = self.obj_pool._data.object_state_w[..., :7].clone()
        poses_xyzw[..., 3:] = math_utils.convert_quat(poses_xyzw[..., 3:], to="xyzw")

        # set into simulation
        self.view.set_transforms(self.obj_pool.reshape_data_to_view(poses_xyzw), indices=view_ids)