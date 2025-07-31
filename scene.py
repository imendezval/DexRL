import numpy as np
np.set_printoptions(suppress=True)

import os, types, logging
import base64, grequests, gevent
import random, math
import torch


import isaaclab.sim as sim_utils

# Interactive Scene Class
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# CFGs
from cfg.franka_cfg import FRANKA_PANDA_HIGH_PD_CFG
from cfg.gripper_cfg import YUMI_CFG
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.sensors import ContactSensorCfg

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
from constants import Camera, Prims, RobotArm, DexNet, Poses
from ObjectGenerator import ObjectGenerator
ObjPool = Prims.ObjPool

dir_ = os.path.dirname(os.path.realpath(__file__))


ObjPool_filter_paths = [
    f"{{ENV_REGEX_NS}}/obj_{i}"
    for i in range(ObjPool.n_obj_pool)
]

class ObjectPoolCfg(object):

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
                rigid_objects=ObjectPoolCfg().object_pool_cfg
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
                                        filter_prim_paths_expr = ObjPool_filter_paths)
    contact_forces_R = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_r", 
                                        filter_prim_paths_expr = ObjPool_filter_paths)



class FrankaScene(InteractiveScene):
    def __init__(self, cfg: InteractiveSceneCfg):
        super().__init__(cfg)

        self.logger = logging.getLogger(__name__)

        self.step_count = 0

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
        self.grasps_DexNet = [None for _ in range(self.num_envs)]
        self.GRASPS        = torch.zeros(self.num_envs, 3, device=self.device)
        self.PRE_GRASPS    = torch.zeros(self.num_envs, 3, device=self.device)
        self.GRASPS_ROT    = torch.zeros(self.num_envs, 4, device=self.device)

        # Bookkeeping
        self.gripped_objs      = torch.full((self.num_envs,), -1, 
                                        dtype=torch.long, device=self.device)
        self.gripped_objs_prev = torch.full((self.num_envs,), -1, 
                                        dtype=torch.long, device=self.device)
        self.grip_log          = torch.zeros(ObjPool.n_obj_pool, 2, 
                                        dtype=torch.uint8, device=self.device)
        self.fail_streaks      = torch.zeros(self.num_envs,
                                        dtype=torch.uint8, device=self.device)
        
    
    def update(self, dt):
        super().update(dt)
        self._fsm_tick()
        self.step_count += 1


    def setup_post_load(self):

        for env_path in self.env_prim_paths:
            arm_path     = env_path + self._arm_path_loc
            gripper_path = env_path + self._gripper_path_loc

            self._attach_gripper(arm_path, gripper_path)
            #sim_schemas.activate_contact_sensors(gripper_path)

        self._load_InverseKinematics()


    def setup_post_reset(self):
        # Initialize franka SingleArticulation(s) for IK
        self.franka_initialize()

        # Initialize ObjectGenerator
        self.ObjGen = ObjectGenerator(self.rigid_object_collections["obj_pool"], 
                                        num_envs = self.num_envs,
                                        env_origins = self.env_origins,
                                        device = self.device)
        
        # Initialize Articulation q_target tensors
        self.q_target_franka = self.articulations["franka"].data.default_joint_pos.clone()

        self.q_target_yumi = torch.full_like(self.articulations["yumi_gripper"].data.joint_pos, 0.05)
        self.yumi_vel_vec  = torch.tensor([[0.0, 0.0]],
                                            device=self.articulations["yumi_gripper"].data.joint_pos.device
                                            ).repeat(self.num_envs, 1)

    
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


    def get_DexNet_pred(self, env_ids):

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

            self.grasps_DexNet[env_id] = [Grasp(row) for row in grasps_np]
    

    def grasps_to_tensors(self, env_ids):

        env_ids = env_ids.cpu().tolist()

        grasps = torch.stack(
            [torch.as_tensor(offset_target_pos(
                            self.grasps_DexNet[env][0].world_pos),
                            device=self.device,
                            dtype=torch.float)
            for env in env_ids]
        )

        pre_grasps = torch.stack(
            [torch.as_tensor(offset_target_pos(
                            self.grasps_DexNet[env][0].setup_pos),
                            device=self.device,
                            dtype=torch.float)
            for env in env_ids]
        )

        grasps_rot = torch.stack(
            [torch.as_tensor(self.grasps_DexNet[env][0].rot_global,
                            device=self.device,
                            dtype=torch.float)
            for env in env_ids]
        )

        return grasps, pre_grasps, grasps_rot
    

    def reset(self, env_ids): # TODO: Vectorize to multiple envs???
        super().reset(env_ids)

        # reset joint states to defaults
        root_state_franka = self.articulations["franka"].data.default_root_state.clone()
        root_state_gripper = self.articulations["yumi_gripper"].data.default_root_state.clone()

        # offset entity root states by origin
        root_state_franka[:, :3] += self.env_origins
        root_state_gripper[:, :3] += self.env_origins

        self.articulations["franka"].write_root_pose_to_sim(root_state_franka[:, :7])
        self.articulations["franka"].write_root_velocity_to_sim(root_state_franka[:, 7:])

        # yumi_gripper pos established by robot assembler
        #self.articulations["yumi_gripper"].write_root_pose_to_sim(root_state_gripper[:, :7])
        self.articulations["yumi_gripper"].write_root_velocity_to_sim(root_state_gripper[:, 7:])

        # set default joint position
        joint_pos_franka, joint_vel_franka = (
            self.articulations["franka"].data.default_joint_pos.clone(),
            self.articulations["franka"].data.default_joint_vel.clone(),
        )

        self.articulations["franka"].write_joint_state_to_sim(joint_pos_franka, joint_vel_franka)

        joint_pos_gripper, joint_vel_gripper = (
            self.articulations["yumi_gripper"].data.default_joint_pos,
            self.articulations["yumi_gripper"].data.default_joint_vel,
        ) 
        self.articulations["yumi_gripper"].write_joint_state_to_sim(joint_pos_gripper, joint_vel_gripper)

        self.logger.info("Franka Reset.")

        self.step_count = 0

    
    def _fsm_tick(self):
        # ──────────────────
        # Picking Sequence
        # ──────────────────

        # -2    regenerate
        # -1    clean
        # 0     SETUP, Request
        # 1     PRE_GRASP
        # 2     GRASP
        # 3     Close Gripper
        # 4     LIFT
        # 5     INTER, identify obj_ids
        # 6     DROP
        # 7     Open Gripper
        # 8     INTER
        # 9     Restart cycle - regenerate (-2) or not (0)


        DEXNET_MASK = (self.request_state == 1)
        if DEXNET_MASK.any():
            env_ids = DEXNET_MASK.nonzero(as_tuple=False).flatten()

            self.get_DexNet_pred(env_ids)

        gevent.sleep(0)


        REGENERATE_MASK = (self.mode == -2)
        if REGENERATE_MASK.any():
            env_ids = REGENERATE_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"Object Generator started in {env_ids.cpu().tolist()}")

            self.ObjGen.generate_objects(env_ids)
            self.counter[REGENERATE_MASK] = self.step_count + 120 # Wait - cleaning pool

            self.mode[REGENERATE_MASK] = -1

        
        CLEAN_MASK = (self.mode == -1) & self.is_counter_reached(self.step_count)
        if CLEAN_MASK.any():
            env_ids = CLEAN_MASK.nonzero(as_tuple=False).flatten()

            self.ObjGen.clean_pool(env_ids)
            self.counter[CLEAN_MASK] = self.step_count + 6 # Wait - DexNet scan

            self.mode[env_ids] = 0


        SETUP_MASK = (self.mode == 0) & self.is_counter_reached(self.step_count)
        if SETUP_MASK.any():
            env_ids = SETUP_MASK.nonzero(as_tuple=False).flatten()

            self.request_DexNet_pred(env_ids)
            self.q_target_franka = self.compute_IK(Poses.setup, Poses.base_rot, env_ids)

            self.mode[SETUP_MASK] = 1


        PRE_GRASP_MASK = ((self.mode == 1) &
                          self.is_target_reached(Poses.setup, Poses.base_rot) &
                          (self.request_state == 2))
        if PRE_GRASP_MASK.any():
            env_ids = PRE_GRASP_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"SETUP position reached in {env_ids.cpu().tolist()}.")
            
            grasps, pre_grasps, grasps_rot = self.grasps_to_tensors(env_ids)

            self.GRASPS[PRE_GRASP_MASK]     = grasps
            self.PRE_GRASPS[PRE_GRASP_MASK] = pre_grasps
            self.GRASPS_ROT[PRE_GRASP_MASK] = grasps_rot

            self.q_target_franka = self.compute_IK(self.PRE_GRASPS, self.GRASPS_ROT, env_ids, is_tensor=True)

            self.mode[PRE_GRASP_MASK] = 2


        GRASP_MASK = (self.mode == 2) & self.is_target_reached(self.PRE_GRASPS, self.GRASPS_ROT, is_tensor=True)
        if GRASP_MASK.any():
            env_ids = GRASP_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"PRE-GRASP position reached in {env_ids.cpu().tolist()}.")

            self.compute_IK_trajectory(self.GRASPS, self.PRE_GRASPS, self.GRASPS_ROT, env_ids, self.step_count)

            self.mode[GRASP_MASK]    = 3 
            self.counter[GRASP_MASK] = self.step_count + 120 # Wait - Clamp sequence fail check

        
        UPDATE_TRAYECTORY_MASK = (self.mode == 3)
        if UPDATE_TRAYECTORY_MASK.any():
            env_ids = UPDATE_TRAYECTORY_MASK.nonzero(as_tuple=False).flatten()
            
            self.q_target_franka = self.update_IK_trajectory(env_ids, self.step_count)


        CLOSE_GRIPPER_MASK = (self.mode == 3) & self.is_target_reached(self.GRASPS, self.GRASPS_ROT, is_tensor=True, atol=5e-3)
        if CLOSE_GRIPPER_MASK.any():
            env_ids = CLOSE_GRIPPER_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"GRASP position reached - Closing Gripper in {env_ids.cpu().tolist()}.")

            self.q_target_yumi[CLOSE_GRIPPER_MASK] = 0
            self.yumi_vel_vec[CLOSE_GRIPPER_MASK]  = -1

            self.mode[CLOSE_GRIPPER_MASK] = 4


        LIFT_MASK = (self.mode == 4) & self.is_obj_clamped()
        if LIFT_MASK.any():
            env_ids = LIFT_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"Object gripped in {env_ids.cpu().tolist()}.")

            self.q_target_franka = self.compute_IK(self.PRE_GRASPS, self.GRASPS_ROT, env_ids, is_tensor=True)

            self.mode[LIFT_MASK]    = 5
            self.counter[LIFT_MASK] = -1


        ### FAILURES ###
        FAIL_GRIP_MASK = (
                (((self.mode == 3) | (self.mode == 4)) & self.is_counter_reached(self.step_count)) |
                ((self.mode == 5) & (~self.is_obj_clamped(thresh=0.1)))
        )
        if FAIL_GRIP_MASK.any(): # GRASP POSITION NOT REACHED / CLAMP WAS BAD (didnt reach pre-grasp)
            env_ids = FAIL_GRIP_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"FAIL - GRIP in {env_ids.cpu().tolist()} - RESTARTING SEQUENCE.")

            # Bookkeep obj_id grip fail + fail streaks
            self.update_grip_stats(env_ids, is_success=False)

            # Open Gripper and retract safe distance
            self.q_target_yumi[FAIL_GRIP_MASK] = 0.05
            self.yumi_vel_vec[FAIL_GRIP_MASK]  = 1
            self.q_target_franka = self.compute_IK(self.PRE_GRASPS, self.GRASPS_ROT, env_ids, is_tensor=True)

            self.mode[FAIL_GRIP_MASK]    = 8
            self.counter[FAIL_GRIP_MASK] = self.step_count + 30 # Wait - retract safe distance

        
        FAIL_DROP_MASK = (
                ((self.mode == 6) | (self.mode == 7)) &
                (~self.is_obj_clamped(thresh=0.1))
        )
        if FAIL_DROP_MASK.any(): # OBJECT DROPPED EARLY
            env_ids = FAIL_DROP_MASK.nonzero(as_tuple=False).flatten()
            
            self.logger.info(f"FAIL - DROP in {env_ids.cpu().tolist()} - RESTARTING SEQUENCE.")
            
            # Bookkeep obj_id grip fail + fail streaks
            self.update_grip_stats(env_ids, is_success=False)

            # Open Gripper and retract safe distance
            self.q_target_yumi[FAIL_DROP_MASK] = 0.05
            self.yumi_vel_vec[FAIL_DROP_MASK]  = 1

            ee_poses, ee_rots = self.get_curr_poses()
            ee_poses[:, 2] += 0.15
            self.q_target_franka = self.compute_IK(ee_poses, ee_rots, env_ids, is_tensor=True)


            self.mode[FAIL_DROP_MASK]    = 8
            self.counter[FAIL_DROP_MASK] = self.step_count + 30 # Wait - retract safe distance
        

        REMOVE_OBJ_MASK = (FAIL_GRIP_MASK | FAIL_DROP_MASK) & (self.fail_streaks == 2)
        if REMOVE_OBJ_MASK.any():
            env_ids = REMOVE_OBJ_MASK.nonzero(as_tuple=False).flatten()
            
            self.logger.info(f"Object grip failed 3 consecutive times in {env_ids.cpu().tolist()} - removing.")
            ...

            self.ObjGen.remove_objs(env_ids, self.gripped_objs[env_ids].unsqueeze(1)) # (E, 1)


        INTER_MASK = (self.mode == 5) & self.is_target_reached(self.PRE_GRASPS, self.GRASPS_ROT, is_tensor=True, atol=1e-2)
        if INTER_MASK.any():
            env_ids = INTER_MASK.nonzero(as_tuple=False).flatten()
        
            self.logger.info(f"LIFT position reached in {env_ids.cpu().tolist()}.")

            self.q_target_franka = self.compute_IK(Poses.inter, Poses.base_rot, env_ids)

            self.mode[INTER_MASK] = 6


        DROP_MASK = (self.mode == 6) & self.is_target_reached(Poses.inter, Poses.base_rot, atol=1e-2)
        if DROP_MASK.any():
            env_ids = DROP_MASK.nonzero(as_tuple=False).flatten()
            
            self.logger.info(f"INTER position reached in {env_ids.cpu().tolist()}.")

            self.q_target_franka = self.compute_IK(Poses.drop, Poses.base_rot, env_ids)

            self.mode[DROP_MASK] = 7
            

        OPEN_GRIPPER_MASK = (self.mode == 7) & self.is_target_reached(Poses.drop, Poses.base_rot, atol=1e-2)
        if OPEN_GRIPPER_MASK.any():
            env_ids = OPEN_GRIPPER_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"DROP position reached - Opening Gripper in {env_ids.cpu().tolist()}.")

            # Bookkeep obj_id grip success
            self.update_grip_stats(env_ids, is_success=True)

            self.q_target_yumi[OPEN_GRIPPER_MASK] = 0.05
            self.yumi_vel_vec[OPEN_GRIPPER_MASK]  = 1

            self.counter[OPEN_GRIPPER_MASK] = self.step_count + 50

            self.mode[OPEN_GRIPPER_MASK] = 8

        
        INTER_MASK2 = (self.mode == 8) & self.is_counter_reached(self.step_count)
        if INTER_MASK2.any():
            env_ids = INTER_MASK2.nonzero(as_tuple=False).flatten()

            self.logger.info(f"Heading to INTER to restart sequence in {env_ids.cpu().tolist()}.")
            
            self.q_target_franka = self.compute_IK(Poses.inter, Poses.base_rot, env_ids)

            self.mode[INTER_MASK2] = 9


        RESTART_MASK = (self.mode == 9) & self.is_target_reached(Poses.inter, Poses.base_rot)
        if RESTART_MASK.any():
            env_ids = RESTART_MASK.nonzero(as_tuple=False).flatten()

            self.logger.info(f"Restarting Picking Sequence in {env_ids.cpu().tolist()}.")

            self.GRASPS[RESTART_MASK]        = 0
            self.PRE_GRASPS[RESTART_MASK]    = 0
            self.GRASPS_ROT[RESTART_MASK]    = 0
            self.request_state[RESTART_MASK] = 0


            EMPTY_BIN_MASK = self.ObjGen.is_bin_empty(env_ids)
            env_ids_empty_bin = env_ids[EMPTY_BIN_MASK]
            env_ids_full_bin  = env_ids[~EMPTY_BIN_MASK]

            self.mode[env_ids_full_bin]  = 0
            self.mode[env_ids_empty_bin] = -2

            self.fail_streaks[env_ids_empty_bin] = 0
            self.gripped_objs[env_ids_empty_bin] = -1

            # TODO: Reposition bin if moved + regenerate?
            # TODO: q of action = 0 case
            # TODO: Gripper friction unpaired

        # apply actions to gripper
        self.articulations["yumi_gripper"].set_joint_velocity_target(self.yumi_vel_vec)
        self.articulations["yumi_gripper"].set_joint_position_target(self.q_target_yumi)       
        self.articulations["franka"].set_joint_position_target(self.q_target_franka)

    