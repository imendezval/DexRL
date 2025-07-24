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
from isaaclab.assets.articulation import ArticulationCfg
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
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation import interface_config_loader
from isaacsim.core.prims import SingleArticulation


from helpers import Grasp, mat_to_quat, quat_diff_mag, offset_target_pos
from constants import Camera, Prims, RobotArm, DexNet, Settings

ObjPool = Prims.ObjPool



dir_ = os.path.dirname(os.path.realpath(__file__))


# @configclass
# class XformCfg(SpawnerCfg):
#     func = staticmethod(lambda prim_path, cfg,
#                         translation=None, orientation=None:
#                         prim_utils.create_prim(prim_path, "Xform",
#                                                translation, orientation))

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

    # cube = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cube",
    #     spawn=sim_utils.MeshCuboidCfg(
    #         size=(0.03, 0.03, 0.03),
    #         physics_material=sim_utils.materials.RigidBodyMaterialCfg(
    #             dynamic_friction=1.0, static_friction=1.0, restitution=0.05 # match with gripper
    #         ),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.3, 1.15)),
    # )
    
    # Object Pool
    # helpers_folder = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/ObjPool",
    #     spawn=XformCfg()
    # )

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

    contact_forces_L = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_l")
    contact_forces_R = ContactSensorCfg(prim_path = "{ENV_REGEX_NS}/yumi_gripper/yumi_gripper/gripper_finger_r")



class FrankaScene(InteractiveScene):
    def __init__(self, cfg: InteractiveSceneCfg):
        super().__init__(cfg)

        self.logger = logging.getLogger(__name__)

        self._arm_path_loc      = "/Franka/franka_instanceable"
        self._gripper_path_loc  = "/yumi_gripper/yumi_gripper"

        self._franka_assets     = []
        self._kinematics_solver = None
        self._articulation_kinematics_solvers = []

        self.q_target_franka   = None #self.articulations["franka"]._data.default_joint_pos.clone()
        self.q_target_yumi     = None
        self.yumi_vel_vec      = None

        self.mode    = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.counter = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        self.DexNet_request = [None for _ in range(self.num_envs)]

        self.GRASPS     = torch.zeros(self.num_envs, 3, device=self.device)
        self.PRE_GRASPS = torch.zeros(self.num_envs, 3, device=self.device)
        self.GRASPS_ROT = torch.zeros(self.num_envs, 4, device=self.device)
        self.IS_GRASP  = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


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
            # print(franka_path)
            franka_single = SingleArticulation(franka_path)
            self._franka_assets.append(franka_single)

            articulation_IK_solver = ArticulationKinematicsSolver(franka_single, self._kinematics_solver, end_effector_name)
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


    def compute_IK(self, target_pos, target_rot, env_ids, is_tensor = False):

        q_target = self.q_target_franka.clone()

        for env_id in env_ids.cpu().tolist():

            robot_base_translation, robot_base_orientation = np.array([0.0, 0.0, 1.05]), np.array([1.0, 0.0, 0.0, 0.0])
            self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

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
    

    def is_counter_reached(self, time_step: int):

        t = torch.as_tensor(time_step, device=self.counter.device)

        counter_mask = (self.counter == -1) | (self.counter == t)
        self.counter[counter_mask] = -1

        return counter_mask
    

    def is_target_reached(self, target_pos, target_rot, is_tensor = False, atol=5e-3):

        
        if not is_tensor:
            if target_pos is None or target_rot is None:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            
            # (E, 3), (E, 4)
            target_pos = torch.as_tensor(target_pos, device=self.counter.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1)
            target_rot = torch.as_tensor(target_rot, device=self.counter.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1)

        ee_pos_list = []
        ee_rot_list = []
        for env_id in range(self.num_envs):
            ee_position, ee_rot_mat = self._articulation_kinematics_solvers[env_id].compute_end_effector_pose()
            ee_rot_quat = mat_to_quat(ee_rot_mat)
            # print(f"ENV_ID:{env_id}: {ee_position}")

            ee_pos_list.append(torch.as_tensor(ee_position, device=self.device, dtype=torch.float32))
            ee_rot_list.append(torch.as_tensor(ee_rot_quat, device=self.device, dtype=torch.float32))

        ee_pos = torch.stack(ee_pos_list) # (E, 3)
        ee_rot = torch.stack(ee_rot_list) # (E, 4)

        pos_err = torch.norm(ee_pos - target_pos, dim=-1)
        rot_err = quat_diff_mag(ee_rot, target_rot)

        is_target_reached = (pos_err < atol) & (rot_err < atol)

        if is_tensor:
            is_target_reached = is_target_reached & (self.IS_GRASP == 2)
        
        return is_target_reached


    def is_obj_clamped(self, thresh = 0.5):
        
        force_L = self.sensors["contact_forces_L"].data.net_forces_w
        force_R = self.sensors["contact_forces_R"].data.net_forces_w

        force_L_mag = torch.linalg.norm(force_L, dim=-1)
        force_R_mag = torch.linalg.norm(force_R, dim=-1)

        obj_clamped_mask = (force_L_mag > thresh) & (force_R_mag > thresh)    

        return obj_clamped_mask.squeeze(-1)

    
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

            self.IS_GRASP[env_id] = 1


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
            self.IS_GRASP[env_id]       = 2
            
            self.logger.info(f"Successfully received grasp predictions from Dex-Net for environment {env_id}")

            grasps_DexNet[env_id] = [Grasp(row) for row in grasps_np]

        return grasps_DexNet
    

    def grasps_to_tensors(self, grasps_DexNet, env_ids):

        env_ids = env_ids.cpu().tolist()
        # print(grasps_DexNet)
        # print(f"ENV IDS: {env_ids}")

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
    
    
    def generate_objects(self, env_ids):
        # (remove 20 previous)
        ...

        N = ObjPool.n_objs_ep

        # Generate list of objects and view_ids
        obj_ids = torch.stack(
            [torch.tensor(sorted(random.sample(range(ObjPool.n_obj_pool), N)),
                          device=self.device)
            for _ in range(len(env_ids))]
        )
        #env_ids     = torch.tensor(env_ids, device=self.device)

        view_ids    = self._env_obj_ids_to_view_ids_abstract(env_ids, obj_ids).to(torch.uint32)


        # Get Physx View
        obj_pool    = self.rigid_object_collections["obj_pool"]
        view        = obj_pool.root_physx_view


        # Randomise Domain
        self._randomise_friction(view, view_ids)

        self._randomise_mass(view, view_ids)

        self._randomise_poses(obj_pool, view, view_ids, env_ids, obj_ids)

        # Visibility???
        ...

    
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
        

    def _randomise_friction(self, view: physx.RigidBodyView, view_ids: torch.tensor):

        S = view.max_shapes
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

        view.set_material_properties(properties, indices=view_ids.cpu())


    def _randomise_mass(self, view: physx.RigidBodyView, view_ids: torch.tensor):

        N = ObjPool.n_obj_pool * Settings.num_envs # == view.count

        mean_mass   = torch.full((N, 1), ObjPool.mu_rest,  dtype=torch.float32)
        std_mass    = torch.full((N, 1), ObjPool.sigma_rest, dtype=torch.float32)
        mass        = torch.normal(mean_mass, std_mass).cpu()

        view.set_masses(mass, indices=view_ids.cpu())


    def _randomise_poses(self, 
                         obj_pool: RigidObjectCollection, 
                         view:     physx.RigidBodyView, 
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
       
        self.write_object_pose_to_sim_abstract(obj_pool, view, object_poses, view_ids, env_ids, obj_ids)

    
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
    

    def write_object_pose_to_sim_abstract(
        self,
        obj_pool: RigidObjectCollection,
        view: physx.RigidBodyView,
        object_pose: torch.Tensor,
        view_ids,
        env_ids: torch.Tensor | None = None,
        object_ids: slice | torch.Tensor | None = None,
    ):

        # set into internal buffers
        obj_pool._data.object_state_w[env_ids[:, None], object_ids, :7] = object_pose.clone() ######
        
        # convert the quaternion from wxyz to xyzw
        poses_xyzw = obj_pool._data.object_state_w[..., :7].clone()
        poses_xyzw[..., 3:] = math_utils.convert_quat(poses_xyzw[..., 3:], to="xyzw")

        # set into simulation
        view.set_transforms(obj_pool.reshape_data_to_view(poses_xyzw), indices=view_ids)


    def move_obj(self, env_id):

        rigid_obj = self.rigid_objects["cube"].root_physx_view
        env_id             = torch.tensor([env_id], dtype=torch.uint32)  # env_idx.to(torch.uint32)
        mat                = rigid_obj.get_material_properties()
        mat[env_id, :, :2] = torch.tensor([0, 0],
                                           dtype=mat.dtype,
                                           device=mat.device)    
        rigid_obj.set_material_properties(mat, env_id)