import numpy as np
np.set_printoptions(suppress=True)

import os, types, logging
import base64, grequests
import torch
import random, math


import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils

import omni.physics.tensors.impl.api as physx

# Interactive Scene Class
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# CFGs
from cfg.franka_cfg import FRANKA_PANDA_HIGH_PD_CFG
from cfg.gripper_cfg import YUMI_CFG
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg#, Articulation
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


from helpers import Grasp, mat_to_quat
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

        self._arm_path_loc = "/Franka/franka_instanceable"
        self._gripper_path_loc = "/yumi_gripper/yumi_gripper"

        self._franka_assets = []
        self._kinematics_solver = None
        self._articulation_kinematics_solvers = []

        self.mode = -1
        self.counter = None

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
    

    def is_counter_reached(self, time_step):
        if self.counter is None:
            return True
        
        if time_step == self.counter:
            self.counter = None
            return True 
        
        else: return False
    

    def is_target_reached(self, target_pos, target_rot, atol=5e-3):

        if target_pos is None or target_rot is None:
            return False

        ee_position, ee_rot_mat = self._articulation_kinematics_solvers[0].compute_end_effector_pose()
        ee_rot = mat_to_quat(ee_rot_mat)

        is_target_reached = np.allclose(ee_position, target_pos, atol=atol) \
                        and np.allclose(ee_rot, target_rot, atol=atol)
        
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
    
    
    def generate_objects(self, env_id):
        # (remove 20 previous)
        ...

        N = ObjPool.n_objs_ep

        # Generate list of 20
        obj_list = sorted(
                    random.sample(range(0, ObjPool.n_obj_pool), N))

        # Get Physx View
        obj_pool    = self.rigid_object_collections["obj_pool"]
        view        = obj_pool.root_physx_view

        env_id  = torch.tensor([env_id], device=self.device)
        obj_ids = torch.tensor(obj_list, device=self.device)

        view_ids = obj_pool._env_obj_ids_to_view_ids(env_id, obj_ids).to(torch.uint32).cpu()

        # Randomise Domain
        self._randomise_friction(view, view_ids)

        self._randomise_mass(view, view_ids)

        self._randomise_poses(N, env_id, obj_ids)

        # Visibility???
        ...


    def _randomise_friction(self, view: physx.RigidBodyView, view_ids):

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

        view.set_material_properties(properties, indices=view_ids)


    def _randomise_mass(self, view: physx.RigidBodyView, view_ids):

        N = ObjPool.n_obj_pool * Settings.num_envs # == view.count

        mean_mass   = torch.full((N, 1), ObjPool.mu_rest,  dtype=torch.float32)
        std_mass    = torch.full((N, 1), ObjPool.sigma_rest, dtype=torch.float32)
        mass        = torch.normal(mean_mass, std_mass).cpu()

        view.set_masses(mass, indices=view_ids)


    def _randomise_poses(self, N, env_id, obj_ids):

        xy = self._sample_discrete_xy_bins(ObjPool.x_bins, 
                                           ObjPool.y_bins, 
                                           ObjPool.spacing,
                                           ObjPool.pos_x,
                                           ObjPool.pos_y,
                                           N, env_id)

        z  = torch.empty((N, 1), device=self.device).uniform_(ObjPool.z_min, ObjPool.z_max)
        
        poses_3d    = torch.cat([xy, z], dim=1)           # (N, 3)

        quartenions = self._sample_uniform_quaternions(N) # (N, 4)

        object_poses = torch.cat([poses_3d, quartenions], dim=1) # (N, 7)
        object_poses = object_poses.unsqueeze(0)                 # (1, N, 7)
       
        self.rigid_object_collections["obj_pool"].write_object_pose_to_sim(object_poses, env_id, obj_ids)
    
    
    def _sample_discrete_xy_bins(self,
                                n_bins_x: int,
                                n_bins_y: int,
                                spacing: float,
                                x0: float,
                                y0: float,
                                N: int,
                                env_id):
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


    def _sample_uniform_quaternions(self, N: int):
        """
        Marsaglia 1972 method — uniform over SO(3).

        Returns
        -------
        q : torch.Tensor, shape (N, 4)  (w, x, y, z)
        """
        u1, u2, u3 = torch.rand(3, N, device=self.device)

        q = torch.zeros((N, 4), device=self.device)
        q[:, 0] = torch.sqrt(1.0 - u1) * torch.sin(2 * math.pi * u2)   # x
        q[:, 1] = torch.sqrt(1.0 - u1) * torch.cos(2 * math.pi * u2)   # y
        q[:, 2] = torch.sqrt(     u1) * torch.sin(2 * math.pi * u3)    # z
        q[:, 3] = torch.sqrt(     u1) * torch.cos(2 * math.pi * u3)    # w
        return q[:, (3, 0, 1, 2)]
    

    def move_obj(self, env_id):

        rigid_obj = self.rigid_objects["cube"].root_physx_view
        env_id             = torch.tensor([env_id], dtype=torch.uint32)  # env_idx.to(torch.uint32)
        mat                = rigid_obj.get_material_properties()
        mat[env_id, :, :2] = torch.tensor([0, 0],
                                           dtype=mat.dtype,
                                           device=mat.device)    
        rigid_obj.set_material_properties(mat, env_id)