import numpy as np
np.set_printoptions(suppress=True)
import torch
import random, math

import isaaclab.utils.math as math_utils

from isaaclab.assets import RigidObjectCollection

from constants import Camera, Prims, Settings
ObjPool = Prims.ObjPool


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
            (obj_pool_poses_rel[..., 0] < (Camera.x_cam - 0.135))           |   
            (obj_pool_poses_rel[..., 0] > (Camera.x_cam + 0.135))   |
            (obj_pool_poses_rel[..., 1] < (Camera.y_cam - 0.195))           |
            (obj_pool_poses_rel[..., 1] > (Camera.y_cam + 0.195))
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
            (obj_bin_poses_rel[..., 0] < (Camera.x_cam - 0.135))           |   
            (obj_bin_poses_rel[..., 0] > (Camera.x_cam + 0.135))           |
            (obj_bin_poses_rel[..., 1] < (Camera.y_cam - 0.195))           |
            (obj_bin_poses_rel[..., 1] > (Camera.y_cam + 0.195))
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