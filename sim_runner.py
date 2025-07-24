# ─────────
# Startup
# ─────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

from gevent import monkey
monkey.patch_all()

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-arm Franka demo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────
# Imports
# ─────────
import numpy as np
np.set_printoptions(suppress=True)
import torch
# from isaaclab.utils import convert_to_torch

import gevent

import isaaclab.sim as sim_utils
from scene import FrankaScene, FrankaSceneCfg

from helpers import Grasp, offset_target_pos
from constants import Settings, Poses


Poses.setup   = offset_target_pos(Poses.setup_TCP)
Poses.lift    = offset_target_pos(Poses.lift_TCP)
Poses.inter   = offset_target_pos(Poses.inter_TCP)
Poses.drop    = offset_target_pos(Poses.drop_TCP)

# TODO: Scale to work independently with different environments
# TODO: Check Franka Arm reaches and rotations
# TODO: Detect if object has been lifted (/bin moved?)
# TODO: Update SegMask
def pick_cube(sim: sim_utils.SimulationContext, scene: FrankaScene):
    sim_dt = sim.get_physics_dt()

    scene.franka_initialize()

    scene.q_target_franka = scene["franka"].data.default_joint_pos.clone()

    gripper_pose = 0.05
    scene.q_target_yumi = torch.full_like(scene["yumi_gripper"].data.joint_pos, gripper_pose)
    scene.yumi_vel_vec  = torch.tensor([[0.0, 0.0]],
                                        device=scene["yumi_gripper"].data.joint_pos.device
                                        ).repeat(scene.num_envs, 1)


    grasps_DexNet = [None for _ in range(scene.num_envs)]

    step_count = 0
    while simulation_app.is_running():
        
        if step_count == 0:
            # reset joint states to defaults
            root_state_franka = scene["franka"].data.default_root_state.clone()
            root_state_gripper = scene["yumi_gripper"].data.default_root_state.clone()

            # offset entity root states by origin
            root_state_franka[:, :3] += scene.env_origins
            root_state_gripper[:, :3] += scene.env_origins

            scene["franka"].write_root_pose_to_sim(root_state_franka[:, :7])
            scene["franka"].write_root_velocity_to_sim(root_state_franka[:, 7:])

            # yumi_gripper pos established by robot assembler
            #scene["yumi_gripper"].write_root_pose_to_sim(root_state_gripper[:, :7])
            scene["yumi_gripper"].write_root_velocity_to_sim(root_state_gripper[:, 7:])

            # set default joint position
            joint_pos_franka, joint_vel_franka = (
                scene["franka"].data.default_joint_pos.clone(),
                scene["franka"].data.default_joint_vel.clone(),
            )

            scene["franka"].write_joint_state_to_sim(joint_pos_franka, joint_vel_franka)

            joint_pos_gripper, joint_vel_gripper = (
               scene["yumi_gripper"].data.default_joint_pos,
               scene["yumi_gripper"].data.default_joint_vel,
            ) 
            scene["yumi_gripper"].write_joint_state_to_sim(joint_pos_gripper, joint_vel_gripper)
            gripper_def_pos = joint_pos_gripper[0][0].item()

            scene.reset()
            scene.logger.info("Franka Reset.")

        # ──────────────────
        # Picking Sequence
        # ──────────────────

        # -1    regenerate, start ounter
        # 0     counter - request, SETUP IK
        # 1     PRE_GRASP IK
        # 2     GRASP IK
        # 3     Close Gripper
        # 4     LIFT  IK
        # 5     INTER IK
        # 6     DROP  IK
        # 7     Open Gripper
        # 8     INTER  IK
        # 9     Restart cycle


        # Tensor:
        # is_request, counter, mode
        # Functions:
        # request_DexNet_pred, compute_IK, is_target_reached, is_obj_clamped, is_counter_reached, generate_objects
        #

        DEXNET_MASK = (scene.IS_GRASP == 1)
        if DEXNET_MASK.any():
            env_ids = DEXNET_MASK.nonzero(as_tuple=False).flatten()

            grasps_DexNet = scene.get_DexNet_pred(env_ids, grasps_DexNet)

        gevent.sleep(0)

        REGENERATE_MASK = (scene.mode == -1)
        if REGENERATE_MASK.any():
            env_ids = REGENERATE_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"Object Generator started in {env_ids.cpu().tolist()}")

            scene.generate_objects(env_ids)
            scene.counter[REGENERATE_MASK] = step_count + 100

            scene.mode[REGENERATE_MASK] = 0


        SETUP_MASK = (scene.mode == 0) & scene.is_counter_reached(step_count)
        if SETUP_MASK.any():
            env_ids = SETUP_MASK.nonzero(as_tuple=False).flatten()

            scene.request_DexNet_pred(env_ids)
            scene.q_target_franka = scene.compute_IK(Poses.setup, Poses.base_rot, env_ids)

            scene.mode[SETUP_MASK] = 1


        PRE_GRASP_MASK = ((scene.mode == 1) &
                          scene.is_target_reached(Poses.setup, Poses.base_rot) &
                          (scene.IS_GRASP == 2))
        if PRE_GRASP_MASK.any():
            env_ids = PRE_GRASP_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"SETUP position reached in {env_ids.cpu().tolist()}.")
            
            grasps, pre_grasps, grasps_rot = scene.grasps_to_tensors(grasps_DexNet, env_ids)

            scene.GRASPS[PRE_GRASP_MASK]     = grasps
            scene.PRE_GRASPS[PRE_GRASP_MASK] = pre_grasps
            scene.GRASPS_ROT[PRE_GRASP_MASK] = grasps_rot

            scene.q_target_franka = scene.compute_IK(scene.PRE_GRASPS, scene.GRASPS_ROT, env_ids, is_tensor=True)

            scene.mode[PRE_GRASP_MASK] = 2


        GRASP_MASK = (scene.mode == 2) & scene.is_target_reached(scene.PRE_GRASPS, scene.GRASPS_ROT, is_tensor=True)
        if GRASP_MASK.any():
            env_ids = GRASP_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"PRE-GRASP position reached in {env_ids.cpu().tolist()}.")

            scene.q_target_franka = scene.compute_IK(scene.GRASPS, scene.GRASPS_ROT, env_ids, is_tensor=True)

            scene.mode[GRASP_MASK] = 3

        ### Close Gripper
        CLOSE_GRIPPER_MASK =  (scene.mode == 3) & scene.is_target_reached(scene.GRASPS, scene.GRASPS_ROT, is_tensor=True, atol=5e-3)
        if CLOSE_GRIPPER_MASK.any():
            env_ids = CLOSE_GRIPPER_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"GRASP position reached - Closing Gripper in {env_ids.cpu().tolist()}.")

            scene.q_target_yumi[CLOSE_GRIPPER_MASK] = 0
            scene.yumi_vel_vec[CLOSE_GRIPPER_MASK]  = -1 # broadcasted

            scene.mode[CLOSE_GRIPPER_MASK] = 4
        

        LIFT_MASK = (scene.mode == 4) & scene.is_obj_clamped()
        if LIFT_MASK.any():
            env_ids = LIFT_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"Object gripped in {env_ids.cpu().tolist()}.")

            scene.q_target_franka = scene.compute_IK(scene.PRE_GRASPS, scene.GRASPS_ROT, env_ids, is_tensor=True)

            scene.mode[LIFT_MASK] = 5


        INTER_MASK = (scene.mode == 5) & scene.is_target_reached(scene.PRE_GRASPS, scene.GRASPS_ROT, is_tensor=True, atol=1e-2)
        if INTER_MASK.any():
            env_ids = INTER_MASK.nonzero(as_tuple=False).flatten()
        
            scene.logger.info("LIFT position reached.")

            scene.q_target_franka = scene.compute_IK(Poses.inter, Poses.base_rot, env_ids)

            scene.mode[INTER_MASK] = 6


        DROP_MASK = (scene.mode == 6) & scene.is_target_reached(Poses.inter, Poses.base_rot, atol=1e-2)
        if DROP_MASK.any():
            env_ids = DROP_MASK.nonzero(as_tuple=False).flatten()
            
            scene.logger.info("INTER position reached.")

            scene.q_target_franka = scene.compute_IK(Poses.drop, Poses.base_rot, env_ids)

            scene.mode[DROP_MASK] = 7
            

        OPEN_GRIPPER_MASK = (scene.mode == 7) & scene.is_target_reached(Poses.drop, Poses.base_rot, atol=1e-2)
        if OPEN_GRIPPER_MASK.any():
            env_ids = OPEN_GRIPPER_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info("DROP position reached - Opening Gripper.")

            scene.q_target_yumi[OPEN_GRIPPER_MASK] = 0.05
            scene.yumi_vel_vec[OPEN_GRIPPER_MASK]  = 1

            scene.counter[OPEN_GRIPPER_MASK] = step_count + 50

            scene.mode[OPEN_GRIPPER_MASK] = 8

        
        INTER_MASK2 = (scene.mode == 8) & scene.is_counter_reached(step_count)
        if INTER_MASK2.any():
            env_ids = INTER_MASK2.nonzero(as_tuple=False).flatten()

            scene.logger.info("Object Dropped.")
            
            scene.q_target_franka = scene.compute_IK(Poses.inter, Poses.base_rot, env_ids)

            scene.mode[INTER_MASK2] = 9


        RESTART_MASK = (scene.mode == 9) & scene.is_target_reached(Poses.inter, Poses.base_rot)
        if RESTART_MASK.any():
            env_ids = RESTART_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info("Restarting Picking Sequence.")

            scene.mode[RESTART_MASK]       = 0 ##################### 
            scene.PRE_GRASPS[RESTART_MASK] = 0
            scene.GRASPS_ROT[RESTART_MASK] = 0
            scene.IS_GRASP[RESTART_MASK]   = 0
            scene.GRASPS[RESTART_MASK]     = 0

        # apply actions to gripper
        #q_target_yumi =  torch.full_like(scene["yumi_gripper"].data.joint_pos, gripper_pose)
        scene["yumi_gripper"].set_joint_velocity_target(scene.yumi_vel_vec)
        scene["yumi_gripper"].set_joint_position_target(scene.q_target_yumi)       
        scene["franka"].set_joint_position_target(scene.q_target_franka)

        # print(scene.mode)

        ### Write to sim + update
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)    
        step_count += 1


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Nice overhead camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = FrankaSceneCfg(num_envs=Settings.num_envs, env_spacing=Settings.env_spacing)
    scene = FrankaScene(scene_cfg)
    scene.setup_post_load()

    sim.reset() 
    scene.logger.info("Setup complete.")
    pick_cube(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()