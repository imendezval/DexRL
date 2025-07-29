# ─────────
# Startup
# ─────────
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s', # %(asctime)s
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

import gevent

import isaaclab.sim as sim_utils
from scene import FrankaScene, FrankaSceneCfg, ObjectGenerator

from helpers import offset_target_pos
from constants import Settings, Poses


Poses.setup   = offset_target_pos(Poses.setup_TCP)
Poses.lift    = offset_target_pos(Poses.lift_TCP)
Poses.inter   = offset_target_pos(Poses.inter_TCP)
Poses.drop    = offset_target_pos(Poses.drop_TCP)

# TODO: Check Franka Arm reaches and rotations
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
    
    ObjGen = ObjectGenerator(scene["obj_pool"], 
                                      num_envs = scene.num_envs,
                                      env_origins = scene.env_origins,
                                      device = scene.device)

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
            #gripper_def_pos = joint_pos_gripper[0][0].item()

            scene.reset()
            scene.logger.info("Franka Reset.")

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
        # 9     Restart cycle


        DEXNET_MASK = (scene.request_state == 1)
        if DEXNET_MASK.any():
            env_ids = DEXNET_MASK.nonzero(as_tuple=False).flatten()

            grasps_DexNet = scene.get_DexNet_pred(env_ids, grasps_DexNet)

        gevent.sleep(0)


        REGENERATE_MASK = (scene.mode == -2)
        if REGENERATE_MASK.any():
            env_ids = REGENERATE_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"Object Generator started in {env_ids.cpu().tolist()}")

            ObjGen.generate_objects(env_ids)
            scene.counter[REGENERATE_MASK] = step_count + 120 # Wait - cleaning pool

            scene.mode[REGENERATE_MASK] = -1

        
        CLEAN_MASK = (scene.mode == -1) & scene.is_counter_reached(step_count)
        if CLEAN_MASK.any():
            env_ids = CLEAN_MASK.nonzero(as_tuple=False).flatten()

            ObjGen.clean_pool(env_ids)
            scene.counter[CLEAN_MASK] = step_count + 6 # Wait - DexNet scan

            scene.mode[env_ids] = 0


        SETUP_MASK = (scene.mode == 0) & scene.is_counter_reached(step_count)
        if SETUP_MASK.any():
            env_ids = SETUP_MASK.nonzero(as_tuple=False).flatten()

            scene.request_DexNet_pred(env_ids)
            scene.q_target_franka = scene.compute_IK(Poses.setup, Poses.base_rot, env_ids)

            scene.mode[SETUP_MASK] = 1


        PRE_GRASP_MASK = ((scene.mode == 1) &
                          scene.is_target_reached(Poses.setup, Poses.base_rot) &
                          (scene.request_state == 2))
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

            scene.compute_IK_trajectory(scene.GRASPS, scene.PRE_GRASPS, scene.GRASPS_ROT, env_ids, step_count)

            scene.mode[GRASP_MASK]    = 3 
            scene.counter[GRASP_MASK] = step_count + 120 # Wait - Clamp sequence fail check

        
        UPDATE_TRAYECTORY_MASK = (scene.mode == 3)
        if UPDATE_TRAYECTORY_MASK.any():
            env_ids = UPDATE_TRAYECTORY_MASK.nonzero(as_tuple=False).flatten()
            
            scene.q_target_franka = scene.update_IK_trajectory(env_ids, step_count)


        CLOSE_GRIPPER_MASK = (scene.mode == 3) & scene.is_target_reached(scene.GRASPS, scene.GRASPS_ROT, is_tensor=True, atol=5e-3)
        if CLOSE_GRIPPER_MASK.any():
            env_ids = CLOSE_GRIPPER_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"GRASP position reached - Closing Gripper in {env_ids.cpu().tolist()}.")

            scene.q_target_yumi[CLOSE_GRIPPER_MASK] = 0
            scene.yumi_vel_vec[CLOSE_GRIPPER_MASK]  = -1

            scene.mode[CLOSE_GRIPPER_MASK] = 4


        LIFT_MASK = (scene.mode == 4) & scene.is_obj_clamped()
        if LIFT_MASK.any():
            env_ids = LIFT_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"Object gripped in {env_ids.cpu().tolist()}.")

            scene.q_target_franka = scene.compute_IK(scene.PRE_GRASPS, scene.GRASPS_ROT, env_ids, is_tensor=True)

            scene.mode[LIFT_MASK]    = 5
            scene.counter[LIFT_MASK] = -1


        ### FAILURES ###
        FAIL_GRIP_MASK = (
                (((scene.mode == 3) | (scene.mode == 4)) & scene.is_counter_reached(step_count)) |
                ((scene.mode == 5) & (~scene.is_obj_clamped(thresh=0.1)))
        )
        if FAIL_GRIP_MASK.any(): # GRASP POSITION NOT REACHED / CLAMP WAS BAD (didnt reach pre-grasp)
            env_ids = FAIL_GRIP_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"FAIL - GRIP in {env_ids.cpu().tolist()} - RESTARTING SEQUENCE.")

            # Bookkeep obj_id grip fail + fail streaks
            scene.update_grip_stats(env_ids, is_success=False)

            # Open Gripper and retract safe distance
            scene.q_target_yumi[FAIL_GRIP_MASK] = 0.05
            scene.yumi_vel_vec[FAIL_GRIP_MASK]  = 1
            scene.q_target_franka = scene.compute_IK(scene.PRE_GRASPS, scene.GRASPS_ROT, env_ids, is_tensor=True)

            scene.mode[FAIL_GRIP_MASK]    = 8
            scene.counter[FAIL_GRIP_MASK] = step_count + 30 # Wait - retract safe distance

        
        FAIL_DROP_MASK = (
                ((scene.mode == 6) | (scene.mode == 7)) &
                (~scene.is_obj_clamped(thresh=0.1))
        )
        if FAIL_DROP_MASK.any(): # OBJECT DROPPED EARLY
            env_ids = FAIL_DROP_MASK.nonzero(as_tuple=False).flatten()
            
            scene.logger.info(f"FAIL - DROP in {env_ids.cpu().tolist()} - RESTARTING SEQUENCE.")
            
            # Bookkeep obj_id grip fail + fail streaks
            scene.update_grip_stats(env_ids, is_success=False)

            # Open Gripper and retract safe distance
            scene.q_target_yumi[FAIL_DROP_MASK] = 0.05
            scene.yumi_vel_vec[FAIL_DROP_MASK]  = 1

            ee_poses, ee_rots = scene.get_curr_poses()
            ee_poses[:, 2] += 0.15
            scene.q_target_franka = scene.compute_IK(ee_poses, ee_rots, env_ids, is_tensor=True)


            scene.mode[FAIL_DROP_MASK]    = 8
            scene.counter[FAIL_DROP_MASK] = step_count + 30 # Wait - retract safe distance
        

        REMOVE_OBJ_MASK = (FAIL_GRIP_MASK | FAIL_DROP_MASK) & (scene.fail_streaks == 2)
        if REMOVE_OBJ_MASK.any():
            env_ids = REMOVE_OBJ_MASK.nonzero(as_tuple=False).flatten()
            
            scene.logger.info(f"Object grip failed 3 consecutive times in {env_ids.cpu().tolist()} - removing.")
            ...

            ObjGen.remove_objs(env_ids, scene.gripped_objs[env_ids].unsqueeze(1)) # (E, 1)


        INTER_MASK = (scene.mode == 5) & scene.is_target_reached(scene.PRE_GRASPS, scene.GRASPS_ROT, is_tensor=True, atol=1e-2)
        if INTER_MASK.any():
            env_ids = INTER_MASK.nonzero(as_tuple=False).flatten()
        
            scene.logger.info(f"LIFT position reached in {env_ids.cpu().tolist()}.")

            scene.q_target_franka = scene.compute_IK(Poses.inter, Poses.base_rot, env_ids)

            scene.mode[INTER_MASK] = 6


        DROP_MASK = (scene.mode == 6) & scene.is_target_reached(Poses.inter, Poses.base_rot, atol=1e-2)
        if DROP_MASK.any():
            env_ids = DROP_MASK.nonzero(as_tuple=False).flatten()
            
            scene.logger.info(f"INTER position reached in {env_ids.cpu().tolist()}.")

            scene.q_target_franka = scene.compute_IK(Poses.drop, Poses.base_rot, env_ids)

            scene.mode[DROP_MASK] = 7
            

        OPEN_GRIPPER_MASK = (scene.mode == 7) & scene.is_target_reached(Poses.drop, Poses.base_rot, atol=1e-2)
        if OPEN_GRIPPER_MASK.any():
            env_ids = OPEN_GRIPPER_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"DROP position reached - Opening Gripper in {env_ids.cpu().tolist()}.")

            # Bookkeep obj_id grip success
            scene.update_grip_stats(env_ids, is_success=True)

            scene.q_target_yumi[OPEN_GRIPPER_MASK] = 0.05
            scene.yumi_vel_vec[OPEN_GRIPPER_MASK]  = 1

            scene.counter[OPEN_GRIPPER_MASK] = step_count + 50

            scene.mode[OPEN_GRIPPER_MASK] = 8

        
        INTER_MASK2 = (scene.mode == 8) & scene.is_counter_reached(step_count)
        if INTER_MASK2.any():
            env_ids = INTER_MASK2.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"Heading to INTER to restart sequence in {env_ids.cpu().tolist()}.")
            
            scene.q_target_franka = scene.compute_IK(Poses.inter, Poses.base_rot, env_ids)

            scene.mode[INTER_MASK2] = 9


        RESTART_MASK = (scene.mode == 9) & scene.is_target_reached(Poses.inter, Poses.base_rot)
        if RESTART_MASK.any():
            env_ids = RESTART_MASK.nonzero(as_tuple=False).flatten()

            scene.logger.info(f"Restarting Picking Sequence in {env_ids.cpu().tolist()}.")

            scene.GRASPS[RESTART_MASK]        = 0
            scene.PRE_GRASPS[RESTART_MASK]    = 0
            scene.GRASPS_ROT[RESTART_MASK]    = 0
            scene.request_state[RESTART_MASK] = 0


            EMPTY_BIN_MASK = ObjGen.is_bin_empty(env_ids)
            env_ids_empty_bin = env_ids[EMPTY_BIN_MASK]
            env_ids_full_bin  = env_ids[~EMPTY_BIN_MASK]

            scene.mode[env_ids_full_bin]  = 0
            scene.mode[env_ids_empty_bin] = -2

            scene.fail_streaks[env_ids_empty_bin] = 0
            scene.gripped_objs[env_ids_empty_bin] = -1

            # TODO: Reposition bin if moved + regenerate?
            # TODO: q of action = 0 case
            # TODO: Gripper friction unpaired


        # apply actions to gripper
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