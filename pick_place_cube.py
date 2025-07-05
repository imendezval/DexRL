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

from pick_utils import Grasp, offset_target_pos


# ─────────────────────
# Predetermined Poses
# ─────────────────────
# offset_local = np.array([0.0, 0.0, 0.3135])
# offset_local = np.array([0.0, 0.0, 0.137])
# offset_local = np.array([0.0, 0.0, 0.035])

offset_local    = np.array([0.0, 0.0, 0.02])

target_rot      = np.array([0, 0.3826834 ,0.9238795,0], dtype=np.float32) # 180 down, 45 z
TP_setup_TCP    = np.array([0.6, 0.3, 1.25], dtype=np.float32)
TP_setup_W      = offset_target_pos(TP_setup_TCP, target_rot, offset_local)

TP_pick_TCP     = np.array([0.6, 0.3, 1.05], dtype=np.float32)
TP_pick_W       = offset_target_pos(TP_pick_TCP, target_rot, offset_local)

TP_lift_TCP     = np.array([0.6, 0.3, 1.25], dtype=np.float32)
TP_lift_W       = offset_target_pos(TP_lift_TCP, target_rot, offset_local)

TP_inter_TCP    = np.array([0.5, 0.0, 1.3], dtype=np.float32)
TP_inter_W      = offset_target_pos(TP_inter_TCP, target_rot, offset_local)

TP_drop_TCP     = np.array([0.6, -0.3, 1.2], dtype=np.float32)
TP_drop_W       = offset_target_pos(TP_drop_TCP, target_rot, offset_local)




# ────────────────────
# Main simulator loop
# ────────────────────
def pick_cube(sim: sim_utils.SimulationContext, scene: FrankaScene):
    sim_dt = sim.get_physics_dt()

    global TP_pick_W
    TP_setup2_W = None
    target_rot_pick = None

    scene.franka_initialize()
    q_target = scene["franka"].data.default_joint_pos

    step_count = 0
    reset_dt = 5000

    object_width = 0.015
    gripper_pose = object_width + 0.005

    grasps = None

    while simulation_app.is_running():
        
        if step_count % reset_dt == 0:
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
            print("[INFO] Franka reset.")

        # ──────────────────
        # Picking Sequence
        # ──────────────────

        if step_count == 20:
            q_target = scene.compute_IK(TP_setup_W, target_rot)
            scene.request_DexNet_pred()

        #print(scene.areq)
        if scene.is_request:
            grasps = scene.get_DexNet_pred()

        # if grasps is not None:
        #     for grasp in grasps:
        #         print(grasp)

        gevent.sleep(0)

        ### SETUP
        if scene.is_target_reached(TP_setup_W, target_rot) and scene.mode == 0 and grasps is not None:
            scene.logger.info("Setup position reached.")

            TP_setup2_W = offset_target_pos(grasps[0].setup_pos, target_rot, offset_local)
            target_rot_pick = grasps[0].rot_global
            for grasp in grasps:
                print(grasp)
            q_target = scene.compute_IK(TP_setup2_W, grasps[0].rot_global)

            scene.mode = 1

        if scene.is_target_reached(TP_setup2_W, target_rot_pick) and scene.mode == 1:
            scene.logger.info("Setup2 position reached.")
            TP_pick_W = offset_target_pos(grasps[0].world_pos, target_rot, offset_local)
            q_target = scene.compute_IK(TP_pick_W, grasps[0].rot_global)

            scene.mode = 2

        ### PICK
        if scene.is_target_reached(TP_pick_W, target_rot_pick, atol=5e-3):
            scene.mode = 3
            scene.logger.info("Predicted grasp pose reached - closing gripper.")

        ### GRIP - Close Gripper
        if scene.mode == 3:
            gripper_pose = object_width
            gripper_close_vec = torch.tensor([[-1.0,-1.0]],device=scene["yumi_gripper"].data.joint_pos.device).repeat(scene.num_envs, 1) 
            scene["yumi_gripper"].set_joint_velocity_target(gripper_close_vec)
        
        ### GRIP
        if scene.is_obj_clamped() and scene.mode == 3:
            scene.mode = 4
            q_target = scene.compute_IK(TP_lift_W, target_rot)
            scene.logger.info("Object gripped.")

        ### LIFT
        if scene.is_target_reached(TP_lift_W, target_rot, atol=1e-2) and scene.mode == 4:
            scene.mode = 5
            q_target = scene.compute_IK(TP_inter_W, target_rot)

        ### INTER
        if scene.is_target_reached(TP_inter_W, target_rot, atol=1e-2):
            q_target = scene.compute_IK(TP_drop_W, target_rot)

            scene.mode = 6 
            
        ### DROP
        if scene.is_target_reached(TP_drop_W, target_rot, atol=1e-2) and scene.mode == 6:
            scene.logger.info("Drop position reached.")
            
            scene.request_DexNet_pred()

            scene.mode = 7

        ### Drop - Open Gripper
        if scene.mode == 7:
            gripper_pose = gripper_def_pos
            gripper_close_vec = torch.tensor([[1.0,1.0]],device=scene["yumi_gripper"].data.joint_pos.device).repeat(scene.num_envs, 1) 
            scene["yumi_gripper"].set_joint_velocity_target(gripper_close_vec)

        # apply actions to robot
        q_target_yumi =  torch.full_like(scene["yumi_gripper"].data.joint_pos, gripper_pose)
        scene["yumi_gripper"].set_joint_position_target(q_target_yumi)       
        scene["franka"].set_joint_position_target(q_target)   

        # print(scene["franka"].data.joint_pos)   
        # print(q_target)

        # print(scene.mode)  


        ### Write to sim + update
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)    
        step_count += 1

        #scene["camera"].data.output["distance_to_image_plane"].shape


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Nice overhead camera
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = FrankaSceneCfg(num_envs=1, env_spacing=2.0)
    scene = FrankaScene(scene_cfg)
    scene.setup_post_load()

    sim.reset() 
    print("[INFO] Setup complete… running!")
    pick_cube(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()