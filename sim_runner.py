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


def pick_cube(sim: sim_utils.SimulationContext, scene: FrankaScene):
    sim_dt = sim.get_physics_dt()

    GRASP       = None
    GRASP_ROT   = None
    PRE_GRASP   = None

    scene.franka_initialize()
    q_target = scene["franka"].data.default_joint_pos

    step_count = 0

    gripper_pose = 0.05

    grasps = None

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

        # -1    regenerate, cstart ounter
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


        if scene.is_request:
            grasps = scene.get_DexNet_pred()

        gevent.sleep(0)

        if scene.mode == -1:
            scene.generate_objects(0)
            scene.counter = step_count + 100

            scene.mode = 0

        ### SETUP
        if scene.mode == 0 and scene.is_counter_reached(step_count):

            q_target = scene.compute_IK(Poses.setup, Poses.base_rot)
            scene.request_DexNet_pred()

            scene.mode = 1

        ### PRE-GRASP
        if scene.mode == 1 and scene.is_target_reached(Poses.setup, Poses.base_rot) and grasps is not None:
            scene.logger.info("SETUP position reached.")

            GRASP     = offset_target_pos(grasps[0].world_pos)
            PRE_GRASP = offset_target_pos(grasps[0].setup_pos)
            GRASP_ROT = grasps[0].rot_global
            grasps    = None

            q_target = scene.compute_IK(PRE_GRASP, GRASP_ROT)

            scene.mode = 2

        ### GRASP
        if scene.mode == 2 and scene.is_target_reached(PRE_GRASP, GRASP_ROT):
            scene.logger.info("PRE-GRASP position reached.")

            q_target = scene.compute_IK(GRASP, GRASP_ROT)

            scene.mode = 3

        ### Close Gripper
        if scene.mode == 3 and scene.is_target_reached(GRASP, GRASP_ROT, atol=5e-3):
            scene.logger.info("GRASP position reached - Closing Gripper.")

            gripper_pose = 0

            gripper_close_vec = torch.tensor([[-1.0,-1.0]],
                                             device=scene["yumi_gripper"].data.joint_pos.device
                                             ).repeat(scene.num_envs, 1)
            scene["yumi_gripper"].set_joint_velocity_target(gripper_close_vec)

            scene.mode = 4
        
        ### LIFT
        if scene.mode == 4 and scene.is_obj_clamped():
            scene.logger.info("Object gripped.")

            q_target = scene.compute_IK(PRE_GRASP, GRASP_ROT)

            scene.mode = 5

        ### INTER
        if scene.mode == 5 and scene.is_target_reached(PRE_GRASP, GRASP_ROT, atol=1e-2):
            scene.logger.info("LIFT position reached.")

            q_target = scene.compute_IK(Poses.inter, Poses.base_rot)

            scene.mode = 6

        ### DROP
        if scene.mode == 6 and scene.is_target_reached(Poses.inter, Poses.base_rot, atol=1e-2):
            scene.logger.info("INTER position reached.")

            q_target = scene.compute_IK(Poses.drop, Poses.base_rot)

            scene.mode = 7
            
        ### Open Gripper
        if scene.mode == 7 and scene.is_target_reached(Poses.drop, Poses.base_rot, atol=1e-2):
            scene.logger.info("DROP position reached - Opening Gripper.")

            gripper_pose = gripper_def_pos
            gripper_close_vec = torch.tensor([[1.0,1.0]],
                                             device=scene["yumi_gripper"].data.joint_pos.device
                                             ).repeat(scene.num_envs, 1) 
            scene["yumi_gripper"].set_joint_velocity_target(gripper_close_vec)

            scene.counter = step_count + 10

            scene.mode = 8

        ### INTER
        if scene.mode == 8 and scene.is_counter_reached(step_count):
            scene.logger.info("Object Dropped.")
            
            q_target = scene.compute_IK(Poses.inter, Poses.base_rot)

            scene.mode = 9

        ### Restart Sequence
        if scene.mode == 9 and scene.is_target_reached(Poses.inter, Poses.base_rot):
            scene.logger.info("Restarting Picking Sequence.")

            scene.mode = 0

        # apply actions to gripper
        q_target_yumi =  torch.full_like(scene["yumi_gripper"].data.joint_pos, gripper_pose)
        scene["yumi_gripper"].set_joint_position_target(q_target_yumi)       
        scene["franka"].set_joint_position_target(q_target)

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