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

import isaaclab.sim as sim_utils
from scene import FrankaScene, FrankaSceneCfg

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

    step_count = 0
    while simulation_app.is_running():
        
        if step_count == 0:
            scene.reset(scene.env_ids)

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
    scene.setup_post_reset()

    scene.logger.info("Setup complete.")
    pick_cube(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()