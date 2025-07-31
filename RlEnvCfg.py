from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, DirectRLEnv

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from scene import FrankaSceneCfg, FrankaScene

from isaaclab.managers import ObservationTermCfg, ActionTermCfg

import torch


class DexNetCrop(ObservationTermCfg):
    func = lambda scene, *_: scene.sensors["camera"].data.output["distance_to_image_plane"][:,None,:,:]  # (E,1,H,W)


class OffsetActionCfg(ActionTermCfg): 
    size=4; 
    func=scene.apply_grasp_offset


class DexRLEnvCfg(ManagerBasedRLEnvCfg):
    scene = FrankaSceneCfg
    decimation = 1
    episode_length = 800


    observations = { "DexNetCrop": DexNetCrop() }
    actions = {
        "offset": {
            "size": 4,
            "lower": [-0.02, -0.02, -0.02, -0.26],
            "upper": [ 0.02,  0.02,  0.02,  0.26],
        },
    }
    rewards = {
        "sparse_pick": {
            "func": lambda scene, *_: scene.is_obj_clamped().float()
                                   - scene.pick_failed.float()
        },
        "offset_penalty": {
            "func": lambda scene, action, *_: -0.1 * (action ** 2).sum(dim=-1)
        },
    }
    terminations = {
        "pick_done": {
            "func": lambda scene, *_: scene.mode.eq(8) | scene.pick_failed
        }
    }


class PickEnv(ManagerBasedRLEnv):
    cfg = DexRLEnvCfg()

    def __init__(self, cfg):

        InteractiveSceneOriginal = InteractiveScene

        import isaaclab.scene
        isaaclab.scene.InteractiveScene = FrankaScene
        super.__init__(cfg)

        isaaclab.scene.InteractiveScene = InteractiveSceneOriginal
        # self.scene = FrankaScene(self.cfg.scene)


    # called once per *RL* step
    def _pre_physics_step(self, actions: torch.Tensor):
        # we only want the first action of the episode
        not_applied = self.scene.mode.eq(1) & (~getattr(self, "_action_applied", False))
        if not_applied.any():
            self.scene.apply_grasp_offset(actions[not_applied])
            self._action_applied = True

    def reset(self, *args, **kwargs):
        self._action_applied = False
        return super().reset(*args, **kwargs)
