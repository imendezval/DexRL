import os
import time

import numpy as np

from gqcnn.grasping import RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage, Point)
from visualization import Visualizer2D as vis

from constants import DexNet, Camera


class DexNetWrapper(object):

    def __init__(self):

        self.logger = Logger.get_logger("Dex-Net Server")

        self.dir_ = os.path.dirname(os.path.realpath(__file__))
        self.dir_gqcnn = os.path.join(self.dir_, "gqcnn")


        self._get_filenames()  
        
        self.grasp_count = 1

        self.camera_intr = CameraIntrinsics.load(self.camera_intr_path)
        self.segmask = BinaryImage.open(self.segmask_path)

        config = YamlConfig(self.config_path)
        self.inpaint_rescale_factor = config["inpaint_rescale_factor"]
        self.policy_config = config["policy"]

        self.policy_config["metric"]["gqcnn_model"] = self.model_path


        self.policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_height"] = Camera.height
        self.policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_width"] = Camera.width
        
        self.policy = FullyConvolutionalGraspingPolicyParallelJaw(self.policy_config)


    def __call__(self, depth_im, vis = False):
        
        self.state, rgbd_im = self.create_state(depth_im)

        grasps = self.predict_grasps()
        if vis:
            self.visualize_grasps(grasps, rgbd_im)

        self.grasp_count += 1

        return grasps
    

    def _get_filenames(self):

        self.segmask_path        = os.path.join(self.dir_, DexNet.segmask_path)
        self.camera_intr_path    = os.path.join(self.dir_, DexNet.cam_intr_path)
        self.config_path         = os.path.join(self.dir_, DexNet.cfg_path)
        self.model_path          = os.path.join(self.dir_, DexNet.model_path)
        

    def create_state(self, depth_data):

        #depth_data = np.load(state)
        depth_im = DepthImage(depth_data, frame=self.camera_intr.frame)
        color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                        3]).astype(np.uint8),
                                frame=self.camera_intr.frame)


        valid_px_mask = depth_im.invalid_pixel_mask().inverse()
        self.segmask = self.segmask.mask_binary(valid_px_mask)

        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)


        # Create state
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        return RgbdImageState(rgbd_im, self.camera_intr, segmask=self.segmask), rgbd_im


    def predict_grasps(self):

        # Inference
        policy_start = time.time()
        actions = self.policy._action(self.state, num_actions=10)
        self.logger.info("Planning took %.3f sec" % (time.time() - policy_start))

        return sorted(actions, key=lambda a: a.q_value, reverse=True)
    

    def reformat_grasps(self, actions):

        rows = []
        for action in actions:
            grasp_center = Point(np.array([*action.grasp.center]), frame=self.camera_intr.frame)

            point3D = self.camera_intr.deproject_pixel(action.grasp.depth, grasp_center)
            
            rows.append([*point3D.data, action.grasp.angle, action.q_value])
        
        rows = np.asarray(rows, dtype=np.float32)
        return rows
    

    def visualize_grasps(self, actions, rgbd_im):

        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth,
                    vmin=self.policy_config["vis"]["vmin"],
                    vmax=self.policy_config["vis"]["vmax"])
        # for action in actions:
            # vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.grasp(actions[0].grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
            actions[0].grasp.depth, actions[0].q_value))
        vis.show(filename = f"./DexNet/grasp_{self.grasp_count}")