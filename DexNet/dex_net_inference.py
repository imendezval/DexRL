import json
import os
import time

import numpy as np

from gqcnn.grasping import RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw

from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage)
from visualization import Visualizer2D as vis


logger = Logger.get_logger("dex_net_inference.py")

dir_ = os.path.dirname(os.path.realpath(__file__))
dir_gqcnn = os.path.join(dir_, "gqcnn")

model_name           = "models/FC-GQCNN-4.0-PJ"
depth_im_filename    = "data/examples/clutter/phoxi/fcgqcnn/depth_4.npy" #"depth_img_1.npy"
segmask_filename     = "data/examples/clutter/phoxi/fcgqcnn/segmask_4.png" # "segmask.png"
camera_intr_filename = "data/calib/phoxi/phoxi.intr"


model_path          = os.path.join(dir_gqcnn, model_name)
depth_im_path       = os.path.join(dir_gqcnn, depth_im_filename)
segmask_path        = os.path.join(dir_gqcnn, segmask_filename)
camera_intr_path    = os.path.join(dir_gqcnn, camera_intr_filename)



camera_intr = CameraIntrinsics.load(camera_intr_path)

config_filename = os.path.join(dir_gqcnn, "cfg/examples/fc_gqcnn_pj.yaml")

config = YamlConfig(config_filename)
inpaint_rescale_factor = config["inpaint_rescale_factor"]
policy_config = config["policy"]

policy_config["metric"]["gqcnn_model"] = model_path



depth_data = np.load(depth_im_path)
depth_im = DepthImage(depth_data, frame=camera_intr.frame)
color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,
                                3]).astype(np.uint8),
                        frame=camera_intr.frame)


segmask = BinaryImage.open(segmask_path)
valid_px_mask = depth_im.invalid_pixel_mask().inverse()
segmask = segmask.mask_binary(valid_px_mask)

depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)


# Create state
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)


# Visualization
if "input_images" in policy_config["vis"] and policy_config["vis"][
        "input_images"]:
    vis.figure(size=(10, 10))
    num_plot = 1
    if segmask is not None:
        num_plot = 2
    vis.subplot(1, num_plot, 1)
    vis.imshow(depth_im)
    if segmask is not None:
        vis.subplot(1, num_plot, 2)
        vis.imshow(segmask)
    vis.show()


policy_config["metric"]["fully_conv_gqcnn_config"][
    "im_height"] = depth_im.shape[0]
policy_config["metric"]["fully_conv_gqcnn_config"][
    "im_width"] = depth_im.shape[1]


# Inference
policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)

policy_start = time.time()
actions = policy._action(state, num_actions=50)
logger.info("Planning took %.3f sec" % (time.time() - policy_start))

# Sorting
actions_sorted = sorted(actions, key=lambda a: a.q_value, reverse=True)

for action in actions_sorted:
    print(action.grasp.angle, action.grasp.center)
# actions_sorted[0].grasp .center .angle .depth .width

# Visualization
if policy_config["vis"]["final_grasp"]:
    vis.figure(size=(10, 10))
    vis.imshow(rgbd_im.depth,
                vmin=policy_config["vis"]["vmin"],
                vmax=policy_config["vis"]["vmax"])
    for action in actions:
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
    vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
        actions[0].grasp.depth, actions[0].q_value))
    vis.show()