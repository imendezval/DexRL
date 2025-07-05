import numpy as np

class Camera:
    height          = 386
    width           = 516

    f_x             = 552.5
    f_y             = 552.5
    c_x             = 255.5
    c_y             = 191.75
    intr_matrix     = [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]

    x_cam           = 0.6
    y_cam           = 0.3
    z_cam           = 1.77
    pos             = np.array([x_cam, y_cam, z_cam])
    rot             = np.array([0, 0.7071, 0.7071, 0])


class Prims:
    class Table:
        path        = "assets/table_instanceable.usd"
        pos         = (0.55, 0.0, 1.05)
        rot         = (0.7071, 0.0, 0.0, 0.7071)

    class KLT_Bin:
        path        = "assets/small_KLT.usd"
        pos_pick    = (0.6, 0.3, 1.12)
        pos_place   = (0.6, -0.3, 1.12)


class RobotArm:
    class FrankaArm:
        path        = "assets/franka/franka_no_gripper.usd"
        pos         = (0.0, 0.0, 1.05)
    
    class Gripper:
        path        = "assets/franka/yumi_gripper_mimic_mu.usd"
        offset_pos  = np.array([0.0, 0.0, 0.1068])


class Poses:
    basic_rot       = np.array([0, 0.3826834 ,0.9238795,0])

class DexNet:
    host            = "0.0.0.0"
    port_num        = 5000
    url             = f"http://{host}:{port_num}/process"

    model_path      = "gqcnn/models/FC-GQCNN-4.0-PJ"             
    cam_intr_path   = "gqcnn/data/calib/phoxi/phoxi.intr"    
    cfg_path        = "gqcnn/cfg/examples/fc_gqcnn_pj.yaml"      
    segmask_path    = "segmask.png"        