import numpy as np

class Settings:
    vis_DexNet      = False
    
    num_envs        = 4
    env_spacing     = 3.0


class Camera:
    height          = 386
    width           = 516

    f_x             = 552.5
    f_y             = 552.5
    c_x             = 255.5
    c_y             = 191.75
    intr_matrix     = [f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1]

    x_cam           = 0.5
    y_cam           = 0.3
    z_cam           = 1.55 # + 0.442
    pos             = np.array([x_cam, y_cam, z_cam])
    rot             = np.array([0, 0.7071, 0.7071, 0])


class Prims:
    class Table:
        path        = "assets/table_instanceable.usd"
        pos         = (0.55, 0.0, 1.04)
        rot         = (0.7071, 0.0, 0.0, 0.7071)
        scale       = (1.2, 1.0, 1.0)

    class KLT_Bin:
        path        = "assets/small_KLT.usd"
        pos_pick    = (0.5,  0.3, 1.075)  #+-0.135, +-0.195
        pos_place   = (0.5, -0.3, 1.075)
        scale       = (1.5, 1.5, 0.5)

    class ObjPool:
        path        = "assets/meshes/USD/kit"
        n_obj_pool  = 129
        n_objs_ep   = 12

        pos_x       = 0.4
        pos_y       = 0.15
        x_bins      = 3
        y_bins      = 5
        spacing     = 0.06
        z_min       = 1.15
        z_max       = 1.35

        mu_static   = 1.20
        mu_dynamic  = 0.90
        sigma       = 0.10

        mu_rest     = 0.05
        sigma_rest  = 0.01

        mu_mass     = 0.175
        sigma_mass  = 0.10


class RobotArm:
    class FrankaArm:
        path        = "assets/franka/franka_no_gripper.usd"
        pos         = (0.0, 0.0, 1.05)
    
    class Gripper:
        path        = "assets/franka/yumi_gripper_mimic_mu.usd"
        offset_pos  = np.array([0.0, 0.0, 0.1068])


class Poses:
    # Transformation from URDF default to base orientation (Gripper: Facing up, 45° rot -> Facing down, no rotation)
    base_rot       = np.array([0, 0.3826834, 0.9238795, 0])     

    # Measured offset from wrist of Franka to desired TCP of Yumi Gripper
    offset          = np.array([0.0, 0.0, 0.02])
    # Rotation of Franka at time of measurement (45° gripper rot)
    offset_rot      = np.array([0.9238795, 0, 0, 0.3826834])

    setup_TCP       = np.array([0.5, 0.3, 1.25], dtype=np.float32)
    lift_TCP        = np.array([0.5, 0.3, 1.25], dtype=np.float32)
    inter_TCP       = np.array([0.4, 0.0, 1.30], dtype=np.float32)
    drop_TCP        = np.array([0.5, -0.3, 1.2], dtype=np.float32)


class DexNet:
    host            = "0.0.0.0"
    port_num        = 5000
    url             = f"http://{host}:{port_num}/process"

    model_path      = "gqcnn/models/FC-GQCNN-4.0-PJ"             
    cam_intr_path   = "gqcnn/data/calib/phoxi/phoxi.intr"    
    cfg_path        = "gqcnn/cfg/examples/fc_gqcnn_pj.yaml"      
    segmask_path    = "segmask.png"