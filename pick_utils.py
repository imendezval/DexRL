import numpy as np
from constants import Camera, Poses


class Grasp:
    def __init__(self, grasp_array : np.ndarray):
        y, x, z, theta, q = grasp_array
        self.x     = x
        self.y     = y
        self.z     = z
        self.theta = theta
        self.q     = q

        self.grasp_array = grasp_array

        self._compute_cam_to_world()
        self._compute_angle_to_quart()
        self._transform_rot_to_global()

    def __repr__(self):
        return f"Grasp(x={self.x}, y={self.y}, z={self.z}, theta={self.theta}, q={self.q})"
    
    def _compute_cam_to_world(self):
        self.x_world = Camera.x_cam + self.x
        self.y_world = Camera.y_cam + self.y
        self.z_world = Camera.z_cam - self.z

        self.world_pos = np.array([self.x_world, self.y_world, self.z_world])
        self.setup_pos = np.array([self.x_world, self.y_world, 1.25])

    def _compute_angle_to_quart(self):
        w = np.cos(self.theta/2)
        x = 0.0
        y = 0.0
        z = np.sin(self.theta/2)

        self.rot = np.array([w, x, y, z])
    
    def _transform_rot_to_global(self):
        self.rot_global = quaternion_multiply(self.rot, Poses.basic_rot)


def offset_target_pos(target_pos_TCP, target_rot, offset_local = np.array([0, 0, 0.3135]), offset_rot = np.array([0.9238795, 0, 0, 0.3826834])):
    
    # mat = target_rot
    # target_rot = np.array([mat[1], mat[2], mat[3], mat[0]], dtype=np.float32)

    # target_orientation_mat = R.from_quat(target_rot).as_matrix()

    offset_local = offset_world_to_local(offset_local, offset_rot)

    target_rot_mat = quat_to_matrix(target_rot)
    offset_world = target_rot_mat @ offset_local 

    target_pos_wrist = target_pos_TCP - offset_world

    return target_pos_wrist


def offset_world_to_local(offset_world_meas, q_meas):
    """
    Turn an offset you recorded in the *world* frame into the
    constant local offset used by the IK helper.

    offset_world_meas : (3,)  vector in world coords
    q_meas            : (4,)  flange quaternion [w, x, y, z] at measurement time
    """
    R_meas = quat_to_matrix(q_meas)
    return R_meas.T @ offset_world_meas


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions in [w, x, y, z] format.
    Returns the resulting quaternion in the same format.
    
    This performs q_result = q2 * q1,
    meaning q1 is applied first, then q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1

    return np.array([w, x, y, z], dtype=np.float32)


def quat_to_matrix(q):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.
    Works with numpy arrays or python lists.
    """
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    return np.array([
        [ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz],
    ])


def mat_to_quat(R, *, tol=1e-8):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].

    Parameters
    ----------
    R   : array-like shape (3, 3)
          Proper orthonormal rotation matrix.
    tol : float, optional
          Numerical tolerance for the trace-positive branch.

    Returns
    -------
    q   : ndarray shape (4,)
          Unit quaternion [w, x, y, z].
    """
    R = np.asarray(R, dtype=float)
    t = R.trace()

    if t > tol:                                        # standard “trace” branch
        s  = 0.5 / np.sqrt(t + 1.0)
        w  = 0.25 / s
        x  = (R[2, 1] - R[1, 2]) * s
        y  = (R[0, 2] - R[2, 0]) * s
        z  = (R[1, 0] - R[0, 1]) * s
    else:                                              # diagonal-dominant branches
        i = np.argmax(np.diag(R))
        if i == 0:            # R[0,0] is largest
            s  = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w  = (R[2, 1] - R[1, 2]) / s
            x  = 0.25 * s
            y  = (R[0, 1] + R[1, 0]) / s
            z  = (R[0, 2] + R[2, 0]) / s
        elif i == 1:          # R[1,1] is largest
            s  = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w  = (R[0, 2] - R[2, 0]) / s
            x  = (R[0, 1] + R[1, 0]) / s
            y  = 0.25 * s
            z  = (R[1, 2] + R[2, 1]) / s
        else:                 # R[2,2] is largest
            s  = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w  = (R[1, 0] - R[0, 1]) / s
            x  = (R[0, 2] + R[2, 0]) / s
            y  = (R[1, 2] + R[2, 1]) / s
            z  = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def wxyz_to_xyzw(q):
    """[w x y z]  ->  [x y z w]"""
    w, x, y, z = q
    return np.array([x, y, z, w], dtype=q.dtype)

def xyzw_to_wxyz(q):
    """[x y z w]  ->  [w x y z]"""
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=q.dtype)
