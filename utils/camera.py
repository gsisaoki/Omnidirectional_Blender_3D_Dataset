import numpy as np
import math

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion (qw, qx, qy, qz) into a 3x3 rotation matrix.
    """
    # Normalize the quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    # Compute rotation matrix
    r11 = 1 - 2 * (qy**2 + qz**2)
    r12 = 2 * (qx*qy - qz*qw)
    r13 = 2 * (qx*qz + qy*qw)
    r21 = 2 * (qx*qy + qz*qw)
    r22 = 1 - 2 * (qx**2 + qz**2)
    r23 = 2 * (qy*qz - qx*qw)
    r31 = 2 * (qx*qz - qy*qw)
    r32 = 2 * (qy*qz + qx*qw)
    r33 = 1 - 2 * (qx**2 + qy**2)

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

def angle_axis_to_quaternion(angle_axis: np.ndarray):
    angle = np.linalg.norm(angle_axis)
    x = angle_axis[0] / angle
    y = angle_axis[1] / angle
    z = angle_axis[2] / angle

    qw = math.cos(angle / 2.0)
    qx = x * math.sqrt(1 - qw * qw)
    qy = y * math.sqrt(1 - qw * qw)
    qz = z * math.sqrt(1 - qw * qw)

    return np.array([qw, qx, qy, qz])

def angle_axis_to_rotation_matrix(angle_axis: np.ndarray):
    quaternion = angle_axis_to_quaternion(angle_axis)
    rotation_matrix = quaternion_to_rotation_matrix(*quaternion)
    return rotation_matrix