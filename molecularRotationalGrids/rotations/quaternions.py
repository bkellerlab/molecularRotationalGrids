import numpy as np

np.random.seed(1)


def inverse_last_coordinate(quat: np.ndarray) -> np.ndarray:
    """
    Get quaternion (q_0, q_1, q_2, q_3), return quaternion (q_0, q_1, q_2, -q_3)

    Args:
        quat: quaternion as numpy array with 4 components

    Returns:
        a quaternion
    """
    assert len(quat) == 4 or quat.shape[1] == 4, "Quaternion must have 4 components!"
    return np.array([quat[0], quat[1], quat[2], -quat[3]])


def inverse_quaternion(quat: np.ndarray) -> np.ndarray:
    """
    Get quaternion (q_0, q_1, q_2, q_3), return quaternion (q_0, -q_1, -q_2, -q_3)

    Args:
        quat: quaternion as numpy array with 4 components

    Returns:
        a quaternion
    """
    assert len(quat) == 4 or quat.shape[1] == 4, "Quaternion must have 4 components!"
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])
