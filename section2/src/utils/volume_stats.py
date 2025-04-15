"""
Contains various functions for computing statistics over 3D volumes
"""
"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    Computes the Dice Similarity coefficient for two 3D volumes.
    Arguments:
        a {Numpy array} -- 3D array with first volume (prediction)
        b {Numpy array} -- 3D array with second volume (ground truth)
    Returns:
        float -- Dice coefficient
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3D inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    volume_sum = np.sum(a_bin) + np.sum(b_bin)

    if volume_sum == 0:
        return 1.0  # Perfect score if both are empty
    return 2.0 * intersection / volume_sum


def Jaccard3d(a, b):
    """
    Computes the Jaccard Similarity coefficient (Intersection over Union) for two 3D volumes.
    Arguments:
        a {Numpy array} -- 3D array with first volume (prediction)
        b {Numpy array} -- 3D array with second volume (ground truth)
    Returns:
        float -- Jaccard index
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3D inputs, got {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise Exception(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)

    intersection = np.sum(a_bin * b_bin)
    union = np.sum((a_bin + b_bin) > 0)

    if union == 0:
        return 1.0  # Perfect if both are empty
    return intersection / union
