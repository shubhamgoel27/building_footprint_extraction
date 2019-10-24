import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.morphology import reconstruction as morph_reconstruction

def morphological_rescontruction(prob_mask, mask_ths, marker_ths):
    """Removes noise from a mask.

    Args:
      prob_mask: the probability mask to remove pixels having less confidence from.
      mask_ths: threshold for creating a binary mask from the probability mask.
      marker_ths: threshold for creating a binary mask to be used as a marker in morphological recontruction

    Returns:
      The mask after applying morphological reconstruction.
    """
    mask = (prob_mask > mask_ths)*1.0
    marker = (prob_mask > marker_ths)*1.0
    
    return morph_reconstruction(marker, mask)

def denoise(mask, eps=3):
    """Removes noise from a mask.

    Args:
      mask: the mask to remove noise from.
      eps: the morphological operation's kernel size for noise removal, in pixel.

    Returns:
      The mask after applying denoising.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


def grow(mask, eps=3):
    """Grows a mask to fill in small holes, e.g. to establish connectivity.

    Args:
      mask: the mask to grow.
      eps: the morphological operation's kernel size for growing, in pixel.

    Returns:
      The mask after filling in small holes.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)
