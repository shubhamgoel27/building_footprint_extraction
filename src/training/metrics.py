"""
To understand mode on jaquard loss and metrics, follow the link below:
https://www.jeremyjordan.me/semantic-segmentation/

To know more on distance_transform:
1. https://github.com/gunpowder78/PReMVOS/blob/a2d8160560509cac7e6d270ad63b2e1d6a1072c6/code/refinement_net/datasets/util/DistanceTransform.py

To know more on weighted_bce_loss:
1. https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits

"""

import numpy as np
from keras import backend as K
import tensorflow as tf
from scipy.ndimage.morphology import distance_transform_edt
import config
from keras.losses import binary_crossentropy

def dice_coeff(y_true, y_pred, eps=K.epsilon()):

    if np.max(y_true) == 0.0:
        return dice_coeff(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    
    dice = (intersection + eps) / (union + eps)
  
    return dice


def dice_loss(y_true, y_pred, eps=1e-6):
    
    dloss = 1 -  dice_coeff(y_true, y_pred)
      
    return dloss


def bce_loss(y_true, y_pred):    
    return binary_crossentropy(y_true, y_pred)
    

def bce_dice_loss(y_true, y_pred):
    
    if not tf.contrib.framework.is_tensor(y_true):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        
    if not tf.contrib.framework.is_tensor(y_pred):
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)  
    
    loss = bce_loss(y_true, y_pred)*config.cross_entropy_weight + \
                            dice_loss(y_true, y_pred)*config.dice_weight
    
    return loss

    
def weight_map(masks):        
    weight_maps = tf.map_fn(lambda mask: weight_map_tf(mask), masks,dtype="float32")
    return weight_maps


def weight_map_tf(mask):
    weight_map = tf.py_func(weight_map_np, [mask], tf.float32)
    weight_map.set_shape(mask.get_shape())    
    return weight_map


def weight_map_np(mask):
    
    mask = np.squeeze(mask)
    
    if np.max(mask) == 0.0:
        weight_map = np.ones_like(mask)
        weight_map = np.expand_dims(weight_map,axis=-1)
    else:    
        distances_bg = distance_transform_edt(1-mask)
        distances_bg = np.clip(distances_bg,0,30)
        distances_bg_norm = (distances_bg - np.min(distances_bg))/(np.max(distances_bg) - np.min(distances_bg))
        inv_distances_bg = 1. - distances_bg_norm
    
        weight_map = np.ones_like(mask)
        w0 = np.sum(weight_map)
        weight_map = 5.*inv_distances_bg
        weight_map = np.clip(weight_map, 0.1,5)
        w1 = np.sum(weight_map)
        weight_map *= (w0 / w1)
        weight_map = np.expand_dims(weight_map,axis=-1)
    
    return weight_map.astype("float32")


def weighted_bce_loss(y_true, y_pred, weight):
    
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + \
                                (1. + (weight - 1.) * y_true) * \
                                (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
        
    return K.sum(loss) / K.sum(weight)


def wbce_dice_loss(y_true, y_pred):
    
    if not tf.contrib.framework.is_tensor(y_true):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        
    if not tf.contrib.framework.is_tensor(y_pred):
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)  
    
    weights = weight_map(y_true)
    
    loss = weighted_bce_loss(y_true, y_pred, weights)*config.cross_entropy_weight + \
                            dice_loss(y_true, y_pred)*config.dice_weight
    
    return loss


