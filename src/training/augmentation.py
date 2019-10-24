import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


class SegmentationAugmentation():
    def __init__(self, seed=None):
        self.random_seed = seed
        
        sometimes = lambda aug: iaa.Sometimes(0.6, aug)
        
        self.sequence = iaa.Sequential(
            [
                # Apply the following augmenters to most images.
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5), # vertically flip 50% of all images
                
                # Apply affine transformations to some of the images
                sometimes(iaa.Affine(rotate=(-10, 10), 
                                     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis 
                                     mode='symmetric', cval=(0))),
                
                iaa.SomeOf((0, 2),
                           [                               
                               iaa.Multiply((0.75, 1.25)),
                               
                               iaa.AddToHueAndSaturation((-10, 10)),  
                               
                               iaa.ContrastNormalization((0.75, 1.25)),
                               
                               # Sharpen each image, overlay the result with the original
                               # image using an alpha between 0 (no sharpening) and 1
                               # (full sharpening effect).
                               iaa.Sharpen(alpha=(0.2, 0.8), lightness=(0.75, 1.5)),    
    
                            ], random_order=True)

            ], 
            random_order=True)
    
    @staticmethod
    def np2img(np_array):
        return np.clip(np_array, 0, 255)[None,:,:,:]
    
    @staticmethod
    def np2segmap(np_array, n_classes=1):
        return ia.SegmentationMapOnImage(np_array, shape=np_array.shape, nb_classes=n_classes + 1)
    
    @staticmethod
    def segmap2np(segmap):
        return segmap.get_arr_int()[:,:,None]
    
    @staticmethod
    def img2np(img):
        return img[0]
    
    def augment_img_and_segmap(self, img, segmap):
        
        sequence = self.sequence.to_deterministic()
        
        aug_img = sequence.augment_images(img)
        aug_segmap = sequence.augment_segmentation_maps([segmap])[0]
        
        return aug_img, aug_segmap


    def run(self, images, segmaps, n_classes=1):
        
        aug_images = []
        aug_segmaps = []
        
        for i in range(len(images)):
            
            img = self.np2img(images[i])
            segmap = self.np2segmap(segmaps[i], n_classes)
            
            aug_img, aug_segmap = self.augment_img_and_segmap(img, segmap)
            
            aug_images.append(self.img2np(aug_img))
            aug_segmaps.append(self.segmap2np(aug_segmap))
            
        return aug_images, aug_segmaps


    def run_single(self, image, mask):
        aug_images, aug_masks = self.run(np.array([image]),
                                         np.array([mask]))
        return aug_images[0], aug_masks[0]