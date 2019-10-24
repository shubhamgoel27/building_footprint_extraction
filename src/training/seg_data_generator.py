"""
This is a utility class for Data Generation in Keras for the specific problem of semantic segmentaion. 
The api is heavily inspired from the following links:
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    https://github.com/BaptisteLevasseur/Semantic-segmentation-on-buildings/blob/master/generator.py

"""

import numpy as np
import keras
import buzzard as buzz

from src.training.augmentation import SegmentationAugmentation
         
#TODO: Steps per epoch to discuss. It is getting defined at 2 places. 
class SegDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_path, img_source, mask_source, batch_size=8, no_of_samples = 500, tile_size=128, 
                 downsampling_factor = 1.0, aug=True):
        'Initialization'
        self.dataset_path = dataset_path
        self.ts = tile_size
        self.ds = downsampling_factor
        self.batch_size = batch_size
        self.no_of_samples = no_of_samples
        self.img_source = img_source
        self.mask_source = mask_source
        self.to_augment = aug
        self.augmentation = SegmentationAugmentation()
    
    def pre_process(self, image_list, isbinary=False):
        
        new_list = []
        
        for i in range(self.batch_size):
            
            #Step1: normalise  
            image = image_list[i]
            
            if isbinary:
                image = (image > 0)*abs(image) #binary mask with value 0 or 1
            else:
#                image = image/127.5-1.0 #normalising the image for values between -1 to +1
                image = (image - np.min(image) + 1.0) / (np.max(image) - np.min(image) + 1.0) #normalising the image for values between 0 to +1
            new_list.append(image)
        
        return new_list

    def random_crop(self,ds, tile_size=128, factor=1):

        # Cropping parameters
        crop_factor = ds.rgb.fp.rsize/tile_size
        crop_size = ds.rgb.fp.size/crop_factor

        # Original footprint
        fp = buzz.Footprint(
            tl=ds.rgb.fp.tl,
            size=ds.rgb.fp.size,
            rsize=ds.rgb.fp.rsize ,
        )

        min = np.random.randint(fp.tl[0],fp.tl[0] + fp.size[0] - factor*crop_size[0]) #x
        max = np.random.randint(fp.tl[1] - fp.size[1] + factor*crop_size[1], fp.tl[1]) #y

#        print(min > fp.tlx, max < fp.tly)
#        print("fp.tlx", fp.tlx, "fp.tly", fp.tly, "min , max", min, max)
        # New random footprint
        tl = np.array([min, max])

        fp = buzz.Footprint(
            tl=tl,
            size=crop_size*factor,
            rsize=[tile_size, tile_size],
        )

        return fp

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        steps_per_epoch = int(np.floor((len(self.img_source)*self.no_of_samples) / self.batch_size))
        return steps_per_epoch
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        # Generate indexes of the batch
        index = np.random.randint(len(self.img_source))

        # Generate data
        X, y = self.__data_generation(index)
                
        return X, y

    def __data_generation(self, index):
        'Generates data containing batch_size samples' 
        
#        print(index)
        ds_rgb = self.img_source[index]
        ds_binary = self.mask_source[index] 
        
        # Initialization
        X = []
        y = []          
        
        if self.to_augment:
            batch_size = self.batch_size//2
        else:
            batch_size = self.batch_size
            
        for i in range(batch_size):
                        
            while True:
                while True:
                    try:
                        fp = self.random_crop(ds_rgb, tile_size=self.ts, factor=self.ds)
                        rgb= ds_rgb.rgb.get_data(band=(1, 2, 3), fp=fp).astype('uint8')
                        
                        binary= ds_binary.rgb.get_data(band=(1), fp=fp).astype('uint8')
                        binary = binary.reshape((self.ts, self.ts,1))
                        binary = binary // 255 #converting numpy into binary with 0 or 1
                                                
                        break
                    
                    except:
                        continue
                if np.sum(rgb == 0) < self.ts*self.ts*0.7*3: #if the zero region is less than 70% of total image
                    break
                    
            X.append(rgb)
            y.append(binary)
        
        if self.to_augment:
            aug_images, aug_masks = self.augmentation.run(X, y)        
            X.extend(aug_images)
            y.extend(aug_masks)

        X = self.pre_process(X) 
        
#        print(len(X), len(y))
              
        return np.array(X), np.array(y)        