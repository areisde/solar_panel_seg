# Imports
import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
from packages.helpers import *



"""

"""
class Dataset(BaseDataset):
    """Helper class for data extraction, transformation and preprocessing.

        At the moment, we do not have one label per image (we have less labels than images, given our preprocessing methods), so we will extract image id from the annotations and will thus not be using all the images.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['solar panel']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            mask_value=1
    ):

        self.ids = [f for f in os.listdir(images_dir) if not f.startswith('.')]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.mask_value = mask_value
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mask  = cv2.imread(self.masks_fps[i], 0)
        
        # Crop
        #image = image[0:480, 0:360]
        #mask = mask[0:480, 0:360]
        #image = image[0:320, 0:320]
        #mask = mask[0:320, 0:320]
        
        # Make images squared
        image = make_it_squared(image)
        mask  = make_it_squared(mask)
        
        # Make images squared
        percentage = 60
        image = resize_img(image, percentage)
        mask  = resize_img(mask, percentage)
        
        #print(image.shape)

        
        # extract certain classes from mask (e.g. cars)
        if self.mask_value == 1:
            masks = [(mask == v) for v in self.class_values]
        else:
            masks = [(mask != v) for v in self.class_values]

        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)