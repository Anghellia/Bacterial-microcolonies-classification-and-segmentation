import os
import cv2
import numpy as np
import pandas as pd


class ClassificationTrainDataset:
    
    def __init__(
            self,
            dataframe,
            images_dir,
            augmentation=None,
            preprocessing=None,
            reshape=False
    ):
	
        self.dataframe = dataframe
        self.images_dir = images_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.reshape = reshape

    def __getitem__(self, i):

        image_path = os.path.join(self.images_dir, self.dataframe.loc[i, 'picture'])
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        label = self.dataframe.loc[i, 'class']

        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        if self.reshape:
            return image.reshape(3, image.shape[0], image.shape[1]), label

        return image, label

    def __len__(self):
        return len(self.dataframe)



class ClassificationTestDataset:
    
    def __init__(
            self, 
            images_dir, 
            augmentation=None, 
            preprocessing=None,
            reshape=False,
    ):
	
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.reshape = reshape

    def __getitem__(self, i):
        
        image = cv2.imread(self.images_fps[i], cv2.COLOR_BGR2RGB)
        
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
            
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        if self.reshape:
            return image.reshape(3, image.shape[0], image.shape[1])

        return image
        
    def __len__(self):
        return len(self.ids)



class SegmentationTrainDataset:

    def __init__(
            self,
            images_dir,
            masks_dir,
            ids,
            augmentation=None,
            preprocessing=None
    ):

        self.ids = ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        
        image = cv2.imread(self.images_fps[i], cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = np.where(mask == np.max(mask), 1, 0).reshape((*mask.shape, 1))

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)



class SegmentationTestDataset:

    def __init__(
            self,
            images_dir,
            preprocessing=None
    ):
	
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        
        image = cv2.imread(self.images_fps[i], cv2.COLOR_BGR2RGB)

        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        return len(self.ids)
