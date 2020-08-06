import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


main_dir = 'data/train'
mask_dir = 'data/train/mask'


def create_labels_from_json(json_dir=main_dir, num_image=261):
    labels = []

    for i in range(1, num_image + 1):
        json_file = os.path.join(json_dir, "%03d.json" % i)

        with open(json_file) as mask_info:
            data = json.load(mask_info)
            class_name = data['shapes'][0]['label']
            labels.append(class_name)

    df = pd.DataFrame(data={'picture': ["%03d.png" % x for x in range(1, num_image + 1)],
                      'class': labels})
    return df


def create_masks_from_json(json_dir=main_dir, num_image=261):
    img_size = (512, 640)
    
    for i in range(1, num_image + 1):
        json_file = os.path.join(json_dir, "%03d.json" % i)
        
        with open(json_file) as mask_info:
            data = json.load(mask_info)
            segmap = create_mask_array(data, img_size)
            
            if 'mask' not in os.listdir(main_dir):
                os.mkdir(mask_dir)
                
            save_path = os.path.join(mask_dir, "%03d.png" % i).replace('\\', '/')
            plt.imsave(save_path, segmap)


def create_mask_array(json_data, img_size):
    mask_array = np.zeros(img_size, dtype=np.int8)

    for poly in json_data['shapes']:
        poly_coords = poly['points']
        cv2.fillPoly(mask_array, np.array([poly_coords]), 1)

    return mask_array


