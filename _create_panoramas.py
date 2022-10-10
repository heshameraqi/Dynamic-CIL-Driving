import os
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
from panorama import Panaroma
import imutils
import multiprocessing
import pickle

# ------------------------------------------------------------------------------
# Core Function
# ------------------------------------------------------------------------------
def core_function(img):
    if not os.path.isfile(img.replace('CentralRGB_', 'PanoramaRGB_')):  # if not created before already
        img_c = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        img_r = cv2.cvtColor(cv2.imread(img.replace('CentralRGB_', 'RightRGB_')), cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(cv2.imread(img.replace('CentralRGB_', 'LeftRGB_')), cv2.COLOR_BGR2RGB)
        
        # Generate Panorama Image
        panaroma = Panaroma()
        img_panorama = panaroma.image_stitch([img_c, img_r])
        
        # Flip to make transformed image (right) is left camera
        img_panorama = cv2.flip(img_panorama, -1)
        img_l = cv2.flip(img_l, -1)
        
        img_panorama = panaroma.image_stitch([img_panorama, img_l])
        
        # Flip back
        img_panorama = cv2.flip(img_panorama, -1)
        
        cv2.imwrite(img.replace('CentralRGB_', 'PanoramaRGB_'), img_panorama[:,:,[2,1,0]])
        
        '''fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        img_l = cv2.flip(img_l, -1)
        ax[0, 0].set_title('Left camera')
        ax[0, 1].set_title('Centre camera')
        ax[0, 2].set_title('Right camera')
        ax[1, 1].set_title('Panorama view')
        ax[0, 0].imshow(img_l)
        ax[0, 0].set_aspect('equal')
        ax[0, 1].imshow(img_c)
        ax[0, 1].set_aspect('equal')
        ax[0, 2].imshow(img_r)
        ax[0, 2].set_aspect('equal')
        ax[1, 1].imshow(img_panorama)
        ax[1, 1].set_aspect('equal')
        plt.show()'''
    
# ------------------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------------------
stitcher = cv2.createStitcher(True)

with open('auc2dataset_metadata_train.pickle', 'rb') as f:
    cam1_image_files_train, cam2_image_files_train, cam3_image_files_train, \
    pgm_image_files_train, high_lvl_cmds_train, steers_train, throttles_train, \
    brakes_train, speeds_train = pickle.load(f)
with open('auc2dataset_metadata_val.pickle', 'rb') as f:
    cam1_image_files_val, cam2_image_files_val, cam3_image_files_val, \
    pgm_image_files_val, high_lvl_cmds_val, steers_val, throttles_val, \
    brakes_val, speeds_val = pickle.load(f)
    
# Serial
'''for img in tqdm(cam1_image_files_train):
    core_function(img)
for img in tqdm(cam1_image_files_val):
    core_function(img)'''

# Parallel
p = multiprocessing.Pool(10)
p.map(core_function, cam1_image_files_train)
p.close()
p.join()

p = multiprocessing.Pool(10)
p.map(core_function, cam1_image_files_val)
p.close()
p.join()
            
