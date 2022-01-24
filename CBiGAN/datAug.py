# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 14:05:38 2021

@author: JoceleideMumbelli
"""

import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import os

img_names = glob(os.path.join(os.getcwd(),'*.jpg'))

for fn in img_names:
    print(fn)


    IMAGE_PATH_TRAIN = fn
    OUTPUT_PATH_TRAIN = r"C:\Users\pesqu\Desktop\Reais"
    
    print(IMAGE_PATH_TRAIN)
    ############################
    # carregar a imagem original e converter em array
    image_Train = load_img(IMAGE_PATH_TRAIN)
    image_Train = img_to_array(image_Train)
    
    image_Train = np.expand_dims(image_Train, axis=0)
    
    # data augmentation
    imgAug = ImageDataGenerator(rotation_range=1, width_shift_range=0.08,
                                height_shift_range=0.03, zoom_range=0.04,
                                fill_mode='nearest')
    imgGen = imgAug.flow(image_Train, save_to_dir=OUTPUT_PATH_TRAIN,
                        save_format='jpg', save_prefix= '_DA_')
    
    counter = 0
    for (i, newImage) in enumerate(imgGen):
        counter += 1
    
        if counter == 10:
            break