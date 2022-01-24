import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

NOME = "img"

# definir caminhos da imagem original e diretório do output
IMAGE_PATH_TRAIN = r"dir" + NOME + ".jpg"
OUTPUT_PATH_TRAIN = r"dir\DA"

############################
# carregar a imagem original e converter em array
image_Train = load_img(IMAGE_PATH_TRAIN)
image_Train = img_to_array(image_Train)

image_Train = np.expand_dims(image_Train, axis=0)

# data augmentation
imgAug = ImageDataGenerator(rotation_range=1, width_shift_range=0.10,
                            height_shift_range=0.10, zoom_range=0.05,
                            fill_mode='nearest')
imgGen = imgAug.flow(image_Train, save_to_dir=OUTPUT_PATH_TRAIN,
                     save_format='jpg', save_prefix= NOME + '_DA_')

counter = 0
for (i, newImage) in enumerate(imgGen):
    counter += 1

    if counter == 15:
        break