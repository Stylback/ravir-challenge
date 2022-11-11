#-------------------
# Prerequisite modules

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import re
import random

#-------------------
# Ensures consistent sorting across operating systems, used in get_file_list

def natural_sort(s):
    sort = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(sort, s)]

#-------------------
# Takes a filepath and subfolder, returns a sorted list of filenames from said subfolder

def get_file_list(main_path, subfolder):
    file_list = []
    data_path = os.path.join(main_path, subfolder)
    
    for root, dirs, files in os.walk(data_path):
        for file in files:
            path = os.path.join(root, file)
            file_list.append(path)
    
    file_list.sort(key = natural_sort)
    return file_list

#-------------------
# Takes two lists of filenames, returns them split in training and validation subsets

def get_train_val_lists(image_list, mask_list, val_ratio):
    
    zipped = list(zip(image_list, mask_list))
    random.shuffle(zipped)
    image_list[:], mask_list[:] = zip(*zipped)
    val_split = int(val_ratio * len(image_list))
    
    train_image_list = image_list[val_split:]
    train_mask_list = mask_list[val_split:]
    val_image_list = mask_list[:val_split]
    val_mask_list = image_list[:val_split]

    return train_image_list, train_mask_list, val_image_list, val_mask_list

#-------------------
# Takes a list of filenames and loads the files into memory
# Images have their pixel values set in the range [0,1] while masks have theirs set according to relevant label

def load_image(data_list, img_w, img_h, img_c, img_type):
    
    if img_type == 'image':
        image_data = np.zeros((len(data_list), img_w, img_h, img_c),dtype='float32')
        for i in range(len(data_list)):
            img = cv2.imread(data_list[i], 0)
            img = cv2.resize(img[:,:], (img_w, img_h))
            img = img.reshape(img_w, img_h)/255.
            image_data[i,:,:,0] = img
    
    else:
        image_data = np.zeros((len(data_list), img_w, img_h, img_c),dtype='float32')
        for i in range(len(data_list)):
            img = cv2.imread(data_list[i], 0)
            img = cv2.resize(img[:,:], (img_w, img_h))
            img[img == 0] = 0
            img[img == 128] = 1
            img[img == 255] = 2
            image_data[i,:,:,0] = img
            
    return image_data

#-------------------
# Takes a list of filenames and extracts weight maps from each before loading them into memory

def extract_weight_maps(data_list, img_w, img_h):
    
    image_data = np.zeros((len(data_list), img_w, img_h, 1), dtype='float32')
    kernel = np.ones((3, 3), np.uint8)
    
    for i in range(len(data_list)):
        img = cv2.imread(data_list[i], 0)
        img = cv2.resize(img[:,:], (img_w, img_h))
        img = img.reshape(img_w, img_h)/255.

        dilated = cv2.dilate(img, kernel)
        eroded = cv2.erode(img, kernel)
        
        img = cv2.subtract(dilated, eroded)    
        image_data[i,:,:,0] = img
        
    return image_data

#-------------------
# Takes the first file in a list of filenames and displays image information for testing/debugging purposes

def get_image_information(data_list, img_w, img_h, img_type, loaded):
    
    if loaded:
        img_shape = np.shape(data_list[0])
        img_uni = np.unique(data_list[0])
        print('\nShape and Unique values of', img_type, ':\n', img_shape, '\n', img_uni)
        plt.imshow(data_list[0])
        plt.show()
        
    else:
        img = cv2.imread(data_list[0],0)
        img_shape = np.shape(img)
        img_uni = np.unique(img)
        print('\nShape and Unique values of', img_type, ':\n', img_shape, '\n', img_uni)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)

#-------------------
# Multiclass Sørensen-Dice coefficient

def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    denominator = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return (2 * intersection) / (denominator + K.epsilon())

#---------------
# Multiclass Sørensen-Dice-loss with weight maps

def weighted_loss(weight_map, weight_strength=1):
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=3)[...,1:])
        y_pred_f = K.flatten(y_pred[...,1:])
        
        weight_f = K.flatten(K.one_hot(K.cast(weight_map, 'int32'), num_classes=3)[...,1:])
        weight_f = weight_f * weight_strength
        weight_f = 1 / (weight_f + 1)
        
        intersect = K.sum(weight_f * (y_true_f * y_pred_f), axis=-1, keepdims = False)
        denom = K.sum(weight_f * (y_true_f + y_pred_f), axis=-1, keepdims = False)
        return K.mean(((2. * intersect + K.epsilon()) / (denom + K.epsilon())))
    return weighted_dice_loss

#-------------------
# Combines multiple generators, used in generator_with_weights

def combine_generator(image_generator, mask_generator, weight_generator):
    while True:
        x = image_generator.next()
        y = mask_generator.next()
        w = weight_generator.next()
        yield([x, w], y)

#-------------------
# Image generator with weights

def generator(x_train, y_train, w_train, batch_size):
    
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow(x_train,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         seed=1)
    
    mask_generator = image_datagen.flow(y_train,
                                       shuffle=False,
                                       batch_size=batch_size,
                                       seed=1)
    
    weight_generator = image_datagen.flow(w_train,
                                            shuffle=False,
                                            batch_size=batch_size,
                                            seed=1)
    
    train_generator = combine_generator(image_generator, mask_generator, weight_generator)
    
    return train_generator

#-------------------
# Plots model history for performance evaluation

def plot_model_history(size_x, size_y, title, x_label, y_label, legend, print_keys, model_hist):
    
    keys = list(model_hist.history.keys())
    if print_keys:
        print("The keys are: ", keys)

    plt.figure(figsize=(size_x, size_y))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    for key in keys:
        plt.plot(model_hist.history[key], label=key)
    
    if legend:
        plt.legend();
    
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()
    
#--------------------
# Performs dimension reduction on predictions before saving them in a directory

def save_predictions(predictions):
    filenames = sorted(os.listdir('/tf/ravir-challenge/dataset/test'))
    path = '/tf/ravir-challenge/predictions'
    os.chdir(path)
    index = 0

    for image in predictions:
        image_data = np.zeros((768, 768), dtype='float32')
        for i in range(len(image[:])):
            for j in range(len(image[i,:])):
                pixel_value = np.max(image[i,j,:])
                pixel_value = pixel_value * 255
                image_data[i][j] = pixel_value
        
        cv2.imwrite(filenames[index], image_data)
        index = index+1
        print(index, ' out of ', len(predictions), ' converted')
