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
# Function to ensure consistent sorting across operating systems

_nsre = re.compile('([0-9]+)')
def natural_sort(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

#-------------------
# Takes a filepath and subfolder, returns a list of filenames from said subfolder

def get_file_list(mainpath, subfolder):
    Img_list = []
    Main_path = mainpath
    Data_path = os.path.join(Main_path, subfolder)
    for root, dirs, files in os.walk(Data_path):
        for file in files:
            MyPath = os.path.join(root, file)
            Img_list.append(MyPath)
    Img_list.sort(key=natural_sort)        
    return Img_list

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
# Takes a list of filenames, loads them into memory with pixel values in the range [0, 1]

def load_image(data_list, img_w, img_h):
    
    image_data = np.zeros((len(data_list), img_w, img_h, 1),dtype='float16')
    
    for i in range(len(data_list)):
        img = cv2.imread(data_list[i],0)
        img = cv2.resize(img[:,:], (img_w, img_h))
        img = img.reshape(img_w, img_h)/255.
        image_data[i,:,:,0] = img
        
    return image_data

#-------------------
# Takes a list of filenames and extracts weight maps from each file

def extract_weight_maps(data_list, img_w, img_h):
    
    image_data = np.zeros((len(data_list), img_w, img_h, 1),dtype='float16')
    
    for i in range(len(data_list)):
        img = cv2.imread(data_list[i],0)
        img = cv2.resize(img[:,:], (img_w, img_h))
        img = img.reshape(img_w, img_h)/255.
        
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(img, kernel)
        eroded = cv2.erode(img, kernel)
        
        img = cv2.subtract(dilated, eroded)
        image_data[i,:,:,0] = img
        
    return image_data

#-------------------
# Loads the first image in a list of filenames and displays information for testing/debugging

def get_image_information(data_list, img_w, img_h, img_type, loaded):
    
    if loaded:
        img_uni = np.unique(data_list[0])
        print('\nUnique values of', img_type, ':\n', img_uni)
        plt.imshow(data_list[0], interpolation='nearest')
        plt.show()
        
    else:
        img = cv2.imread(data_list[0],0)
        img_uni = np.unique(img)
        print('\nUnique values of', img_type, ':\n', img_uni)
        plt.figure(figsize=(4, 4))
        plt.imshow(img)

#-------------------
# dice_coef

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

#---------------
# Weighted dice

def weighted_loss(weight_map, weight_strength=1):
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        weight_f = K.flatten(weight_map)
        weight_f = weight_f * weight_strength
        weight_f = 1 / (weight_f + 1)
        weighted_intersection = K.sum(weight_f * (y_true_f * y_pred_f))
        return -(2. * weighted_intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return weighted_dice_loss


#-------------------
# Combines multiple generators, used in generator_with_weights

def combine_generator(gen1, gen2, gen3):
    while True:
        x = gen1.next()
        y = gen2.next()
        w = gen3.next()
        yield([x, w], y)

#-------------------
# Image generator with weights

def generator_with_weights(x_train, y_train, weights_train, batch_size):
    
    data_gen_args = dict(rotation_range=10.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         horizontal_flip=True)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    weights_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow(x_train,
                                         shuffle=False,
                                         batch_size=batch_size,
                                         seed=1)
    
    mask_generator = mask_datagen.flow(y_train,
                                       shuffle=False,
                                       batch_size=batch_size,
                                       seed=1)
    
    weight_generator = weights_datagen.flow(weights_train,
                                            shuffle=False,
                                            batch_size=batch_size,
                                            seed=1)
    
    train_generator = combine_generator(image_generator, mask_generator, weight_generator)
    
    return train_generator

#-------------------
# Plot function

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