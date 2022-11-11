# Final Project: RAVIR Challenge

Miranda Gisudden & Jonas Stylbäck

# Introduction

The microvasculature system plays a role in diseases such as diabetes. This system can be directly observed only in the retina, which makes this area interesting for research. The RAVIR dataset is a new dataset that can be used for segmentation of veins and arteries in the retina. In this project, a deep learning model was developed for this kind of segmentation. A U-Net model was used together with data augmentation and weight maps for improved detection of vessel boundaries.

# Methods

------OBS! Skriv mer sen om specifika parameterinställningar------

U-Net is a neural network that was developed specifically for medical image segmentation. A U-Net model for multiclass classification was developed in this project. The model used batchnormalization and dropout layers with a dropout rate of 0.2.

Data augmentation was done with Keras' ImageDataGenerator class.

Weight maps for the training and validation images were calculated from the masks. This was in order for the model to improve it's detection of vessel boundaries, which otherwise can be challenging for the model to detect since the structures are so small.


# Results

# Discussion and further improvements