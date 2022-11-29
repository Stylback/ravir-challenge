# Final Project: RAVIR Challenge

Miranda Gisudden & Jonas Stylbäck

## Introduction

The microvasculature system plays a role in diseases such as diabetes. This system can be directly observed only in the retina, which makes this area interesting for research. The RAVIR dataset is a new dataset that can be used for segmentation of veins and arteries in the retina. In this project, a deep learning model was developed for segmentation of retinal veins and arteries. A U-Net model was used together with data augmentation and weight maps for improved detection of vessel boundaries.

## Methods

The test set contains 19 images of retinal vessels with no corresponding masks, while the training set consist of 23 images of retinal vessels each with a corresponding mask. The training set was divided into a train and validation set with a validation ratio of 20%.

U-Net is a neural network that was developed specifically for medical image segmentation. A U-Net model for multiclass classification was developed in this project. The model used batch-normalization and dropout layers with a dropout rate of 0.3. Data augmentation was performed using Keras' `ImageDataGenerator` class. Weight maps for the training and validation images were calculated from the masks. This was in order for the model to improve it's detection of vessel boundaries, which otherwise can be challenging for the model to detect since the vessel structures are so small. Model parameters, e.g. learning rate and number of filters, were adjusted until the best looking results were achieved. Number of filters was set to `8`, optimizer to `Adam`, learning rate to `10e-4`, loss function to `weighted dice loss` with a `weight strength` of `10`, and metric to `dice coefficient`. Thereafter, the model was trained.

For prediction, the test set was used. Here, the images did not have masks, so a dummy weight map filled with 1:s was used in their place. Predictions were saved to a folder and then manually uploaded to the RAVIR Grand Challenge website, where evaluation was conducted server-side.

## Results

The model achieved poorly with a mean dice score `0.0169 ± 0.0119`, placing it at the bottom of the RAVIR Challenge leaderboards. This result was to be expected however, as will be discussed in the upcoming section.

![leaderboard](https://github.com/Stylback/ravir-challenge/blob/main/media/leaderboard.png?raw=true)

## Discussion and further improvements

The task was challening for several reasons; small dataset, small structures and high similarity between object classes were the most present. We did manage to construct a pipeline according to our outline, but we were unable to obtain satsifying results due to a flawed dice implementation.

Assuming we would have gotten our loss function to work accordingly, we believe there would still have been multiple areas of improvement which we will cover further in this section.

We did consider scraping the function and weight map altogether and go for a more traditional dice coefficient, but ultimately decided against it. There are lessons to learn from our shortcomings and we hope to share it with others trying a similiar approach.

### Custom dice coefficient

The elephant in the room and the main reason for our low performance was due to a flawed loss function implementation. Our goal was to implement a **multi-class dice coefficient** that could also handle **weight maps**, this would allows us to both better isolate veins from artieris and keep the small structures seperated from the background.

Despite not giving rise to errors, our current implementation does not allow the model to learn across epochs. Why this is we do not yet know.

![histogram](https://github.com/Stylback/ravir-challenge/blob/main/media/histogram.png?raw=true)

### Data augmentation

Due to the small datset, image augmentation is a must. We implemented a modest variation of augmentation in our image generator as we saw performance hits with greater variation. We believe that with a working loss function we would have been able to combine greater variations in augmentation with additional epochs to achieve better model performance.

### Hyperparameters

As the model would not learn, it was difficult to evaluate the impact and performance of different hyperparameter settings. We had to go on the initial model performance as displayed by the histogram, which also means that every hyperparameter was tuned to the biggest **inital** performance. With a working loss function, we would instead have tuned our hyperparameters to the largest long-term performance.

### K-fold Cross-validation

K-fold cross-validation could increase model performance. We would have not been able to do too many folds due to the limited dataset, but doing what we can to eliminate sample bias is always preferable to doing nothing.

### Dummy weight map

In order to make predictions on the test dataset using our pipeline, we still needed to provide a weight map for the loss function. Our solution was to feed a matrix filled with 1:s along with the test images, as such the weight map would have minimal effect on the prediction.

At the time of writing we noticed an unintentional interaction with regards to the dummy weight map. The weight strength variable used in training would also be present in testing, which meant all predictions would be subject to a tenfold, evenly distributed weight. This could be remedied by setting the weight strength to 1 before conducting predictions.

Ideally, remodeling the piple to separate the weight maps from the prediction altogheter would be preferred.

### Predictions

The model outputs a 3 channel deep image which should correspond to the three classes; vein, artery and background respectivly. As our predictions must to be a single channel deep only according to the challenge guidelines, we had to compress our predictions before upload. The method used is flawed, we remake each image pixel-by-pixel by taking the maximum pixel value across all channels. This is not just inefficient, it also reduces class separation in the final image as can be seen below.

| Before compression | After compression |
| --- | --- | 
| ![before compression](https://github.com/Stylback/ravir-challenge/blob/main/media/before_comp.png?raw=true) | ![after compression](https://github.com/Stylback/ravir-challenge/blob/main/media/after_comp.png?raw=true) |
