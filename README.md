# AI-For-MedicalScience
Automated DR detection system which will be provided as a service to the doctors to use it for the betterment of humanity.

## What was the purpose for selecting this project?
### Abstract
**Diabetic retinopathy** is a leading problem throughout the world and many people are losing their
vision because of this disease. The disease can get severe if it is not treated properly at its early
stages. The damage in the retinal blood vessel eventually blocks the light that passes through the
optical nerves which makes the patient with Diabetic Retinopathy blind. Therefore, in our
research we wanted to find out a way to overcome this problem and thus using the help of
**Convolutional Neural Network** (ConvNet), we wereable to detect multiple stages of severity for
Diabetic Retinopathy. There are other processes present to detect Diabetic Retinopathy and one
such process is manual screening, but this requires a skilled ophthalmologist and takes up a huge
amount of time. Thus our automatic diabetic retinopathy detection technique can be used to
**replace** such **manual processes** and theophthalmologist can spend more time taking proper care
of the patient or at least decrease the severity of this disease.

### Condition Of Diabetic Retinopathy In India

Currently, In India diabetes is a disease that affects over 65 million persons in India. 

Diabetes-related eye disease, of which retinopathy is the most important, affects nearly one out of every ten persons with diabetes, according to point prevalence estimates. Many few of them are aware of that if they have diabetes for over several years they may come across the diabetic complication.

To spread awareness among people major hospitals in India organizes the free eye checkup camps in villages where people can get their eye checkup for free. 

Those retinal images of people were collected and sent to an expert Ophthalmologist. After that, the Ophthalmologist examines those images and the summons those patients who were likely to suffer from Diabetic Retinopathy. 

This summoned patient than were informed that they are likely to suffer from Diabetic Retinopathy and should consult the expert Ophthalmologist for a proper checkup.

This whole process takes almost half a month or more and to shorten this gap we had come up with the idea which almost cut down these process into one or two days, which help the Ophthalmologist to focus more on the treatment and avoid the hectic work of identifying which patient has Diabetic Retinopathy and which doesn't.

## What was our approach?
### Why we have selected ConvNet to solve this problem? / Objective
Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world. The condition is estimated to affect over 93 million people.

The need for a comprehensive and automated method of diabetic retinopathy screening has long been recognized, and previous efforts have made good progress using image classification, pattern recognition, and machine learning. With photos of eyes as input, the goal of this project is to create a new model, ideally resulting in realistic clinical potential.

The motivations for this project are twofold:

1. Image classification has been a personal interest for years, in addition to classification on a large scale data set.

2. Time is lost between patients getting their eyes scanned (shown below), having their images analyzed by doctors, and scheduling a follow-up appointment. By processing images in real-time, EyeNet would allow people to seek & schedule treatment the same day

      ![DR Manual Screening](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/dr_scan.gif)

### From where we get dataset to train our model?
The data originates from a [2015 Kaggle competition](https://www.kaggle.com/c/diabetic-retinopathy-detection). However, is an atypical Kaggle dataset. In most Kaggle competitions, the data has already been cleaned, giving the data scientist very little to preprocess. With this dataset, this isn't the case.

All images are taken of different people, using different cameras, and of different sizes. Pertaining to the preprocessing section, this data is extremely noisy, and requires multiple preprocessing steps to get all images to a useable format for training a model.

The training data is comprised of 35,126 images, which are augmented during preprocessing.

### Exploratory Data Analysis
The very first item analyzed was the training labels. While there are five categories to predict against, the plot below shows the severe class imbalance in the original dataset.

![DR_vs_Frequency_table](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/DR_vs_Frequency_tableau.png)

**Confusion matrix** of **original** train **CSV**.
![trainLabels_confusion_matrix](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/trainLabels_confusion_matrix.png)

Of the original training data, 25,810 images are classified as not having retinopathy, while 9,316 are classified as having retinopathy.

Due to the class imbalance, steps taken during preprocessing in order to rectify the imbalance, and when training the model.

Furthermore, the variance between images of the eyes is extremely high. The first two rows of images show class 0 (no retinopathy); the second two rows show class 4 (proliferative retinopathy).

1. class 0 (no retinopathy)
![No_DR_white_border_1](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/No_DR_white_border_1.png)
![No_DR_white_border_2](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/No_DR_white_border_2.png)
                
2. class 4 (proliferative retinopathy)
![Proliferative_DR_white_border_1](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/Proliferative_DR_white_border_1.png)
![Proliferative_DR_white_border_2](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/Proliferative_DR_white_border_2.png)

### Different types of data preprocessing and data augmentation techniques we use to deal with major class imbalance
The preprocessing pipeline is the following:
1. [Gregwchase](https://github.com/gregwchase/dsi-capstone) approach
    - [x] **[Crop](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/1_crop_and_resize.py)** images into 1800x1800 resolution
    - [x] **[Resize](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/1_crop_and_resize.py)** images to 512x512/256x256 resolution
    - [x] **[Remove](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/2_find_black_images.py)** totally **black images** form dataset
    - [x] **[Rotate](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/3_rotate_images.py)** and **mirror**(Rotate DR images to 90°,120°,180°,270° + mirror, and only mirror non-DR images)
    - [x] **[Update](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/4_reconcile_label.py)** **CSV** so it should contain all the augmented images and there respective labels
    - [ ] **[Convert](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/5_image_to_array.py)** images to numpy array
    
2. [Ms.Sheetal Maruti Chougule/Prof.A.L.Renke](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/documents/research%20paper/New_Preprocessing_approach_for_Images_in-Diabetic_Retinopathy_Screening%20.pdf) approach
    - [x] Image **[Denoising](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/6_Denoise_and_CLAHE.py)**
    - [x] **[CLAHE](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/6_Denoise_and_CLAHE.py)** (Contrast Limited Adaptive Histogram Equalization)
    
3. [Ben Graham](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/documents/research%20paper/competitionreport.pdf) approach(Only Works in python2.7)
    - [x] **[Rescale](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/Ben%20Graham/1_remove_boundary_effects.py)** the images to have the same radius (300 pixels or 500 pixels)
    - [x] Subtracted the local average color; the **[local average gets mapped to 50% gray](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/Ben%20Graham/1_remove_boundary_effects.py)**
    - [x] Clipped the images to 90% size to **[remove the boundary effects](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/src/Preprocessing%20Scripts/Train/Ben%20Graham/1_remove_boundary_effects.py)**
    
#### 1. Gregwchase approach
##### Crop images into 1800x1800 resolution
In total, the original dataset totals 35 gigabytes. All images were croped down to 1800 by 1800.

##### Resize images to 512x512/256x256 resolution
All images were scaled down to 512 by 512 and 256 by 256. Despite taking longer to train, the detail present in photos of this size is much greater then at 128 by 128.

##### Remove totally black images form dataset
Additionally, 403 images were dropped from the training set. Scikit-Image raised multiple warnings during resizing, due to these images having no color space. Because of this, any images that were completely black were removed from the training data.

##### Rotate and mirror (Rotate DR images to 90°,120°,180°,270° + mirror, and only mirror non-DR images)
All images were rotated and mirrored.Images without retinopathy were mirrored; images that had retinopathy were mirrored, and rotated 90, 120, 180, and 270 degrees.

The first images show two pairs of eyes, along with the black borders. Notice in the cropping and rotations how the majority of noise is removed.

![sample_images_unscaled](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/sample_images_unscaled.jpg)

![17_left_horizontal_white](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/17_left_horizontal_white.jpg)

After rotations and mirroring, the class imbalance is rectified, with a few thousand more images having retinopathy. In total, there are 106,386 images being processed by the neural network.

![DR_vs_frequency_balanced](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/DR_vs_frequency_balanced.png)

**Confusion matrix** of **new CSV** after image augmentation.
![trainlabel_master_v2_confusion_matrix](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/trainlabel_master_v2_confusion_matrix.png)
    
### Our neural network architecture

## How we make it accessible to doctors?
In our research, to tackle the aforementioned challenges, we built a predictive model for Computer-Aided Diagnosis (CAD), leveraging eye fundus images that are widely used in present-day hospitals, given that these images can be acquired at a relatively low cost.
Additionally, based on our CAD model, we developed a novel tool for diabetic retinopathy diagnosis that takes the form of a prototype web application. The main contribution of this research stems from the novelty of our predictive model and its integration into a prototype web application.

### How the prediction pipline works? 
First start the flask app 
```
python app.py
```
1. Take the retinal image of person one per each eye
2. Upload the image to website
![upload_image_1](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/upload_image_1.jpg)
![upload_image_1_2](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/upload_image_1_2.jpg)

3. We have created a REST API which takes two images as input and return JSON response
4. The response from API is displayed into bar graph
![upload_image_2](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/upload_image_2.jpg)

5. You can also generate PDF which contain images you upload and their predictions for doctors can refer it for later use
![PDF_Generated_1](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/PDF_Generated_1.jpg)
![PDF_Generated_1_2](https://github.com/Tirth27/AI-For-MedicalScience/blob/master/images/readme/PDF_Generated_1_2.jpg)

## Credits
This project cannot be completed without you guys [@github/PatrioticParth](https://github.com/PatrioticParth) and [@github/hv245](https://github.com/hv245). Thanks for your support :) 

## References

1. [Denoise](https://docs.opencv.org/3.3.0/d5/d69/tutorial_py_non_local_means.html) and [CLAHE](https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)
2. Ben Graham [1](https://github.com/btgraham/SparseConvNet) [2](https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801)
3. [Gregwchase](https://github.com/gregwchase/dsi-capstone)
