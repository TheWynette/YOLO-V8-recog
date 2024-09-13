
numnumnumnumnum - v1 2024-08-07 1:01pm
==============================

This dataset was exported via roboflow.com on August 7, 2024 at 1:02 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 96 images.
Number are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Randomly crop between 0 and 25 percent of the image
* Random rotation of between -15 and +15 degrees
* Random shear of between -14° to +14° horizontally and -13° to +13° vertically
* Random brigthness adjustment of between -23 and +23 percent
* Random exposure adjustment of between -15 and +15 percent
* Random Gaussian blur of between 0 and 3.2 pixels
* Salt and pepper noise was applied to 1.8 percent of pixels


