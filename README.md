# Glaucoma classification
Glaucoma and related diagnostic features classification for [JustRAIGS grand challenge](https://justraigs.grand-challenge.org/justraigs/).

For details of the project see [ISBI 2024 paper](https://github.com/TomaszKubrak/Glaucoma_classification_JustRAIGS/blob/main/ISBI24_paper_1625_camera_ready.pdf).
DOI: 10.1109/ISBI56570.2024.10635144

The algorithm is available to use on [Grand Challenge website](https://grand-challenge.org/algorithms/justraigs_v1/).


## Acknowledgments
- The `trim_and_resize` function in `Images_preprocessing.py` and `inference.py` was adapted from [Aladdin Persson's Machine Learning Collection](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Kaggles/DiabeticRetinopathy), used under the MIT license.

- The `UNet_masks.py` file was adapted from [Paresh shahare' Optic Disc Segmentation Collection](https://github.com/Paresh-shahare/Optic-Disc-Segmentation). Information about the license are not provided by the author. 

- The `Masks_preprocessing.py`, `YoloV8_training.py` and `YoloV8_inferences.py` were adapted from [Computer vision engineer's Image Segmentation YoloV8 Collection](https://github.com/computervisioneng/image-segmentation-yolov8), used under the AGPL-3.0 license.
