# AI for Healthcare - Liver tumors Detection


## Description
This project aims to develop a liver cancer detection system using deep learning techniques. The system utilizes a RetinaNet model trained on a dataset of liver CT scan images to identify cancerous tumors.

- Detection of cancerous tumors in liver slices 2D images extracted from CT scan 3D images
- Data preprocessing and augmentation techniques
- Evaluation of model performance using Mean Average Precision (mAP) metrics
- Comparison of different model variations and parameter settings

## Results
The model was evaluated on three different datasets: un-normalized, augmented, and normalized. The results showed that the normalized dataset achieved the highest mAP score, indicating better performance in detecting cancerous tumors. However, the augmented dataset performed well in detecting large tumors. Further analysis is required to optimize the model's ability to detect small tumors, which are crucial for early-stage cancer diagnosis.

## Future Work
There are several areas for potential improvement and future work:
- Explore alternative backbones that trained on medical images (instead of imageNet) to enhance model performance.
- Investigate ensemble techniques by combining predictions from models trained on different image axes.
- Incorporate a two-stage detection approach to first identify the liver region then train a model on a crop image of the liver only.
- Experiment with additional models and algorithms to assess their performance in liver cancer detection.

