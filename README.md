# Evaluating the influence of the face as a feature on multiple saliency models

Testing multiple architectures and features and how influence they are by face presence in human saliency/attention model. the repository also includes Python implementation of Itti&Koch model and metrics for saliency based on original matlab code.

#### -- Project Status: [Active]

## Project Intro/Objective
Verification of influence of face presence in attention modeling, including multiple approaches (architectures and features included) to saliency modeling.

### Methods Used:
* Saliency Maps
* Face Detection
* Data Visualization

### Technologies
* Python
* Pandas, jupyter
* tensorflow
* [pysaliency]https://github.com/matthias-k/pysaliency

## Project Description
We proposed three hypotheses: That the addition of faces has a positive influence on the performance of traditional models, that this influence is dependent on the context of the picture and that adding faces can make them compete with the recently proposed models based on neural networks. To answer these questions, we evaluated the performance of nine traditional models with and without faces and one model based on a deep neural network for different metrics. The results show a clear benefit for the models when faces are added, although the level of importance we should give to the faces is judged differently across metrics. Moreover, the best performing traditional model complemented with faces can approach the performance of a model based on a deep neural network. From this we conclude, that adding weighted object recognition onto saliency model with hand-crafted features can yield useful insights and should be done also for other salient objects and image regions.

We have implemented a pipeline that allows creation of mixture of features, metrics and faces influence ratio and its evaluation.

### Models Used:
* Itti&Koch architecture
* AIM: An information theoretic approach
* SUN: A Bayesian framework for Saliency Using Natural Statistics
* CAS: Contrast-Aware Saliency
* CovSal: Non-linearly integrating features using region covariances
* GBVS: Graph-Based Visual Saliency
* DeepGaze II: Reading fixations from deep features trained on object recognition
* ICF: Intensity Contrast Features
* Face

### Metrics Used:
* AUC: Area under the ROC curve
* NSS: Normalized Scanpath Saliency
* SIM: Similarity
* Information Gain (IG)

### Dataset:
* [COCO](http://cocodataset.org/#download)
* [Salicon](http://salicon.net/challenge-2017/)

Datasets were merged to extract different context for analysis of the data.

## Needs of this project
* scaling the approach to full COCO Dataset
* normalization of the pipeline

## Team:
* [Annika Thierfelder](https://github.com/athierfelder)
* [Aleksandra Teska](https://github.com/alTeska)

## Dependancies:
* [pysaliency](https://github.com/matthias-k/pysaliency)
* [face_recognition](https://github.com/ageitgey/face_recognition)

## Acknowledgments:
* [Institute for Cognitive Systems, Technical University Munich](http://www.ics.ei.tum.de/en/home/) - for supervision
