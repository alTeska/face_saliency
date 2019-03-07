# Evaluating the influence of the face as a feature on multiple saliency models

Testing multiple architectures and features and how influence they are by face presence in human saliency/attention model. the repository also includes Python implementation of Itti&Koch model and metrics for saliency based on original matlab code.

#### -- Project Status: [Active]

## Project Intro/Objective
Verification of influence of face presence in attention modeling, including multiple approaches (architectures and features included) to saliency modeling.

### Methods Used:
* Saliency Maps
* [Face Detection](https://github.com/ageitgey/face_recognition)
* Data Visualization

### Technologies
* Python
* Pandas, jupyter
* tensorflow

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

### Metrics Used:
* AUC: Area under the ROC curve
* NSS: Normalized Scanpath Saliency
* SIM: Similarity
* Information Gain (IG)

### Dataset:
* [COCO](http://cocodataset.org/#download)
* [Salicon](http://salicon.net/challenge-2017/)

Datasets were merged to extract different context for analysis of the data.

### Results
All of the statistical results are stored in the following folder:
* [results/plots](https://github.com/alTeska/NISE_saliency/results/plots)

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

## Sample saliencies:

<figure>
<p align="center">
    <img src="results/plots/face_addition_baby.jpg" width="90%">

</p>
<figcaption>
Saliency maps with different value of the face addition factor.
</figcaption><br>

<figure>
<p align="center">
    <img src="/results/sample_saliency/42482_original.jpg" width="220">
    <img src="/results/sample_saliency/42482_aim.jpg" width="220">
    <img src="/results/sample_saliency/cas.jpg" width="220">
    <img src="/results/sample_saliency/covsal.jpg" width="220">
    <img src="/results/sample_saliency/deepgaze.jpg" width="220">
    <img src="/results/sample_saliency/gbvs.jpg" width="220">
    <img src="/results/sample_saliency/icf.jpg" width="220">
    <img src="/results/sample_saliency/ittikoch.jpg" width="220">
    <img src="/results/sample_saliency/sun.jpg" width="220">
    <!-- <img src="/results/sample_saliency/42482_faces.jpg" width="320"> -->
</p>
<figcaption>
  Examples for the saliency maps predicted by the different models. In order: original, AIM, CAS, CovSal, DeepGazeII, GBVS, ICF, Itti&Koch, SUN.
</figcaption>

</figure>
