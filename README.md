# NLPSupportVectorMachine
Natural Language Processing (NLP) spam/ham email classification via full custom Support Vector Machine (hard/soft margin, linear/polynomial/Gaussian kernel) and Principal Component Analysis / Linear Discriminant Analysis implementation.

__For usage instrution please check [Usage README](https://github.com/SamyuelDanyo/NLPSupportVectorMachine/blob/master/docs/README.txt)__

__For full documentation - system design, experiemnts & findings please read [NLPSupportVectorMachineDoc Report](https://github.com/SamyuelDanyo/NLPSupportVectorMachine/blob/master/docs/NLPSupportVectorMachineDoc.pdf)__

![SVM Fit LDA 2-D Projection](/res/LOG_LDA_Soft_0.1_Pol_3_SVM_plot.PNG)

## Introduction
In this report I present my implementation of the Support Vector Machine (SVM) classification algorithm. In addition, two pattern recognition methods: Principal Component Analysis (PCA) & Linear Discriminant Analysis (LDA), are utilized for transforming the data-space into, a better formed, feature space. The three methods are full-custom implementations in Python. Additionally, visualizations, classification results and analysis of the SPAM E-mail Dataset, are presented. 

The goal of the project is to study the effectiveness of SVM for classification, as well as, the effects of the classifier’s configuration & the input data structure on performance.

The dataset is comprised of 4601 samples, each with 57 features. Out of them 1813(39.4%) are labeled as spam, leaving 2248(60.6%) as non-spam. The full description of the SPAM E-mail dataset and clarification on what is assumed as spam can be seen in [1].

The dataset is randomized and divided into three sub-sets – train with 2000(43.5%) samples, test with 1536(33.4%) samples & validation with 1065(23.1%) samples. Three preprocessing techniques are applied to the dataset in order to condition it better for classification: standardization, binarization & log-transform.

For classification analysis of the model fit, a few different metrics are observed: accuracy, overall error rate(1-accuracy), false positives error rate and false negatives error rate.

While the main metrics for evaluating the performance of the methods are the resultant accuracy/overall error rate, for the specific case of the SPAM E-mail dataset – a deeper insight can be drawn from the false positives/negatives error rate. As mentioned by [1], the false positives (classifying a non-spam email as spam) is very undesirable as this can lead to the loss of important correspondence. Hence, when I discuss the performance of the model fits, a special attention is given to the false positives rate with the aim of minimizing it.

Additionally, the effect of data preprocessing & feature space transform are studied. Various configurations between the preprocessing functions: standardization; normalization; binarization; log transform; & the feature extraction techniques (data transformation): PCA; LDA; (with different dimensionalities), were examined. Three configurations were selected to be presented in this report: binarization + No transform (BNY), standardization + 15-D PCA (STD_PCA), log + 2-D LDA (LOG_LDA).

Furthermore, the effect that some hyperparameters have on the SVM model fit are studied: type of kernel (linear, polynomial & gaussian); type of margin (hard vs soft for polynomial kernel); the degree of the polynomial kernel; the value of the penalty parameter C (cost of violating constraints) for polynomial kernel.

The methods are implemented mainly through using NumPy for matrices manipulation and calculations. The CVXOPT package is used for solving the dual quadratic problem for SVM. Shared helper functions are implemented for preprocessing the dataset, visualizing the LDA feature space, building classification confusion tables and extracting the evaluation metrics. Further method-specific helper functions are implemented for wrapped training, testing, displaying results and complete fitting using matplotlib for plotting graphs and some other basic helper packages.

## PCA for Feature Extraction
### Design
The transform aims to find the directions (components) that maximize the variance of the input (training) dataset. PCA projects the input data space onto a sub-feature space with minimum information loss.

## LDA for Feature Extraction & Visualization
### Design
The transform aims to find the directions that maximize the variance (separation) between classes & minimize the variance (difference) inside the classes (represented by maximizing the Fisher’s ratio: __*max( (m1-m2)**2 / (v1+v2) )*__. LDA projects an input data space onto a sub-feature space with minimum class-discriminatory information loss.

## SVM for Classification
### Design
The SVM classifier aims to find the optimal hyperplane decision boundary (separation) between two classes {-1,1}. It approaches the task by maximizing the margin (distance) between the data and the hyperplane (the hyperplane can be shifted to make sure it separates the margin equally). The vectors going through the points between which is the margin are called support vectors. Maximizing the margin is proven to yield the optimal separation (leads to lower probability of misclassification) by Cristianini and Shawe-Taylor. Formulated as optimization problem, where we try to maximize the normalized (geometric) margin, while keeping the absolute (functional) margin the same (=1), leading to minimization of the weights (hyperplane parameters). The problem can be summed up as:

__*min(1/2W.T*W)
subject to: d*(W.T*X+b)>=1 (d = class) 
find: w,b*__

Such primal optimization problem can be approached, using Lagrange Theorem, aiming to find the equality and inequality Lagrange multipliers. The optimization problem can be further remodeled, using the Kuhn-Tucker Theorem, to transform it from primal to dual problem, where a single parameter (the Lagrange multipliers) yields the solution. The dual problem is:

__*min(W=a*d*SV (SV=X for X which a=>0) or max(ai -1/2ai*aj*di*dj*Xi.T*Xj),
subject to: a*d=0
            a=>0
Find: a*__

In order to solve the problem, dual quadratic programming is required, which yields the Lagrange multipliers and hence the hyperplane parameters.

The obvious problem though is that this is a linear separation. Meaning it will work for linearly separable classes. In order to improve the performance on overlapping data, the soft margin is introduced. The concept allows for data-points to be inside the margin (or on the wrong side of it), by introducing the slack variables, which added together and scaled by ‘C’ (the cost of violating the constraints) are the error penalty. The optimization problem aims not only to maximize the margin but also minimize the error:

__*min(1/2W.T*W + C*slack) 
subject to: d*(W.T*X+b)>=1 -slack
            slack>=0, hence 0<=a<= C (as part of the dual problem)*__
            
The third SVM concept is the kernel transform (known as the Kernel trick). It is based on Cover's Theorem, which states that: “Probability that classes are linearly separable increases when data points in input space are nonlinearly mapped to a higher dimensional feature space.”. In order to deal with non-linear data (which is not separable in the data space), the kernel transform K(), maps the input data into a feature space, before extracting the support vectors.

## Experimental Setup
All experiments (without evaluation) are performed for each of the three preprocessing/transform configurations (binarization + No transform, standardization + 15-D PCA, log + 2-D LDA) of the dataset.
  __Train & Test | Hard Margin SVM With Linear Kernel – the train set is used to fit SVM model. The test set is used to test its classification performance.__
  
  __Train & Test | Hard Margin SVM With Polynomial Kernel – the train set is used to fit SVM model. The test set is used to test its classification performance. The experiment is performed for p ∈ {2, 3, 4, 5}.__
  
  __Train & Test | Soft Margin SVM With Polynomial Kernel – the train set is used to fit SVM model. The test set is used to test its classification performance. The experiment is performed for p ∈ {1, 2, 3, 4, 5} for C ∈ {0.1, 0.6, 1.1, 2.1}.__
  
  __EVALUATION | Soft Margin (C=1.1) SVM With Gaussian Kernel – the train set is used to fit SVM model. The evaluation set is used to test its classification performance.__

## Resuts
__For full results & obervations please read [NLPSupportVectorMachineDoc Report](https://github.com/SamyuelDanyo/NLPSupportVectorMachine/blob/master/docs/NLPSupportVectorMachineDoc.pdf)__

__For all figures please check [NLPSupportVectorMachine Resources](https://github.com/SamyuelDanyo/NLPSupportVectorMachine/blob/master/res)__

__LDA Dataset Projection__
![LDA Dataset Projection](/res/LOG_LDA_train_plot.png)

__SVM Soft Margin[2.1] Polynomial[2] Kernel Fit__
![SVM Soft Margin Polynomial Kernel Fit](/res/LOG_LDA_Soft_2.1_Pol_2_SVM_plot.PNG)

__SVM Soft Margin[2.1] Polynomial[2] Kernel Fit Confusion Table__
![SVM Soft Margin Polynomial Kernel Fit Confusion Table](/res/LOG_LDA_Soft_2.1_Pol_2_Conf.PNG)

__SVM Soft Margin[0.1] Polynomial[2] Kernel Fit Perofrmance__
![SVM Soft Margin Polynomial Kernel Fit Performance](/res/pol_soft_perf.PNG)

__SVM Fit Binarized Dataset Results__
![SVM Fit Binarized Dataset Results](/res/BNY_Results.PNG)

__SVM Fit Standardized Dataset + PCA Results__
![SVM Fit Standardized Dataset + PCA Results](/res/STD_PCA_Results.PNG)

__SVM Fit Logarithmic-Transformed Dataset + LDA Results__
![SVM Fit Logarithmic-Transformed Dataset + LDA Results](/res/LOG_LDA_Results.PNG)

__SVM Fit Standardized/Logarithmic-Transformed Dataset Results__
![SVM Fit Standardized/Logarithmic-Transformed Dataset Results](/res/STD_LOG_TRANSFORM_EFFECT_Results.PNG)
