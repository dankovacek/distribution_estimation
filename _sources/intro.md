# Introduction

The duration curve represents the proportion of time an observed variable exceeds some value.  In hydrology, the flow duration curve (FDC) is used in applications related to water resource planning as it represents a long-term expectation of water availability at some location.  Predicting long-term water resource availability at ungauged locations is an open problem in hydrology.  In this notebook, we present several approaches to estimating FDCs at ungauged locations.  Using a large sample of monitored locations around British Columbia, Canada, we test predicting the FDC for each location using a range of input information and methods, and test the accuracy of the prediction using the observed data.

This site contains the code and additional supporting information for a computational experiment exploring the estimation of probability distributions (flow duration curves) for ungauged locations.  The three methods tested are as follows:

1. **Parametric Probability Distribution**: use XGboost to predict sufficient statistics of the log-normal (LN) distribution using catchment attributes.  This is done in two different ways, first by predicting mean and standard deviation of streamflow (per unit area) and estimating the LN location and scale by method of moments (MoM), and second by predicting the LN location and scale directly.  
2. **k-Nearest Neighbours**: estimate unit area runoff at an ungauged location using daily streamflow from an ensemble of nearest neighbours.  Here we test the ensemble size $k$, the criteria used for selecting the ensemble, and the weights or proportions assigned to ensemble members reflecting their strength of influence.
3. **Neural Network**: train a neural network model (LSTM) to predict daily streamflow from daily meteorological time series inputs (precipitation, temperature, shortwave radiation, vapour pressure, snow water equivalent). 


```{figure} ../images/method_flowchart.jpeg
---
alt: Flowchart describing the methodology of the experiment.
name: method-flowchart
width: 500px
align: center
---
The full experimental workflow from data preprocessing for three different model inputs that are each used to generate an estimated flow distribution, and the subsequent large-sample analysis.  
```

The experiments are organized in the following chapters (notebooks), corresponding to the workflow described in the figure above:

1.  **Data**: introduce the input data and describe where to get additional data from outside sources.
2.  **Methods**: introduce the various methods for estimating flow duration curves, define partitions for 5-fold CV and compare the target variable distribution over validation folds.
3.  **Predict Runoff Statistics**: The hydrological signature prediction is done first to generate the predicted sufficient statistics of the log-normal distribution for the sample of catchments.
4.  **Computation of reference distributions**: to evaluate the accuracy of estimated FDCs, we first compute a reference distribution on the observed data.  This is done by kernel density estimation (KDE).  The pre-processing of reference distribution makes the nearest neighbours experiment much more efficient in avoiding duplicate computations.
5.  **FDC Estimation Validation**: here we iterate through all catchments in the sample and bring together the FDC estimates of the different approaches.
6.  **Results Explorer**: The results are compiled to compare FDC estimation methods over the large sample.  

## Computational Notes

The prediction of hydrological signatures is done here with the widely used XGBoost library is used for its implementation{cite}`chen2016xgboost`.  (notes about xgboost).  The XGBoost library facilitates parallel CPU and GPU training, making it feasible to run a large number of models to test sensitivity to key assumptions.  The neural network estimation component is done using the [Neural Hydrology](https://github.com/neuralhydrology/neuralhydrology) library from {cite:p}`kratzert2022joss`.

The computation-intensive pre-processing steps can be bypassed by downloading the pre-processed data files as described in each experiment notebook.

## Contents of this book

```{tableofcontents}
```

## Citations 

```{bibliography}
:filter: docname in docnames
```
