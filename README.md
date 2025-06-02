# Implementation of "Prediction of Electricity Generation Using Onshore Wind and Solar Energy in Germany"

This repository contains the implementation of the models and methods described in the paper "Prediction of Electricity Generation Using Onshore Wind and Solar Energy in Germany" published in MDPI Energies.

## Overview

The paper compares several deep learning models for forecasting electricity generation from renewable energy sources:

1. Five Transformer-based models:
   - Standard Transformer
   - Informer
   - Autoformer
   - FEDformer
   - Non-Stationary Transformer

2. Baseline models:
   - LSTM (Long Short-Term Memory)
   - ARIMA (Autoregressive Integrated Moving Average)


## Model Descriptions

### FEDformer
The FEDformer model uses frequency-enhanced decomposition and Fourier transform for time series forecasting.