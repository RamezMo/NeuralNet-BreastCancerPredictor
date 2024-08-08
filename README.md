# Breast Cancer-Predictor-Model : Deep Learning Model for Predicting Breast Cancer with nearly 98.3% accuracy

## Introduction

In the realm of healthcare, predictive modeling is instrumental in early detection and intervention for various medical conditions. This project showcases the use of neural network techniques to predict breast cancer outcomes using a dataset from Kaggle. By analyzing this dataset, this model aims to support healthcare professionals in accurately identifying and diagnosing breast cancer at an early stage, ultimately improving patient outcomes and treatment efficacy.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#Data-Preparation)
3. [Exploratory Data Analysis (EDA)](#Exploratory-Data-Analysis) 
4. [Model Evaluation](#Evaluate-models)

# Data Preparation

### Importing Necessary Libraries

First, import the necessary libraries for data analysis and machine learning.

```python
#Importing Needed Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
```

## Load dataset using full path
Load the dataset into DataFrame.

```python
cancer_data = pd.read_csv('/kaggle/input/breast-cancer-dataset/breast-cancer.csv')
```

## Display the first few rows of the dataset
inspect the first few rows to understand its structure.
```python
cancer_data.head(10)
```

![image](https://github.com/RamezMo/Stroke-Predictor-Model/assets/142325393/ce9231b7-af35-4f5e-b174-fecf6783d475)


## How Many Instances and Features ?
Check for and handle missing values to ensure a clean dataset.
```python
cancer_data.shape
```

#Exploratory-Data-Analysis

## Checking for missing values
Check for and handle missing values to ensure a clean dataset.
```python
cancer_data.isnull().sum() 
```

![image](https://github.com/user-attachments/assets/29bf853f-afca-4dc5-ad0c-b333cd632b3d)


##Display Variables data type and number of non NULLs values in
Notice that all columns 'except diagnosis column have numerical DataType'
Notice that all columns have 0 NULLs
```python
cancer_data.info()
```
![image](https://github.com/user-attachments/assets/f73bdb4b-e499-46bf-b2c4-fbdfaaeaf633)


## Didscover if Target column values are Balanced or not 
```python
cancer_data.diagnosis.value_counts()
```
![image](https://github.com/user-attachments/assets/a2fbd05a-12eb-48be-9aa3-9e16f7bb2025)


#data preperation
## Data Transformation

Convert categorical variables into numerical ones for Deep learning models.

```python
cancer_data.replace({'B':1,'M':0},inplace=True)
```

## Summary statistics

```python
cancer_data.describe()
```

![image](https://github.com/user-attachments/assets/886c0e8e-1aaf-4eeb-ba92-759f574e173b)


## Split data into training and testing sets

```python
x = cancer_data.drop('diagnosis',axis=1)
y = cancer_data.diagnosis
```



## Standardize The Data
Before passing the data into the Neural Network model, we need to scale it to get better accuracies
```python
scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)
x_test_std = scaler.transform(x_test)
```

##Building the Neural Network
Setting up the layers of Neural Network
```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(31,)),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')    
])
```
## #Visualizing accuracy

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training data','validation data'],loc='lower right')])
```

![image](https://github.com/user-attachments/assets/eddb239a-d39d-42f3-9e6a-54f1e45d37f8)


#Visualizing loss
```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training data','validation data'],loc='upper right')
```

![image](https://github.com/user-attachments/assets/83ea6b34-35f5-47d7-9d8a-9ea26b76dbd3)


## Evaluate models
after training the model and predicting it on test data it makes accuracy of nearly 98.3%
![image](https://github.com/user-attachments/assets/a931ed00-3e5c-49c3-94f6-74aed2a9069f)


