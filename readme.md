This repository contains Jupyter Notebooks that I used to learn the fundamentals of machine learning. These notebooks explore algorithms and techniques commonly used in supervised learning tasks.

# 1.  Supervised Machine Learning with KNN and Decision Trees
This Python script explores K-Nearest Neighbors (KNN) and Decision Tree algorithms for supervised machine learning tasks.
This includes
1. Understanding the fundamentals of KNN and Decision Tree algorithms.
2. Implement these algorithms in Python using the Iris flower dataset.
3. Visualize and evaluate the performance of the models.


## 1.1 Data Loading and Preprocessing

The script imports the Iris dataset from scikit-learn and performs the following preprocessing steps:

Split the data into features (X) and target variables (y).
Visualize the data using pairplots and 3D scatter plots for dimensionality reduction.
Handle outliers in the 'sepal_width' column to improve model accuracy.
Split the data into training and testing sets using train_test_split with shuffling and a 20% test size.
Normalize the features using a Normalizer to bring all values into the range of [0, 1].
## 1.2 Training and Predicting

A KNN classifier is instantiated with n_neighbors=3.
The model is fitted to the training set using knn_classifier.fit(X_train, y_train).
Predictions are made on the test set using knn_classifier.predict(X_test).
## 1.3 Evaluation

A confusion matrix is generated using confusion_matrix to visualize the model's performance.
Accuracy is calculated using accuracy_score and displayed as a percentage.
## 1.4 Hyperparameter Tuning

The script explores the impact of the 'n_neighbors' parameter on KNN performance using cross-validation.
It iterates through a range of k values, performs 10-fold cross-validation, and plots the misclassification error.
The optimal number of neighbors (k) is identified based on the minimum misclassification error.
