# Kaggle competitions project
This is a python project that contains scripts for machine learning. They are used for training AI models with training datasets so that they can make accurate predictions.
## Datasets present used in the project
The datasets used in this repository are used from the kaggle organisation website which has a vast collection of datasets and the ones for this repository were got from the following kaggle competitions;
-Organic Compound Thermophysical property competition
-Student Exam score prediction competition
-Space titanic competition
There was also use of the fashion mnist dataset in tensorflow for training an ai model to classify different clothes

## Overview of Machine Learning
Machine Learning is a field that intersects AI,computer science and statistics. It can be thought of as teaching a computer to solve a problem by introducing it to data so that it can "learn" from it and gain insights that are then used to make predictions about some new data it is provided it
## Flow of a machine learning program
So a machine learning program first of all takes a dataset and processes the information to a format that can be used throughout the program.
After this step, the ai model is "trained" with the data from the dataset so that it "learns" patterns from it to be used in making predictions
From here we provide new data to the model so that it can use it to make predictions e.g(providing different features to an ai model to predict a student's exam score such as the age, gender, course,study hours e.t.c)

## Dependencies
These are the different libraries used in python to design and run the code so as to achieve the desired output
-scikit learn(traditional machine learning models)
-tensorflow(deep learning)
-pandas(manipulating datasets)
matplotlib(visualising data)


## Problems Solved with Machine Learning
We can put them into two groups; regression problems and classification problems.
# Regression Problem
This is a problem whereby our ml algorithmn is expected to learn from data and give a prediction that is a continuous value.Examples are the predictions made by the model when predictionns are made for student exam_score prediction whereby the prediction is a value e.g 79.1,76,99 e.t.c
# Classification Problem
This is a problem whereby your model uses the data to make a prediction about a group to which it belongs. Say it is to predict the type of clothing then it takes in data with features such as length of clothing, number of openings, width and others all these are used to assign it to one of the different types of clothings.This is different from a regression problem where a "score" has to be given as in a classification problem you are simply putting it into a praticular group.

# Types of Machine Learning algorithmns
-Supervised learning
-Unsupervised learning
-Reinforcement learning

This repository only contains code for Supervised learning algorithmns.These are the algorithmns which you provide data that contains the features(information used to make the predictions) and the target(what is to be predicted).This is similar to a teacher who provides information about different cases and tells the student what is right and what is wrong and later provides the student with new unseen data for him to decide what is right and what is wrong.
 