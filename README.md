# ML-Multilabel_Classification

## Introduction

This repository contains examples of best practices for solving Machine Learning Multilabel Classification problems throughout the entire machine learning product life cycle, from data analysis to placing the final machine learning model in production. The code is organized into Jupyter notebooks, with each notebook focusing on a specific classification technique or dataset.

My goal is to provide a clear and concise set of examples that can help you learn how to effectively tackle classification problems, and to guide you through each step of the process, from data exploration and model training to deployment and monitoring. I believe that understanding the entire process of developing a machine learning model, from data cleaning and preprocessing to deploying and maintaining the model in a production environment, is crucial for creating successful machine learning products.

In addition to the notebooks, I've also included a set of scripts and utilities that can help you automate common tasks in the machine learning product life cycle. These tools are designed to help you streamline your workflow and save time, allowing you to focus on building better models.

Whether you're just getting started with Machine Learning or you're a seasoned practitioner, I believe you'll find something of value in this repository. I encourage you to explore the notebooks, try out the code for yourself, and let me know if you have any questions or feedback.

## Dataset

The dataset used in this project is the "Reuters" dataset, which was obtained from OpenML. This dataset contains information about news articles from the Reuters newswire service. The dataset includes 21,578 documents from 1987, with each document belonging to one or more categories such as "earnings," "acquisitions," "money-fx," and "grain."

The Reuters dataset is a widely used benchmark for text classification tasks and provides a rich source of data for natural language processing applications. With over 20,000 documents and a vocabulary of nearly 30,000 words, this dataset allows for the exploration of complex techniques in text analysis and classification. Additionally, the inclusion of multiple categories allows for the development of models that can accurately classify documents into one or more categories, making it a valuable resource for both academic and commercial use.

## 1 - Preprocessing dataset

The first step in solving a machine learning classification problem is to preprocess the dataset. Using the Pandas library, we load the "reuters.csv" file and then analyze it to determine which features are numerical, which are categorical, and which is the target variable. It's essential to identify and handle any missing values or NaNs in the dataset, either by replacing or removing them.

The next critical step is to transform categorical features into numerical ones using OneHotEncoder and scale numerical features using Standarscaler. Additionally, we need to transform the classes present in the target variable into numerical labels using LabelEncoder. After completing these preprocessing steps, we can save the datasets ready for use by the Sklearn machine learning libraries.

In this case, the dataset is already prepared for the machine learning process, the data has already been encoded. However, we still need to transform the values of the TRUE and FALSE labels into 1 and 0 to be able to use the data with machine learning models. To do this, we can use the LabelEncoder class of Sklearn. After completing these preprocessing steps, the dataset is ready to be used with Sklearn's machine learning models.

## 2 - Classification lab

Once we have preprocessed our reuters dataset, the next step is to train and test various classification models with Sklearn into MultiOutputClassifier module. By loading the dataset in the X and y format, we can create a training set and a testing set, ensuring that our models can generalize to unseen data.

We can then instantiate multiple multilabel classification models and fit them to our training set. Some of the models we can use include Logistic Regression, Decision Trees, Random Forests, Support Vector Machines (SVMs), and K-Nearest Neighbors (KNN).

After training the models, we can evaluate their performances on the testing set by calculating metrics such as precision, recall, accuracy, and F1 score. 

Through this process, we can identify the most promising machine learning models for our dataset and select the one that performs best in terms of precision, recall, and overall accuracy. By selecting the best-performing model, we can build a reliable and effective machine learning classifier that can make predictions in production environments.

## 3 - Model optimization

Once we have identified the most promising model, it's time to integrate it into a pipeline to achieve two results:

1) Transform the input data that we want to predict using our machine learning model.
2) Perform a Grid Search process to search for the best hyperparameters.

By integrating the model into a pipeline, we can streamline the process of making predictions on new data and optimize the model's performance through hyperparameter tuning.

The Grid Search process involves selecting a set of hyperparameters and exhaustively searching for the combination of hyperparameters that result in the best performance of the model. This process can be computationally expensive, but Sklearn provides efficient tools to perform Grid Search on a range of hyperparameters.

Once we have completed this process, we can compare the performance of the base model and the optimized model to see if the hyperparameter tuning has improved the model's performance. This comparison can be done by evaluating metrics such as accuracy, precision, recall, and F1-score.

In summary, model optimization involves integrating the most promising model into a pipeline, performing hyperparameter tuning through Grid Search, and comparing the performance of the base and optimized models. This process can significantly improve the performance of the machine learning model, making it more accurate and suitable for production-level predictions.

## 4 - Pipeline for production

In the final step, we reconstruct the pipeline of our model by including the necessary input data transformations and the previously optimized model with the best hyperparameters. We can then train the pipeline one last time on the X and y data of our reuters dataset and save it to disk using the joblib library.

At this point, our multilabel classification predictive model is ready to be integrated into any production application. The pipeline can be easily deployed to a cloud environment, such as Amazon Web Services (AWS), Google Cloud Platform (GCP) or Microsoft Azure, or to an on-premises server. Once deployed, the pipeline can receive input data and provide predictions in real-time, enabling us to make accurate decisions and automate processes based on the predictions made by our machine learning model.

## Final notes

You can use these four steps outlined in the Jupyter Notebook worksheets for any multilabel classification work. Of course, these should be understood as a solid starting point that needs to be adapted to the needs of your project. However, from personal experience, they can significantly accelerate the workflow for the entire life cycle of the predictive machine learning model.

Happy multilabel classifying!
