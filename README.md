# Sentiment Analysis on IMDb Reviews

## Overview
This project performs sentiment analysis on IMDb movie reviews using various machine learning models. The dataset consists of movie reviews labeled as either positive or negative. The primary objective is to build and evaluate models capable of accurately predicting the sentiment of the reviews.

## Table of Contents
- [Introduction](#introduction)
- [Feature Extraction](#feature-extraction)
- [Classification Algorithms](#classification-algorithms)
- [Preprocessing](#preprocessing)
- [Results](#results)
- [Files](#files)
- [Usage](#usage)
- [References](#references)

## Introduction
The IMDb dataset used in this analysis comprises 50,000 movie reviews labeled with sentiments. For this project, a subset of 5,000 reviews was utilized to build and evaluate different sentiment classification models. The goal is to determine the most effective feature extraction and classification techniques for sentiment analysis.

## Feature Extraction
Two feature extraction techniques were used to convert text data into numerical features:
1. **Count Vectorization (CV):**
   - Converts text documents into a matrix representation of word frequency.
   - Generates a feature matrix with 5,000 features.

2. **TF-IDF Vectorization:**
   - Calculates the importance of each word in a document relative to the entire corpus.
   - Generates a feature matrix with 5,000 features using weighted scores.

## Classification Algorithms
The following classification algorithms were evaluated:
1. **Logistic Regression:**
   - A linear model for binary classification tasks.
   - Uses the logistic function to model the relationship between the target and feature variables.

2. **Support Vector Machine (SVM):**
   - A powerful supervised learning algorithm for classification.
   - Finds the optimal hyperplane that maximizes the margin between classes.

3. **Random Forest:**
   - An ensemble learning technique that builds multiple decision trees.
   - Produces the class mode of individual trees for classification tasks.

4. **Gradient Boosting Classifier:**
   - An ensemble learning technique that builds a strong predictive model by combining weak learners iteratively.

## Preprocessing
The preprocessing steps include:
1. **Loading the Dataset:**
   - Loaded the IMDb dataset containing reviews and sentiment labels using Pandas.
   
2. **Data Cleaning:**
   - Removed HTML tags using regular expressions.
   - Removed non-alphanumeric characters and converted text to lowercase.
   
3. **Tokenization and Stopword Removal:**
   - Split the text into individual words.
   - Removed common stopwords to reduce noise.
   
4. **Lemmatization:**
   - Reduced words to their base or dictionary form.
   
5. **Subset Selection:**
   - Randomly selected a subset of 5,000 rows from the preprocessed dataset.
   
6. **Saving Preprocessed Data:**
   - Saved the processed text and sentiment labels to a CSV file for future use.

## Results
### Count Vectorization
- **Imbalanced Dataset:**
  - Logistic Regression: F1-Score: 0.8330, Accuracy: 0.8330
  - SVM: F1-Score: 0.8351, Accuracy: 0.8266
  - Random Forest: F1-Score: 0.8340, Accuracy: 0.8308
  - Gradient Boosting: F1-Score: 0.8198, Accuracy: 0.8088
  
- **Balanced Dataset:**
  - Logistic Regression: F1-Score: 0.8332, Accuracy: 0.8333
  - SVM: F1-Score: 0.8231, Accuracy: 0.8233
  - Random Forest: F1-Score: 0.8292, Accuracy: 0.8268
  - Gradient Boosting: F1-Score: 0.8048, Accuracy: 0.8052

### TF-IDF Vectorization
- **Imbalanced Dataset:**
  - Logistic Regression: F1-Score: 0.8560, Accuracy: 0.8518
  - SVM: F1-Score: 0.8573, Accuracy: 0.8528
  - Random Forest: F1-Score: 0.8308, Accuracy: 0.8276
  - Gradient Boosting: F1-Score: 0.8161, Accuracy: 0.8036
  
- **Balanced Dataset:**
  - Logistic Regression: F1-Score: 0.8562, Accuracy: 0.8563
  - SVM: F1-Score: 0.8582, Accuracy: 0.8583
  - Random Forest: F1-Score: 0.8240, Accuracy: 0.8280
  - Gradient Boosting: F1-Score: 0.8010, Accuracy: 0.8010

## Files
- `CV_&_TF_IDF_IMbalanced_&_Balanced.ipynb`: Jupyter Notebook containing the code for feature extraction, model training, and evaluation.
- `Sentiment Analysis on IMDB Reviews.pdf`: Report detailing the project and results.
- `subset_preprocessed_imdb_reviews.csv`: Preprocessed IMDb reviews dataset.

## Usage
To run the code and reproduce the results:
1. Open the Jupyter Notebook `CV_&_TF_IDF_IMbalanced_&_Balanced.ipynb`.
2. Follow the steps in the notebook to perform feature extraction, model training, and evaluation.
3. Adjust parameters and models as needed to explore different configurations.

## References
- IMDb Dataset: [Kaggle IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)
- Scikit-learn Documentation: [Scikit-learn](https://scikit-learn.org/stable/)

---

For further details on the methodology and results, please refer to the `Sentiment Analysis on IMDB Reviews.pdf` report.
