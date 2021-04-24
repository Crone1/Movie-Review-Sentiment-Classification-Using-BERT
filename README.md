# Overview

This repository contains my submission for assignment 3 in the module 'Natural Language Processing'.
This project looks at movie review sentiment classification using a BERT transformer and attempts to compare this model to more traditional machine learning models, such as Multinomial Naive Bayes and Logistic Regression.
This project looks at the strengths and weaknesses of these models along with their similarities and their differences.

# File Description


### Data

The data is contained in the *Data* folder.
While the data in this folder is in '.tar' format, this is handled in the notebook and turned into a corpus of both positive and negative documents.

### Code

My code for this analysis was written in Python and is made up of four main files.
These are:

1. *'bert_movie_sentiment_analysis.ipynb'*
   * This is the notebook used to carry out this analysis.
   * This notebook calls from all of the other '.py' files which executing.

2. *'Load_data_functions.py'*
   * This file contains the functions needed to load the data into the notebook and to split it into a train & test set.

3. *'Bert_functions.py'*
   * This file contains the functions related to the BERT model.
   * These functions define how the model is defined, how it is trained, and how the analysis is carried.

4. *'Traditional_model_functions.py'*
   * This file contaisn the functions related to the traditional models used to compare against the BERT model.
   * The functions defined in this file set out the framework for Multinomail Naive Bayes and Logistic Regression to be run.
