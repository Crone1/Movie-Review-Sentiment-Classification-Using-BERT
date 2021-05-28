# Overview
This repository contains the work I did in using a BERT transformer for classification of movie review sentiment.
It also attempts to compare this transformer model to more traditional machine learning models looking at the strengths and weaknesses of these models along with their similarities and differences. This builds on from work I did in my repository on [movie sentiment analysis using sklearn models](https://github.com/Crone1/Movie-Review-Sentiment-Classification-Using-Sklearn).
This project was completed as part of a college module 'Natural Language Technologies'.


# File Description

### Data

The *'Data'* folder contains the original data used in this modelling process in '.tar' format.
This is handled in the notebook and turned into a corpus of both positive and negative documents.
If this *'Data'* folder is empty, the *'chromedriver.exe'* file is needed to scrape the data from the internet.
This driver may be out of date depending on th version of google chrome you have installed.
This issue can be solved by downloading the updated version of chromedriver that matches your version of chrome here.


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
