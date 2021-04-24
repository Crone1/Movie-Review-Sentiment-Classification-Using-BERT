
# general packages
import time
import numpy as np
import pandas as pd

# for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

# models needed for modelling
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Class to define the model
# -----------------------------

class Model:
    
    def __init__(self, model, ngram_size=1, clip_counts=True):

        # define the variables needed to train the model
        self.vocab = set()
        self.token_to_vocab_index_dict = {}

        # define the parameters of this model instance
        self.model = model
        self.ngram_size = ngram_size
        self.clip_counts = clip_counts

    def extract_document_features(self, data_with_labels):

        '''
        Create a matrix of features where the rows are all the documents and the columns are all the words
        We populate the matrix based on the occurances of the words in the documents

        Params:
            self: instance of Model class
            data_with_labels: list - this is a list of tuples containing a document and it's corresponging label

        Returns:
            numpy array - this is the populated matrix of features which corresponds to the documents X the words in the vocabulary
        '''

        # create numpy array of required size
        columns = len(self.vocab)
        rows = len(data_with_labels)
        feature_matrix = np.zeros((rows, columns), dtype=np.int32)

        # Populate feature matrix
        for doc_index, (document, _) in enumerate(data_with_labels):
            for sentence in document:

                for i in range(len(sentence) - (self.ngram_size - 1)):
                    # get each word pattern of size 'self.ngram_size'
                    token = " ".join(sentence[i: i+self.ngram_size])

                    # get the vocab index for this row
                    try:
                        vocab_index = self.token_to_vocab_index_dict[token]
                    except KeyError:
                        # token not in vocab --> skip this token & continue with next token
                        continue

                    # Populate the individual value in the feature matrix
                    if self.clip_counts:
                        feature_matrix[doc_index, vocab_index] = 1
                    else:
                        feature_matrix[doc_index, vocab_index] += 1

        return feature_matrix

    def train(self, data_with_labels):
        
        '''
        Train the model on the given data

        Params:
            self: instance of Model class
            data_with_labels: list - this is a list of tuples containing a document and it's corresponding label

        Returns:
            None
        '''

        # define a set of all the vocabulary in the data using the input list of documents
        for document, _ in data_with_labels:
            for sentence in document:
                for i in range(len(sentence) - (self.ngram_size - 1)):
                    # add each word pattern of size 'self.ngram_size'
                    token = " ".join(sentence[i: i+self.ngram_size])
                    self.vocab.add(token)

        # create reverse map for fast token lookup
        for vocab_index, token in enumerate(self.vocab):
            self.token_to_vocab_index_dict[token] = vocab_index

        # extract the features - create numpy array of required size
        extracted_features = self.extract_document_features(data_with_labels)

        # create column vector with target labels - prepare target vector
        targets = get_targets(data_with_labels)

        # train the model on the features - pass numpy array to sklearn to train NB
        self.model.fit(extracted_features, targets)

    def predict(self, test_data):

        '''
        Take in some test documents and generate a predicted label for these documents using the trained model

        Params:
            self: instance of Model class
            test_data: list - a list of documents and their corresponding labels

        Returns:
            list - this is a list containing the models predicted labels for the given test set
        '''

        # get the features of the document
        features = self.extract_document_features(test_data)

        # Get the predictions for model
        y_pred = self.model.predict(features)

        # Define the prediction labels
        labels = []
        for is_positive in y_pred:
            if is_positive:
                labels.append('pos')
            else:
                labels.append('neg')

        return labels


def get_targets(data_with_labels):

    '''
    Create a column vector of the target labels for all the documents in this dictionary of data

    Params:
        data_with_labels: list - this is a list of tuples containing a document and it's corresponding label

    Returns:
        numpy array - a one column array of 1's and 0's stating if a document is labelled as pos/neg respectively
    '''

    # prepare target vector
    targets = np.zeros(len(data_with_labels), dtype=np.int8)

    # iterate through the list of documents & labels and populate the target vector
    index = 0
    for _, label in data_with_labels:
        if label == 'pos':
            targets[index] = 1
        index += 1

    return targets


def evaluate_predictions(test_data, y_pred):

    '''
    Evaluate the predicted labels for a given test set of data using the actual labels for this set

    Params:
        test_data: list - a list of documents and their corresponding labels
        y_pred: list - this is a list containing the models predicted labels for the given test set

    Returns:
        tuple - (an accuracy score for the predictions, a confusion matrix evaluating the predictions)
    '''

    # turn the predicted variables into a binary 1 or 0 based on 'pos' or 'neg'
    binary_y_pred = np.where(np.array(y_pred) == 'pos', 1, 0)

    # generate the actual y values for this test data
    y_true = get_targets(test_data)

    # get an accuracy score for these predictions and also get a confusion matrix
    return accuracy_score(y_true, binary_y_pred), confusion_matrix(y_true, binary_y_pred)


def create_fold_prediction_eval_table(test_data, predictions):

    # define a dataframe to house the data's prediction evalutaions
    prediction_eval_df = pd.DataFrame(columns=['index', 'real_class', 'predicted_class', 'correct', 'tested_document'],
                                      index=list(range(len(test_data))))

    # iterate through the test dataset and populate the dataframe with the predictions
    for i, ((doc_list, label), prediction) in enumerate(zip(test_data, predictions)):

        # turn the document list into a coherent looking document
        list_of_sentences = []
        for tokens_list in doc_list:
            sentence = " ".join(tokens_list)
            list_of_sentences.append(sentence)
        str_doc = "\n".join(list_of_sentences)

        # populate the dataframe
        prediction_eval_df.loc[i, "index"] = i
        prediction_eval_df.loc[i, "real_class"] = label
        prediction_eval_df.loc[i, "predicted_class"] = prediction
        prediction_eval_df.loc[i, "correct"] = "yes" if prediction == label else "no"
        prediction_eval_df.loc[i, "tested_document"] = str_doc

    return prediction_eval_df


def evaluate_model(model, model_name, train_test_tuples, fold_verbose=False):

    '''
    Evaluate the model by using cross validation to train the model on the different training sets and then generating a prediction on the test sets
    
    Params:
        model: class instance - an instance of the Model class defining a model with specified parameters
        model_params: dictionary - this dictionary maps the parameter name to the name of the parameter configuration of the model we are evaluating
        train_test_tuples: list: - a list of 'k' training and test set tuples - eg. (training_data_1, test_data_1)
        fold_verbose: boolean - whether we want to print the model accuracy scores of each fold as we go along in trining the model on them
        plot_folds: boolean - whether we want to output a plot of the time each fold took and the accuracy each fold god

    Returns:
        dataframe - a dataframe detailing the parameters of the model along with the evaluation scores for that model and the time it took to train & evaluate on all of the cross validation folds
                  - this basically summarises the whole evaluation process
    '''

    # set up the dataframe we output with the evaluation summary statistics
    fold_eval_df = pd.DataFrame(columns=['fold_no', 'fold_time', 'fold_accuracy'], index=range(len(train_test_tuples)))

    # iterate through the cross validation train-test sets
    fold_accuracies, fold_durations = [], []
    fold_doc_pred_tables = []
    eval_start = time.time()
    for i, (train_data, test_data) in enumerate(train_test_tuples):
        iteration_start = time.time()
        if fold_verbose:
            print("   Fold {}".format(i + 1), end="")

        # train the model on each set as we iterate
        model.train(train_data)

        # get an accuracy score for the trained model
        predictions = model.predict(test_data)
        accuracy, _ = evaluate_predictions(test_data, predictions)
        fold_accuracies.append(accuracy)
        iteration_duration = time.time() - iteration_start

        # store the documents being predicted
        fold_pred_eval_table = create_fold_prediction_eval_table(test_data, predictions)
        fold_doc_pred_tables.append(fold_pred_eval_table)

        # add the results from this fold to the fold evaluation dataframe
        fold_eval_df.loc[i, 'fold_no'] = i + 1
        fold_eval_df.loc[i, 'fold_time'] = iteration_duration
        fold_eval_df.loc[i, 'fold_accuracy'] = accuracy

        # output the model accuracy if requested
        if fold_verbose:
            print(" - {} seconds --> Accuracy score = {}".format(round(iteration_duration), accuracy))

    # create the evaluation summary statistics for this model
    n = float(len(fold_accuracies))
    avg = sum(fold_accuracies) / n
    variance = sum([(x-avg)**2 for x in fold_accuracies]) / n
    eval_duration = time.time() - eval_start

    # create a dataframe with one row summarising the model evaluation
    eval_values = {'Full Name': model_name,
                   'Avg Accuracy': avg,
                   'Accuracy Std Dev': variance**0.5,
                   'Min Accuracy': min(fold_accuracies),
                   'Max Accuracy': max(fold_accuracies),
                   'Total Time (s)': round(eval_duration, 2),
                   'All Fold Averages': str(fold_accuracies)
                  }

    return pd.DataFrame(eval_values, index=[0]), fold_eval_df, fold_doc_pred_tables
