
# general packages
import os
import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# for evaluation
from sklearn.metrics import accuracy_score, confusion_matrix

# for plotting
import matplotlib.pyplot as plt

# package for lemmatisation
import spacy

# models needed for modelling
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# packages for sentiment lexicon
from nltk.corpus import opinion_lexicon

# for sorting the column
from pandas.api.types import CategoricalDtype


def get_documents(data_dict, fold, label):

    '''
    Get the documents that fall under a specified fold and label pair

    Params:
        data_dict: dictionary - maps the cross validation fold & document label to the documents that fall under this fold-label
        fold: int - the cross validation fold that we want to get the documents for
        label: string - the label that we want to get the documents for (pos/neg)

    Return:
        list - a list of all the documents in the given dictionary that fall under the specified fold and label key
    '''

    return data_dict[(fold, label)]


def get_document_preview(document, max_length, sep):

    '''
    Preview a given number of characters from a given document

    Params:
        document: list of lists - a list of all tokens in a list of all sentences in a specified document
        max_length: int - the maximum number of characters to be output in our preview string
        sep: string - the string used to join the words in the document together in the output - can be a space (" ") or pipe ("|") or other string

    Returns:
        string - a preview of the specified document that is less than 'max_length' characters long
    '''

    # iterate through the tokens in the document
    char_count = 0
    reached_limit = False
    preview_words = []
    for sentence in document:
        for token in sentence:

            # check if adding this character would create a string bigger than desired
            if char_count + len(token) + len(preview_words) > max_length:
                reached_limit = True
                break

            # add this token to the list of words in our preview
            preview_words.append(token)
            char_count += len(token)

        if reached_limit:
            break

    # create a string from the created list
    return sep.join(preview_words)


def preview_docs_with_fold_label_pair(data_dict, fold, label, max_length=50, sep=" "):

    """
    Get a preview of each document that falls under a specified fold & labal pair.

    Params:
        data_dict: dictionary - this dictionary maps a tuple to a list of lists containing tokens
        fold: int - the cross validation fold that we want to get the documents for
        label: string - the label that we want to get the documents for (pos/neg)
        max_length: int - the maximum number of characters to be output in our preview string - defaults to 50
        sep: string - the string used to join the words in the document together in the output - this defaults to space (" ") but can be pipe ("|") or other string

    Returns:
        dataframe - a table of the # documents associated with the specified fold & label pair, the # sentences in each document, and then the first 'max length' characters in this document.
    """

    preview_df = pd.DataFrame(columns=['doc_num', 'sentences', 'start_of_first_sentence'])
    list_of_documents = get_documents(data_dict, fold, label)
    for doc_num, document in enumerate(list_of_documents):

        # get a preview of this document
        doc_preview = get_document_preview(document, max_length=max_length, sep=sep)

        # add this documents preview to a summary dataframe of all the documents
        one_doc_preview_df = pd.DataFrame({'doc_num': doc_num, 'sentences': len(document), 'start_of_first_sentence': doc_preview}, index=[0])
        preview_df = pd.concat([preview_df, one_doc_preview_df], axis=0).reset_index(drop=True)
        
    return preview_df


# -----------------------------
# Class to define the model
# -----------------------------

class Model:
    
    def __init__(self, model, ngram_size=1, clip_counts=True, add_sent_lex_feat=False):

        # define the variables needed to train the model
        self.vocab = set()
        self.token_to_vocab_index_dict = {}

        # define a list of positive and negative lexicons using an installed package
        self.pos_lex_set = set(opinion_lexicon.positive())
        self.neg_lex_set = set(opinion_lexicon.negative())

        # define the parameters of this model instance
        self.model = model
        self.ngram_size = ngram_size
        self.clip_counts = clip_counts
        self.add_sent_lex_feat = add_sent_lex_feat

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
        columns = len(self.vocab) + (2 if self.add_sent_lex_feat else 0)
        rows = len(data_with_labels)
        feature_matrix = np.zeros((rows, columns), dtype=np.int32)

        # Populate feature matrix
        for doc_index, (document, _) in enumerate(data_with_labels):
            for sentence in document:

                if self.add_sent_lex_feat:
                    # add the sentiment lexicon features - these are unaffected by the clip counts
                    pos_sent_count, neg_sent_count = get_sentence_sentiment(sentence, self.pos_lex_set, self.neg_lex_set)
                    feature_matrix[doc_index, -1] += pos_sent_count
                    feature_matrix[doc_index, -2] += neg_sent_count

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


def get_sentence_sentiment(sentence, pos_lex_set, neg_lex_set):

    """
    Take in a sentence and count the number o positive tokens and the number of negative tokens in this sentence
    
    Params:
        sentence: string - a sentence that we want to analyse the sentiment of
        pos_lex_set: set of strings - a list of positive lexicons
        neg_lex_set: set of strings - a list of the negative lexicons

    Returns:
        tuple of integers - (# positive tokens in the sentence, # positive tokens in the sentence)
    """

    pos_sent = neg_sent = 0
    for word in sentence:
        if word in pos_lex_set:
            pos_sent += 1
        elif word in neg_lex_set:
            neg_sent += 1

    return pos_sent, neg_sent


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


def print_first_n_predictions(model, test_data, num_predictions, len_preview):

    '''
    Use a model to generate a specified number of predictions and output a table containing the predicted label and the actual abel

    Params:
        model: scikit-learn model - a trained model
        test_data: list - a list of documents and their corresponding labels
        num_predictions: int - the number of prediction Vs actual rows we want in our output dataframe
        len_preview: int - the maximum number of characters to be output in our preview string

    Returns:
        pandas dataframe - a dataframe containing the predicted label side by side with the actual label for a given document
    '''

    # get the models predictions for the test data
    predictions = model.predict(test_data)

    # evaluate the generated predictions and output the results
    accuracy, confusion_matrix = evaluate_predictions(test_data, predictions)
    print("Accuracy =", accuracy)
    print("Confusion_matrix:\n", confusion_matrix)

    # create a dataframe displaying a specified number of these predictions
    pred_df = pd.DataFrame(columns=['label', 'prediction', 'documents'])
    for i in range(num_predictions):
        document, label = test_data[i]
        doc_preview = get_document_preview(document, len_preview, "|")

        # add this prediction to the dataframe of all output predictions
        one_pred = pd.DataFrame({'label': label, 'prediction': predictions[i], 'documents': doc_preview}, index=[0])
        pred_df = pd.concat([pred_df, one_pred], axis=0)

    return pred_df.reset_index(drop=True)


def define_models_with_params(model_dict, data_dict, ngram_size_dict, clip_counts_vals, sent_lex_vals):

    """
    1. Take in multiple dictionaries and lists of the different values that can occur with different model parameters
    2. Iterate through these dictionaries and lists
    3. Create an instance of the model class for each of these different parameter combinations

    Params:
        model_dict: dictionary {str: model} - a map from a model name to a defined scikit-learn model
        data_dict: dictionary {tuple, list} - maps the cross validation fold & document label to the documents that fall under this fold & label pair
        ngram_size_dict: dictionary {str: int} - map the name of an ngram size to an integer of the number of tokens contained in that ngram
        clip_counts_vals: list - this list contains a list of the boolean values we want to test for whether to clip the word counts or not
        sent_lex_vals: list - this list contains a list of the boolean values we want to test for whether to add a sentiment lexicon feature to the extracted features or not

    Returns:
        list of tuples - the tuples contain a dictionary of the models parameters, the data to use, and an instance of the model class with these parameters
                       - [(parameter dictionary, data to use, instance of model class), ....]
    """

    # iterate through all the combinations of parameters for these models
    models_to_compare = []
    for model_to_use_name, model_to_use in model_dict.items():
        for data_type_name, data in data_dict.items():
            for ngram_size_name, ngram_size in ngram_size_dict.items():
                for clip_count_bool in clip_counts_vals:
                    for sent_lex_bool in sent_lex_vals:

                        # define a model name using all the parameters
                        model_full_name = '{} {} {} (ClipCounts={}, SentLexFeature={})'.format(data_type_name, ngram_size_name, model_to_use_name, clip_count_bool, sent_lex_bool)

                        # create a dictionary mapping these models parameter names to their values
                        params_dict = {"full_name": model_full_name, "model_name": model_to_use_name, "data_type": data_type_name, "ngram": ngram_size_name, "clip_counts": clip_count_bool, "sent_lex_feat": sent_lex_bool}

                        # define the model for these parameters
                        model_instance = Model(model=model_to_use, ngram_size=ngram_size, clip_counts=clip_count_bool, add_sent_lex_feat=sent_lex_bool)

                        # create a list of tuples - [(parameter dictionary, data to use, instance of model class), ....]
                        models_to_compare.append((params_dict, data, model_instance))

    print("============================================================================================================")
    print("| There are a total of {} models that have been defined through different combinations of these parameters.|".format(len(models_to_compare)))
    print("============================================================================================================")

    return models_to_compare


def plot_fold_eval_scores(fold_eval_df):

    """
    Create a plot of the evaluation scores and the time for each cross validation fold during the model training and evaluation process for a specific model

    Params:
        fold_eval_df: dataframe - a dataframe containing a row for each fold tested and then a column for the time taken and the accuracy score achived

    Returns:
        Nonde
    """

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, hspace=0)
    axes1, axes2 = gs.subplots(sharex=True)

    # set the axes names
    axes1.set_ylabel('Fold Time (s)')
    axes2.set_ylabel('Fold Accuracy')
    axes2.set_xlabel('Fold No.')

    # plot the evaluation scores
    axes1.plot(fold_eval_df['fold_no'], fold_eval_df['fold_time'])
    axes2.plot(fold_eval_df['fold_no'], fold_eval_df['fold_accuracy'])
    #axes2.set_ylim([0, 1])
    plt.show()


def evaluate_model(model, model_params, train_test_tuples, fold_verbose=False, plot_folds=False):

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

    # extract the parameters
    full_name = model_params["full_name"]
    data_type = model_params["data_type"]
    ngram = model_params["ngram"]
    model_name = model_params["model_name"]
    clip_counts = model_params["clip_counts"]
    sent_lex_feat = model_params["sent_lex_feat"]

    # set up the dataframe we output with the evaluation summary statistics
    fold_eval_df = pd.DataFrame(columns=['fold_no', 'fold_time', 'fold_accuracy'], index=range(len(train_test_tuples)))

    # iterate through the cross validation train-test sets
    fold_accuracies, fold_durations = [], []
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

        # add the results from this fold to the fold evaluation dataframe
        fold_eval_df.loc[i, 'fold_no'] = i + 1
        fold_eval_df.loc[i, 'fold_time'] = iteration_duration
        fold_eval_df.loc[i, 'fold_accuracy'] = accuracy

        # output the model accuracy if requested
        if fold_verbose:
            print(" - {} seconds --> Accuracy score = {}".format(round(iteration_duration), accuracy))

    # plot the fold evaluation results if requested
    if plot_folds:
        plot_fold_eval_scores(fold_eval_df)

    # create the evalutation summary statistics for this model
    n = float(len(fold_accuracies))
    avg = sum(fold_accuracies) / n
    variance = sum([(x-avg)**2 for x in fold_accuracies]) / n
    eval_duration = time.time() - eval_start

    # create a datframe with one row summarising the model evaluation
    eval_values = {'Full Name': full_name,
                   'Data Type': data_type,
                   'Ngram': ngram,
                   'Model': model_name,
                   'Clip Counts': clip_counts,
                   'Sentiment Lexicon Feature': sent_lex_feat,
                   'Avg Accuracy': avg,
                   'Accuracy Std Dev': variance**0.5,
                   'Min Accuracy': min(fold_accuracies),
                   'Max Accuracy': max(fold_accuracies),
                   'Total Time (s)': round(eval_duration, 2),
                   'All Fold Averages': str(fold_accuracies)
                  }

    return pd.DataFrame(eval_values, index=[0])


def read_in_all_model_evaluation_results(folder_with_model_results):

    """
    Read in all the model evaluation files in the given folder and add them to a pandas dataframe

    Params:
        folder_with_model_results: string - the filepath to the folder that contains the files of the model evaluation results

    Returns:
        dataframe - this dataframe contains a row for each trained model and columns detailing this models parameters and its evaluation results
    """

    # define the order of the columns
    all_models_results = pd.DataFrame(columns=["Full Name", "Data Type", "Model", "Ngram", "Clip Counts", "Sentiment Lexicon Feature"])

    # iterate through each file in the specified folder
    for file in os.listdir(folder_with_model_results):
        # add the eval results for this model to a dataframe of all results
        model_results = pd.read_csv(os.path.join(folder_with_model_results, file))
        all_models_results = pd.concat([all_models_results, model_results], axis=0).reset_index(drop=True)

    # make the model column into a categorical so I can order it better
    order = CategoricalDtype(['Multinomial Naive Bayes', 'Random Forest Classifier', 'Logistic Regression', 'Support Vector Machine', 'Stochastic Gradient Descent Classifier', 'Decision Tree'], ordered=True)
    all_models_results['Model'] = all_models_results['Model'].astype(order)

    sorted_df = all_models_results.sort_values(by=["Data Type", "Model", "Ngram", "Clip Counts", "Sentiment Lexicon Feature"]).reset_index(drop=True)

    print("A total of", len(sorted_df), "models have been run.")
    print("The total time it took to evaluate these models was", round(sum(sorted_df["Total Time (s)"])/3600), "hours.")

    return sorted_df


def compare_models_plot(all_models_df, param_colname, base_param_value, min_max_boundaries=False):

    """
    Plot the accuracy scores of different trained models comparing a base parameter value in a column to all the other values in that column

    Params:
        all_models_df: dataframe - a dataframe containing all the trained models results and the details of their parameters
        param_colname: string - the name of the parameter that we want to compare the baseline with a changed value
        base_param_value: string/int/boolean - this is the baseline parameter value in the specified column
        compare_param_value: string/int/boolean - this is the changed parameter value in the specified column
        min_max_boundaries: boolean - this is whether you want to plot the minimum and maximum averages of the specified model across all the folds

    Returns:
        None
    """

    # subset the input dataframe to the rows with and without the counts clipped
    base_param_df = all_models_df[all_models_df[param_colname] == base_param_value]
    compare_param_df = all_models_df[all_models_df[param_colname] != base_param_value]

    # define the figureset
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(nrows=len(compare_param_df), hspace=0)
    axis = gs.subplots(sharex=True)

    # set the title and x/y-axis labels
    axis[0].set_title("Comparing the {} {} to the other {}s".format(base_param_value, param_colname, param_colname), size=20, pad=10)
    plt.xlabel('Accuracy', fontsize=16)

    # set the x-axis limits based on the minimum and maximum averages
    base_min_acc, base_max_acc = min(base_param_df["Avg Accuracy"]), max(base_param_df["Avg Accuracy"])
    comp_min_acc, comp_max_acc = min(compare_param_df["Avg Accuracy"]), max(compare_param_df["Avg Accuracy"])
    global_min_acc = comp_min_acc - 0.02 if comp_min_acc < base_min_acc else base_min_acc  
    global_max_acc = comp_max_acc + 0.02 if comp_max_acc > base_max_acc else base_max_acc
    plt.xlim(global_min_acc - 0.01, global_max_acc + 0.01)

    # iterate through the subplots and define their plots
    for ax, (index, comparison_row) in zip(axis, compare_param_df.iterrows()):

        # set the axes row names
        ax.set_ylabel(comparison_row[param_colname], size=16, rotation='horizontal', ha='right', labelpad=10)
        ax.set_yticks([])

        # subset the row from the base dataframe that has the same configuration as our comparison row except for the specified parameter
        cols = ["Data Type", "Model", "Ngram", "Clip Counts", "Sentiment Lexicon Feature"]
        cols.remove(param_colname)
        base_row = base_param_df
        for col in cols:
            base_row = base_row[base_row[col] == comparison_row[col]]

        # get the accuracy values for the models for the baseline models and the difference between them
        base_acc = float(base_row.iloc[0]["Avg Accuracy"])
        compare_acc = float(comparison_row["Avg Accuracy"])
        diff = compare_acc - base_acc

        # plot the accuracies and the difference between them
        ax.scatter(base_acc, comparison_row[param_colname], color="blue")
        ax.scatter(compare_acc, comparison_row[param_colname], color="orange")
        ax.grid(axis="x")

        if min_max_boundaries:
            # add the min and max accuracy indicators
            plt.annotate(text='|', xy=(float(base_row["Min Accuracy"]), model_num), color="blue")
            plt.annotate(text='|', xy=(float(base_row["Max Accuracy"]), model_num), color="blue")
            plt.annotate(text='|', xy=(float(compare_row.iloc[0]["Min Accuracy"]), model_num), color="orange")
            plt.annotate(text='|', xy=(float(compare_row.iloc[0]["Max Accuracy"]), model_num), color="orange")


        # add the arrows showing the change to the plot
        ax.annotate(text='', xy=(base_acc, comparison_row[param_colname]), xytext=(compare_acc, comparison_row[param_colname]), arrowprops=dict(arrowstyle='<-'))

        # add the % increase/decrease of the models to the plot
        if diff > 0:
            ax.annotate(text=str(round(diff*100, 4))+"%", xy=(compare_acc + 0.003, comparison_row[param_colname]), ha="left")
        elif diff < 0:
            ax.annotate(text=str(round(diff*100, 4))+"%", xy=(compare_acc - 0.003, comparison_row[param_colname]), ha="right")


def compare_param_plot(all_models_df, param_colname, base_param_value, compare_param_value, min_max_boundaries=False):

    """
    Plot the accuracy scores of different trained models using a base parameter value VS a changed parameter value in a specified column

    Params:
        all_models_df: dataframe - a dataframe containing all the trained models results and the details of their parameters
        param_colname: string - the name of the parameter that we want to compare the baseline with a changed value
        base_param_value: string/int/boolean - this is the baseline parameter value in the specified column
        compare_param_value: string/int/boolean - this is the changed parameter value in the specified column
        min_max_boundaries: boolean - this is whether you want to plot the minimum and maximum averages of the specified model across all the folds

    Returns:
        None
    """

    # subset the input dataframe to the rows with and without the counts clipped
    base_param_df = all_models_df[all_models_df[param_colname] == base_param_value]
    compare_param_df = all_models_df[all_models_df[param_colname] == compare_param_value]

    # reverse the rows of this dataframe - so that the baseline comes at the top
    base_param_df = base_param_df.reindex(index=base_param_df.index[::-1])
    compare_param_df = compare_param_df.reindex(index=compare_param_df.index[::-1])

    # create the figure to plot on
    fig = plt.figure(figsize=(12, 12))
    plt.grid(axis='x')
    if type(base_param_value) == bool and type(compare_param_value) == bool:
        plt.title("Comparing model accuracy with and without '{}'".format(param_colname), fontsize=20)
    else:
        plt.title("{} - Comparing the accuracy of {} VS {}".format(param_colname, base_param_value, compare_param_value), fontsize=20)
    plt.xlabel('Accuracy', fontsize=16)

    # set the x-axis limits based on the minimum and maximum averages
    base_min_acc, base_max_acc = min(base_param_df["Avg Accuracy"]), max(base_param_df["Avg Accuracy"])
    comp_min_acc, comp_max_acc = min(compare_param_df["Avg Accuracy"]), max(compare_param_df["Avg Accuracy"])
    global_min_acc = comp_min_acc - 0.02 if comp_min_acc < base_min_acc else base_min_acc  
    global_max_acc = comp_max_acc + 0.02 if comp_max_acc > base_max_acc else base_max_acc
    plt.xlim(global_min_acc - 0.01, global_max_acc + 0.01)

    # iterate through the models and plot the accuracy with and without the counts clipped
    plotted_models = []
    model_num = 0
    for _, base_row in base_param_df.iterrows():

        # subset the row from the comparison dataframe that has the same configuration as our base row except for the specified parameter
        cols = ["Data Type", "Model", "Ngram", "Clip Counts", "Sentiment Lexicon Feature"]
        cols.remove(param_colname)
        compare_row = compare_param_df
        for col in cols:
            compare_row = compare_row[compare_row[col] == base_row[col]]

        if not compare_row.empty:
            plotted_models.append(compare_row.iloc[0]["Model"])

            # get the differece between the two accuracies
            base_acc = float(base_row["Avg Accuracy"])
            compare_acc = float(compare_row.iloc[0]["Avg Accuracy"])
            diff = compare_acc - base_acc
            
            # plot the accuracies and the difference between them
            plt.scatter(base_acc, model_num, color="blue")
            plt.scatter(compare_acc, model_num, color="orange")

            if min_max_boundaries:
                # add the min and max accuracy indicators
                plt.annotate(text='|', xy=(float(base_row["Min Accuracy"]), model_num), color="blue")
                plt.annotate(text='|', xy=(float(base_row["Max Accuracy"]), model_num), color="blue")
                plt.annotate(text='|', xy=(float(compare_row.iloc[0]["Min Accuracy"]), model_num), color="orange")
                plt.annotate(text='|', xy=(float(compare_row.iloc[0]["Max Accuracy"]), model_num), color="orange")

            # add the arrows showing the change to the plot
            plt.annotate(text='', xy=(base_acc, model_num), xytext=(compare_acc, model_num), arrowprops=dict(arrowstyle='<-'))

            # add the % increase/decrease of the models to the plot
            if diff > 0:
                plt.annotate(text=str(round(diff*100, 4))+"%", xy=(compare_acc + 0.003, model_num), ha="left")
            elif diff < 0:
                plt.annotate(text=str(round(diff*100, 4))+"%", xy=(compare_acc - 0.003, model_num), ha="right")

            model_num += 1

    # Add the y ticks to this plot
    plt.yticks(ticks=range(len(plotted_models)), labels=plotted_models, size=18)
    plt.figure(num=1, figsize=(12, len(plotted_models) * 1.5))

    # Add a legend to the plot
    if type(base_param_value) == bool and type(compare_param_value) == bool:
        legend = ["With "+param_colname, "Without "+param_colname] if base_param_value else ["Without "+param_colname, "With "+param_colname]
    else:
        legend = [base_param_value, compare_param_value]
    plt.legend(legend, loc="lower right")


def add_value_labels(ax):

    """
    Add labels to the end of each bar in a bar chart.

    Params:
        ax: matplotlib.axes.Axes - The matplotlib object containing the axes of the plot to annotate.
    
    Returns:
        None
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_y() + rect.get_height() / 2
        x_value = rect.get_width()

        # Create annotation
        ax.annotate(text='{}%'.format(round(x_value * 100, 2)), xy=(x_value, y_value), xytext=(2, -3.5), textcoords="offset points", ha='left')
