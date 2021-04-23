
# general packages
import os
import time
import pandas as pd

# downlaod the data
from selenium import webdriver
import shutil

# read in the data
import tarfile


# ======================
# |  LOAD IN THE DATA  |
# ======================

def get_download_folder():

    '''
    Find the Downloads folder on the users local PC

    Return:
        str - the path to the downloads folder
    '''

    home = os.path.expanduser("~")
    return os.path.join(home, "Downloads")


def load_data(data_directory, chromedriver_location):

    '''
    Load the documents in the data into a dictionary
    If the user has the data in the data directory specified then we use that
    If not, we download it from the web and put it in this folder

    Params:
        data_directory: str - the path to the data directory that the user wants the data file to be in
        chromedriver_location - the location of the chromedriver on the users PC

    Return:
        dictionary - this maps a tuple to a list of lists
                   - {(cross validation fold, document label): [[list of tokens in doc1], [list of tokens in doc2], ...], ...}
    '''

    # get the path to where the file should be
    path_to_tar = os.path.abspath(os.path.join(data_directory, 'review_polarity.tar.gz'))

    # check if the our file doesn't exist in this directory
    if not os.path.isfile(path_to_tar):
        # if the file doesn't exist, we must download it and put it in the directory for next time
        data_url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz'

        # Download the data
        driver = webdriver.Chrome(chromedriver_location)
        driver.get(data_url)
        time.sleep(5)
        driver.close()

        # Move the downloaded file to the apropriate location
        current_file_location = os.path.join(get_download_folder(), "review_polarity.tar.gz")
        shutil.move(current_file_location, path_to_tar)

    # we now know the file exists in our directory so can read the data from it
    with open(path_to_tar, 'rb') as tgz_stream:
        data = {}
        with tarfile.open(mode='r|gz', fileobj=tgz_stream) as tar_archive:

            # iterate through the files in the tar folder
            for tar_member in tar_archive:

                # get the filepath components of the file
                path_components = tar_member.name.split('/')
                filename = path_components[-1]

                # exclude the README file
                if filename.startswith('cv') and filename.endswith('.txt') and '_' in filename:

                    # store this files data in a dictionary
                    label = path_components[-2]
                    fold = int(filename[2])
                    key = (fold, label)
                    if key not in data:
                        data[key] = []

                    # obtain the document, split it into sentences and words, and store it in the dictionary
                    f = tar_archive.extractfile(tar_member)
                    document = [line.decode('utf-8').split() for line in f.readlines()]
                    data[key].append(document)

    return data


# ====================================
# |  SPLIT THE DATA TO TRAIN & TEST  |
# ====================================

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


def get_train_test_splits(data_dict):

    '''
    Use the folds used as keys in the dictionary to split the data into train and test sets with labels for cross-validation

    Params:
        data_dict: dictionary - maps the cross validation fold & document label to the documents that fall under this fold-label

    Returns:
        list: - a list of 'k' training and test cross validation set pairs
              - [(training_data_1, test_data_1), (training_data_2, test_data_2), ...]
    '''

    # get the folds in the dataset
    all_folds = set()
    all_labels = set()
    for (folds, label) in data_dict.keys():
        all_folds.add(folds)
        all_labels.add(label)

    # load the folds
    labelled_documents_in_folds = []
    for fold in all_folds:

        # get a list of all documents and their labels for this fold
        labelled_dataset = []
        for label in all_labels:
            for document in get_documents(data_dict, fold, label):
                labelled_dataset.append((document, label))

        # add these lists to a list for all the folds
        labelled_documents_in_folds.append(labelled_dataset)

    # create training-test splits from these labelled documents
    train_test_tuples = []
    for i in range(len(labelled_documents_in_folds)):
        test_data = labelled_documents_in_folds[i]
        training_data = []
        for j in range(len(labelled_documents_in_folds) - 1):
            fold_num = (i + j + 1) % len(labelled_documents_in_folds)
            assert fold_num != i
            training_data.extend(labelled_documents_in_folds[fold_num])

        train_test_tuples.append((training_data, test_data))

    return train_test_tuples


def count_docs_in_train_test_split(train_test_splits):

    """
    Create a dataframe with the counts of the number of documents in each train and test set for each cross validation fold

    Params:
        train_test_splits: list - a list of 'k' training and test cross validation set pairs
                                - [(training_data_1, test_data_1), (training_data_2, test_data_2), ...]

    Returns:
        dataframe - A dataframe with the counts of the number of documents in each train & test set for each cross validation fold
    """

    num_docs_in_split = pd.DataFrame(columns=["train_set_size", "test_set_size"], index=range(len(train_test_splits)))
    for i, (train_data, test_data) in enumerate(train_test_splits):
        num_docs_in_split.loc[i, "train_set_size"] = len(train_data)
        num_docs_in_split.loc[i, "test_set_size"] = len(test_data)

    return num_docs_in_split
