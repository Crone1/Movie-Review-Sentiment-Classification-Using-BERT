

# =====================
# |  IMPORT PACKAGES  |
# =====================

# general packages
import pandas as pd
from tqdm.auto import tqdm
import os

# package for getting doc length distribution
from collections import defaultdict

# packages for document slicer
from torch.utils.data import Dataset, DataLoader, RandomSampler

# packages for slicing the data
import pytorch_lightning as pl
from torchnlp.encoders import LabelEncoder

# packages for the classifier
import torch
from transformers import AutoModel
import torch.nn as nn
import logging as log
from torchnlp.utils import lengths_to_mask
from collections import OrderedDict
from torch import optim

# packages for training the classifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

# packages for plotting
import matplotlib.pyplot as plt


# =================================
# |  TEST TOKENISER WITH EXAMPLE  |
# =================================

def print_token_summary_of_pretokenised_sentences(tokeniser, pretokenised_sentences_list):

    # tokenise the text
    tokenised_text = tokeniser(pretokenised_sentences_list, is_split_into_words=True)

    for i, token_ids in enumerate(tokenised_text['input_ids']):

        # create a list of the tokens
        tokens_list = tokeniser.convert_ids_to_tokens(token_ids)
        
        # create a dataframe for this sentences values
        output_df = pd.DataFrame()
        output_df["sentence_num"] = [i] * len(tokens_list)
        output_df["input_ids"] = token_ids
        output_df["tokens"] = tokens_list
        output_df["word_ids"] = tokenised_text.word_ids(batch_index = i)

        # output the dataframe
        print(output_df.reset_index(drop=True).to_markdown())
        print("\n")


def print_encoding_of_pretokenised_sentences(tokeniser, pretokenised_sentences_list):

    for i, sentence in enumerate(pretokenised_sentences_list):
        
        output_df = pd.DataFrame(columns=["token", "encoding"])
        for j, token in enumerate(sentence):
            # https://stackoverflow.com/questions/62317723/tokens-to-words-mapping-in-the-tokenizer-decode-step-huggingface
            output_df.loc[j, "token"] = token
            output_df.loc[j, "encoding"] = tokeniser.encode(token, add_special_tokens=False)
            
        # output the dataframe
        print(output_df.to_markdown())
        print("\n")


# =======================
# |  GET THE DOCUMENTS  |
# =======================

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


# ==========================================
# |  ANALYSE DOCUMENT LENGTH DISTRIBUTION  |
# ==========================================

def flatted_each_document_to_a_list_of_tokens(data_dict, label, fold):

    doc_list = []
    for document in get_documents(data_dict, label=label, fold=fold):
        all_tokens_in_doc = [token for sentence in document for token in sentence]
        doc_list.append(all_tokens_in_doc)

    return doc_list


def get_distribution_of_document_lengths(data_dict, tokeniser, bin_width):

    # create a dictionary to contain the distribution of document lengths
    distribution = defaultdict(lambda: 0)

    # iterate through all reviews
    for fold in range(10):
        for label in ['pos', 'neg']:

            # flatten each document into a list of tokens
            batch = flatted_each_document_to_a_list_of_tokens(data_dict, label, fold)

            # pre-tokenise these documents input
            tokenised_batch = tokeniser(batch, is_split_into_words=True)

            # find the largest bin length that contains a document
            max_length_bin = 0
            for token_ids in tokenised_batch['input_ids']:

                # get the bin that the num tokens in this document falls into
                length = len(token_ids)
                length_bin = length // bin_width

                # increment the bin length corresponding to this document
                distribution[(label,   length_bin)] += 1
                distribution[('total', length_bin)] += 1

                # if this bin length hasn't been seen before, update the maximum seen value
                if length_bin > max_length_bin:
                    max_length_bin = length_bin

    return distribution, max_length_bin


def print_doc_breakdown_of_bins(distribution, bin_width, max_length_bin):

    bin_df = pd.DataFrame(columns=["Bin_length", "pos", "neg", "total"])
    for length_bin in range(0, max_length_bin+1):

        # create a dictionary to house this bin length row
        row_dict = {"Bin_length": '{} -> {}'.format(bin_width*length_bin, bin_width*(1+length_bin)-1)}

        # count the number of each document class in this bin
        for label in ['pos', 'neg', 'total']:
            row_dict[label] = distribution[(label, length_bin)]

        # turn this dictionary to a dataframe
        df_row = pd.DataFrame(row_dict, index=[0])

        # add this row to the other bin lengths
        bin_df = pd.concat([bin_df, df_row], axis=0).reset_index(drop=True)

    print(bin_df.to_markdown())


# ======================================
# |  CREATE THE DOCUMENT SLICER CLASS  |
# ======================================

# basic usage of pytorch and lightning from:
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html and https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py

class SlicedDocuments(Dataset):
    
    """
    * Dataset objects behave like a list of dictionaries, one dictionary for each data training or test item, with user-defined keys.
    * DataLoader objects shuffle data provided by a Dataset object and create batches of data.
    * LightningDataModule objects create DataLoader objects for training, validation and test data.
    """

    def __init__(self, raw_data, tokeniser=None, fraction_for_first_sequence=0.5, max_sequence_length=256, second_part_as_sequence_B=False, preproc_batch_size=8):
        
        '''
        Extracts slices from labelled documents
        Args:
            raw_data: list of (document, label) pairs tokeniser:
                      transformer tokeniser to obtain subword units;
                      this tokeniser will be used to decide how many words to include in each slice;
                      it is recommended to use the same tokeniser that will be used to tokenise the data
 
            fraction_for_first_sequence: 1 = take slice from the start of the document,
                                         0 = take slice from the end of the document, > 0 and < 1 = take two slices,
                                         one of this relative size from the start and then fill to max_sequence_length from the end of the document

            max_seq_len: produce sequences up to this length allow_partial_tokens: whether to slice between subword units of tokens;
                         if set to True the produced sequences will always have max_seq_len items unless the document is very short
        '''
        
        assert max_sequence_length >= 5
        
        self.max_sequence_length = max_sequence_length
        self.second_part_as_sequence_B = second_part_as_sequence_B
        self.tokeniser = tokeniser
        # initialise the sequence lengths
        self.init_sequence_lengths(fraction_for_first_sequence)
        # initialise the slices
        self.init_slices(raw_data, preproc_batch_size)

    def init_sequence_lengths(self, fraction_for_first_sequence):
        
        available_for_two_sequences = self.max_sequence_length - 3
        self.first_sequence_length = max(0, int(fraction_for_first_sequence * available_for_two_sequences))
        self.last_sequence_length = max(0, available_for_two_sequences - self.first_sequence_length)
        
        assert self.first_sequence_length + self.last_sequence_length <= available_for_two_sequences
        
        if self.first_sequence_length == 0:
            self.last_sequence_length += 1
            
        elif self.last_sequence_length == 0:
            self.first_sequence_length += 1
            
        if self.first_sequence_length == 0 or self.last_sequence_length == 0:
            # do not use a second [SEP] marker when there is always only one sequence
            self.second_part_as_sequence_B = False

    def init_slices(self, raw_data, preproc_batch_size):
        self.slices = []
        next_batch = []
        next_labels = []
        for document, label in raw_data:
            all_tokens_in_doc = [token for sentence in document for token in sentence]
            next_batch.append(all_tokens_in_doc)    
            next_labels.append(label)

            if len(next_batch) >= preproc_batch_size:
                self.add_batch(next_batch, next_labels)
                next_batch = []
                next_labels = []

        if next_batch:
            self.add_batch(next_batch, next_labels)

    def add_batch(self, batch, labels):

        # determine, for each document in the batch, how many tokens to include from the start of the document
        lengths_1 = self.get_lengths(batch) if self.first_sequence_length else len(batch) * [0]

        # determine, for each document in the batch, how many tokens to include from the end of the document
        lengths_2 = self.get_lengths(batch, part=2, lengths_1=lengths_1) if self.last_sequence_length else len(batch) * [0]

        # TODO: In an earlier version, we did not check the following condition, creating a second sequence for short documents even though only one sequence is requested.
        #       First results indicate that this bug actually improves performance.
        #       Future work should investigate this and, if this effect is confirmed, propose a clean solution to exploit this effect.

        if self.first_sequence_length:
            # sometimes there is space for more tokens from the start even though no more from the end fit
            lengths_1 = self.expand_lengths(batch, lengths_1, lengths_2)

        # TODO: For cases with length_1 + length_2 > len(tokens), should we adjust the parts to not overlap?

        # prepare texts
        for batch_idx, tokens in enumerate(batch):
            parts = []
            length_1 = lengths_1[batch_idx]
            length_2 = lengths_2[batch_idx]
            part_1 = tokens[:length_1]
            part_2 = tokens[-length_2:] if length_2 > 0 else []

            if self.second_part_as_sequence_B:
                parts.append(part_1)
                parts.append(part_2)
            else:
                parts.append(part_1 + part_2)

            assert len(parts) > 0    
            self.slices.append((parts, labels[batch_idx]))

    def get_lengths(self, batch, part=1, lengths_1=None, lengths_2=None):
        
        if part == 3:
            # clone
            lower_limits = lengths_1[:]
        else:
            lower_limits = len(batch) * [0]
            
        upper_limits = []
        for tokens in batch:
            upper_limits.append(min(len(tokens), self.max_sequence_length))
            
        # we want each upper limit to be a sequence length that is too big but sometimes the full document (or max_length words) can fit --> test for this special case
        for batch_idx, fit in enumerate(self.get_fit(batch, upper_limits, part, lengths_1, lengths_2)):
            if fit:
                # update lower limit to match upper limit to mark this document as not needing any further length search
                lower_limits[batch_idx] = upper_limits[batch_idx]

        while True:
            # prepare next lengths to test and check whether search is finished
            new_limits = []
            all_done = True
            for batch_id in range(len(batch)):
                if lower_limits[batch_id] + 1 >= upper_limits[batch_id]:
                    new_limits.append(lower_limits[batch_id])
                else:
                    all_done = False
                    new_limits.append((lower_limits[batch_id] + upper_limits[batch_id])//2)

            if all_done:
                return lower_limits

            # adjust lower and upper limits
            for batch_idx, fit in enumerate(self.get_fit(batch, new_limits, part, lengths_1, lengths_2)):
                if fit:
                    lower_limits[batch_idx] = new_limits[batch_idx]
                else:
                    upper_limits[batch_idx] = new_limits[batch_idx]

    def get_fit(self, batch, limits, part, lengths_1, lengths_2=None):

        sliced_batch_A = []
        sliced_batch_B = []
        for batch_idx, tokens in enumerate(batch):

            if part == 1:
                length_1 = limits[batch_idx]
                length_2 = 0
            elif part == 2:
                length_1 = lengths_1[batch_idx]
                length_2 = limits[batch_idx]
            else:
                length_1 = limits[batch_idx]
                length_2 = lengths_2[batch_idx]

            part_1 = tokens[:length_1]
            part_2 = tokens[-length_2:] if length_2 > 0 else []

            if self.second_part_as_sequence_B:
                sliced_batch_A.append(part_1)
                sliced_batch_B.append(part_2)
            else:
                sliced_batch_A.append(part_1 + part_2)

        if self.second_part_as_sequence_B:
            tokenised = self.tokeniser(sliced_batch_A, sliced_batch_B, is_split_into_words = True)
        else:
            tokenised = self.tokeniser(sliced_batch_A, is_split_into_words = True)

        # check lengths in subword pieced
        retval = []
        for batch_idx, subword_ids in enumerate(tokenised['input_ids']):
            if part == 1:
                # account for [CLS] and [SEP]
                length = len(subword_ids) - 2
                if self.second_part_as_sequence_B:
                    # account for second [SEP] token
                    length -= 1
                fit = length <= self.first_sequence_length
            else:
                fit = len(subword_ids) <= self.max_sequence_length
            retval.append(fit)

        return retval
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            assert isinstance(idx, int)
        parts, label = self.slices[idx]
        retval = {'parts': parts, 'label': label}
        return retval
    
    def expand_lengths(self, batch, lengths_1, lengths_2):
        '''
        pushes lengths_1 as far out as possible
        '''
        return self.get_lengths(batch, part=3, lengths_1=lengths_1, lengths_2=lengths_2)


# ===============================
# |  TEST SLICE DOCUMENT CLASS  |
# ===============================

def test_document_slicer_on_train_test_split(training_data, tokeniser, start_end_fraction, max_seq_length):

    # slice the documents using the defined class
    # each sliced document is a dictionary with two keys - the parts of the document and the label of the document
    # the parts contains a list of either 1 or 2 parts -  each sub-part contains the word tokens in that part of the doc
    # {"parts": [[token1, token2, token3, ... ], [token_1, token_2, token_3, ... ]],
    #  "label": pos/neg
    # }
    sliced_docs = SlicedDocuments(training_data, tokeniser, fraction_for_first_sequence=start_end_fraction, max_sequence_length=max_seq_length, second_part_as_sequence_B=True)
    print("There are {} training documents in this cross validation fold".format(len(sliced_docs)))

    # create a dictionary to store the length counts
    length_to_count_map = defaultdict(lambda: 0)

    # create a dataframe to store the summary of each sliced doc
    sliced_doc_summary_df = pd.DataFrame(columns=['doc_idx', 'seq_len', 'num_tokens_from_start', 'num_tokens_from_end', 'total_tokens'])

    # iterate throgh the sliced documents
    for doc_idx, sliced_doc in enumerate(sliced_docs):

        # get the parts of the sliced document - ignore the label
        parts = sliced_doc['parts']

        # tokenise the parts of the sliced document
        # this returns a dictionary with two keys - the token input_ids and the attention mask of the tokens
        # these keys map to lists of lists where the internal list corresponds to each token across the two parts
        # {"input_ids": [[id1, id2, id3, ... ]],
        #  "attention_mask": [[mask_v1, mask_v2, mask_v3, ... ]]
        # }
        if len(parts) == 2:
            tokenised = tokeniser([parts[0]], [parts[1]], is_split_into_words=True)
        else:
            tokenised = tokeniser([parts[0]], is_split_into_words=True)

        # count the number of tokens in this document sequence
        seq_length = len(tokenised['input_ids'][0])

        # increment the count of the number of documents with this specified length
        length_to_count_map[seq_length] += 1

        # get the length of the two token sequence sub-parts in these document parts
        num_token_in_start_seq = len(parts[0])
        num_token_in_end_seq = len(parts[1]) if len(parts) == 2 else 0

        # add summary details on the first 10 documents to a dataframe
        if doc_idx < 10:
            one_row_dict = {'doc_idx': doc_idx,
                            'seq_len': seq_length,
                            'num_tokens_from_start': num_token_in_start_seq,
                            'num_tokens_from_end': num_token_in_end_seq,
                            'total_tokens': num_token_in_start_seq + num_token_in_end_seq}
            one_row_df = pd.DataFrame(one_row_dict, index=[0])
            sliced_doc_summary_df = pd.concat([sliced_doc_summary_df, one_row_df], axis=0).reset_index(drop=True)

    return sliced_doc_summary_df, length_to_count_map


# ==================================
# |  CREATE THE DATA SLICER CLASS  |
# ================================== 

# https://github.com/ricardorei/lightning-text-classification/blob/master/classifier.py

class SlicedDataModule(pl.LightningDataModule):
    
    def __init__(self, classifier, global_batch_size, data_split=None, **kwargs):
        super().__init__()
        self.hparams = classifier.hparams
        self.classifier = classifier
        self.global_batch_size = global_batch_size
        if data_split is None:
            # this happens when loading a checkpoint
            data_split = (None, None, None)

        elif len(data_split) == 2:
            # add empty validation set
            tr_data, val_data = self.split(data_split[0], 0.9)
            data_split = (tr_data, val_data, data_split[1])

        self.data_split = data_split
        self.kwargs = kwargs
        self.label_encoder = LabelEncoder(['pos', 'neg'], reserved_labels=[])

    def train_dataloader(self) -> DataLoader:
        assert self.hparams.batch_size <= self.global_batch_size
        dataset = SlicedDocuments(raw_data=self.data_split[0], **self.kwargs)
        return DataLoader(dataset=dataset, sampler=RandomSampler(dataset), batch_size=self.hparams.batch_size, collate_fn=self.classifier.prepare_sample, num_workers=self.hparams.loader_workers)
    
    def val_dataloader(self) -> DataLoader:
        assert self.hparams.batch_size <= self.global_batch_size
        if not self.data_split[1]:
            # TODO: check documentation what to return
            return None

        return DataLoader(dataset=SlicedDocuments(raw_data=self.data_split[1], **self.kwargs), batch_size=self.hparams.batch_size, collate_fn=self.classifier.prepare_sample, num_workers=self.hparams.loader_workers)
    
    def test_dataloader(self) -> DataLoader:
        assert self.hparams.batch_size <= self.global_batch_size
        return DataLoader(dataset=SlicedDocuments(raw_data=self.data_split[2], **self.kwargs),
                          batch_size=self.hparams.batch_size, collate_fn=self.classifier.prepare_sample,
                          num_workers=self.hparams.loader_workers)

    def split(self, data, ratio):
        # get label distribution:
        distribution = defaultdict(lambda: 0)
        for _, label in data:
            distribution[label] += 1

        # get target frequencies of labels in first set
        still_needed = defaultdict(lambda: 0)
        for label in distribution:
            still_needed[label] = int(ratio*distribution[label])

        # split data accordingly
        dataset_1 = []
        dataset_2 = []
        for item in data:
            label = item[1]
            if still_needed[label] > 0:
                dataset_1.append(item)
                still_needed[label] -= 1
            else:
                dataset_2.append(item)

        return dataset_1, dataset_2


# =================================
# |  CREATE THE CLASSIFIER CLASS  |
# =================================

class Classifier(pl.LightningModule):
    
    def __init__(self, hyper_params=None, **kwargs) -> None:
        super().__init__()
        if type(hyper_params) is dict:
            hyper_params = pl.utilities.AttributeDict(hyper_params)

        self.global_batch_size = hyper_params["batch_size"]
        self.model_name = kwargs["model_name"]
        del kwargs["model_name"]
        self.hparams = hyper_params
        self.batch_size = hyper_params.batch_size
        self.data = SlicedDataModule(self, self.global_batch_size, **kwargs)

        if 'tokeniser' in kwargs:
            # attribute expected by lightning
            self.tokenizer = kwargs['tokeniser']
        else:
            # this happens when loading a checkpoint
            # TODO: this may break ability to use the model
            self.tokenizer = None

        self.__build_model()
        self.__build_loss()
        if hyper_params.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False

        self.nr_frozen_epochs = hyper_params.nr_frozen_epochs
        self.record_predictions = False
            
    def __build_model(self) -> None:
        '''
        Init BERT model, tokeniser and classification head
        '''

        # Q: Why not use AutoModelForSequenceClassification?
        self.bert = AutoModel.from_pretrained(self.model_name, output_hidden_states=True)
        
        self.classification_head = nn.Sequential(nn.Dropout(0.2),
                                                 nn.Linear(self.bert.config.hidden_size, 1536),
                                                 nn.Tanh(),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(1536, 256),
                                                 nn.Tanh(),
                                                 nn.Dropout(0.1),
                                                 nn.Linear(256, self.data.label_encoder.vocab_size)
                                                )

    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()
        
    def unfreeze_encoder(self) -> None:
        if self._frozen:
            log.info('\n== Encoder model fine-tuning ==')
            for param in self.bert.parameters():
                param.requires_grad = True
            self._frozen = False
            
    def freeze_encoder(self) -> None:
        for param in self.bert.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict:
        if self.training:
            self.eval()

        with torch.no_grad():
            batch_inputs, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(batch_inputs)
            logits = torch.Tensor.cpu(model_out["logits"]).numpy()
            predicted_labels = [self.data.label_encoder.index_to_token[prediction] for prediction in numpy.argmax(logits, axis=1)]
            sample["predicted_label"] = predicted_labels[0]

        return sample
    
    def start_recording_predictions(self):
        self.record_predictions = True
        self.reset_recorded_predictions()
        
    def stop_recording_predictions(self):
        self.record_predictions = False
        
    def reset_recorded_predictions(self):
        self.seq2label = {}
        
    def forward(self, batch_input):
        tokens  = batch_input['input_ids']
        lengths = batch_input['length']
        mask = batch_input['attention_mask']

        # Run BERT model.
        word_embeddings = self.bert(tokens, mask).last_hidden_state
        sentence_embedding = word_embeddings[:,0]  # at position of [CLS]
        logits = self.classification_head(sentence_embedding)

        # Hack to conveniently use the model and trainer to get predictions for a test set:
        if self.record_predictions:
            logits_np = torch.Tensor.cpu(logits).numpy()
            predicted_labels = [self.data.label_encoder.index_to_token[prediction] for prediction in numpy.argmax(logits_np, axis=1)]
            
            for index, input_token_ids in enumerate(tokens):
                key = torch.Tensor.cpu(input_token_ids).numpy().tolist()
                # truncate trailing zeros
                while key and key[-1] == 0:
                    del key[-1]
                self.seq2label[tuple(key)] = predicted_labels[index]

        return {"logits": logits}
    
    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.

        Params:
            predictions: model specific output. Must contain a key 'logits' with a tensor [batch_size x 1] with model predictions
            labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])
    
    def prepare_sample(self, sample: list, prepare_target: bool=True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        assert len(sample) <= self.global_batch_size
        assert self.tokenizer is not None
        with_1_part = 0
        with_2_parts = 0
        batch_part_1 = []
        batch_part_2 = []
        for item in sample:
            parts = item['parts']
            if len(parts) == 2:
                with_2_parts += 1
                batch_part_1.append(parts[0])
                batch_part_2.append(parts[1])

            else:
                with_1_part += 1
                batch_part_1.append(parts[0])

        assert not (with_1_part and with_2_parts)
        kwargs = {'is_split_into_words': True,
                  'return_length':       True,
                  'padding':             'max_length',
                  # https://github.com/huggingface/transformers/issues/8691
                  'return_tensors':      'pt',
                 }

        if with_2_parts:
            encoded_batch = self.tokenizer(batch_part_1, batch_part_2, **kwargs)
        else:
            encoded_batch = self.tokenizer(batch_part_1, **kwargs)

        if not prepare_target:
            return encoded_batch, {}

        # Prepare target:
        batch_labels = []
        for item in sample:
            batch_labels.append(item['label'])

        assert len(batch_labels) <= self.global_batch_size

        try:
            targets = {"labels": self.data.label_encoder.batch_encode(batch_labels)}
            return encoded_batch, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        # Q: What is this about?
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        output = OrderedDict({"loss": loss_val})
        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True)
        # can also return just a scalar instead of a dict (return loss_val)
        return output
   
    def test_or_validation_step(self, test_type, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:

        inputs, targets = batch
        model_out = self.forward(inputs)
        loss_val = self.loss(model_out, targets)
        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)
        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)

        output = {test_type + "_loss": loss_val,
                  test_type + "_acc" :  val_acc,
                  'batch_size'       : len(batch),
                  #'predictions'     : labels_hat,
                 }

        return OrderedDict(output)
    
    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        return self.test_or_validation_step('val', batch, batch_nb, *args, **kwargs)
    
    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        return self.test_or_validation_step('test', batch, batch_nb, *args, **kwargs)
    
    # validation_end() is now validation_epoch_end()
    # https://github.com/PyTorchLightning/pytorch-lightning/blob/efd272a3cac2c412dd4a7aa138feafb2c114326f/CHANGELOG.md
    
    def test_or_validation_epoch_end(self, test_type, outputs: list) -> None:

        val_loss_mean = 0.0
        val_acc_mean = 0.0
        total_size = 0
        for output in outputs:
            val_loss = output[test_type + "_loss"]

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output[test_type + "_acc"]
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            # We weight the batch accuracy by batch size to not give higher weight to the items of a smaller, final bacth.
            batch_size = output['batch_size']
            val_acc_mean += val_acc * batch_size
            total_size += batch_size

        val_loss_mean /= len(outputs)
        val_acc_mean /= total_size
        self.log(test_type+'_loss', val_loss_mean)
        self.log(test_type+'_acc',  val_acc_mean)

    def validation_epoch_end(self, outputs: list) -> None:
        self.test_or_validation_epoch_end('val', outputs)
                                     
    def test_epoch_end(self, outputs: list) -> None:
        self.test_or_validation_epoch_end('test', outputs)
        
    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [{"params": self.classification_head.parameters()},
                      {"params"       : self.bert.parameters(),
                       "lr"           : self.hparams.encoder_learning_rate,
                       #"weight_decay" : 0.01,  # TODO: try this as it is in the BERT paper
                      },
                     ]

        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """
        Pytorch lightning hook
        """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()


# ==========================================
# |  DEFINE A CLASSIFIER FOR EACH CV FOLD  |
# ==========================================

def define_classifier_and_trainer_for_each_cv_fold(train_test_splits, classifier_params, model_params, predefined_variables):

    # Define a BERT classifier for each CV fold using the specified parameters
    print("Defining the classifiers for each CV fold:")
    classifiers_list = []
    for cv_data_split in tqdm(train_test_splits):
        # create the classifier for this cv fold
        classifier = Classifier(hyper_params=classifier_params,
                                model_name=predefined_variables["model_name"],
                                # parameters for SlicedDataModule:
                                data_split=cv_data_split,
                                # parameters for SlicedDocument():
                                tokeniser=predefined_variables["tokeniser"],
                                fraction_for_first_sequence=model_params["start_end_fraction"],
                                max_sequence_length=predefined_variables["max_sequence_length"],
                                second_part_as_sequence_B=False,
                                preproc_batch_size=model_params["preproc_batch_size"],
                               )
        # add this classifier to a list of classifiers
        classifiers_list.append(classifier)

    # Set the model up so that it stops running if there is no improvement in accuracy
    # https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html
    early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=model_params["min_early_stopping_delta"],
                                        patience=model_params["model_patience"], verbose=False, mode='max')

    # Define a training framework for each of these classifiers
    print("Defining the trainers for each CV fold:")
    trainers_list, modeL_callbacks = [], []
    for cv_num, classifier in tqdm(enumerate(classifiers_list)):

        # Set the model up so that the model is saved at checkpoints when its accuracy improves
        save_top_model_callback = ModelCheckpoint(save_top_k=3, monitor='val_acc', mode='max',
                                                  filename='cv{cv_num}-{val_acc:.4f}-{epoch:02d}-{val_loss:.4f}')

        # define the trainer
        trainer = pl.Trainer(callbacks=[early_stop_callback, save_top_model_callback],
                             max_epochs=model_params["max_epochs"],
                             min_epochs=classifier.hparams.nr_frozen_epochs + 2,
                             gpus=classifier.hparams.gpus,
                             accumulate_grad_batches=4,  # compensate for small batch size
                             # limit_train_batches=10,  # use only a subset of the data during development for higher speed
                             check_val_every_n_epoch=1,
                             # https://github.com/PyTorchLightning/pytorch-lightning/issues/6690
                             logger=pl.loggers.TensorBoardLogger(os.path.abspath('lightning_logs')),
                             )
        # add this model callback & trainer to a list of other callbacks & trainers
        modeL_callbacks.append(save_top_model_callback)
        trainers_list.append(trainer)

    return classifiers_list, trainers_list, modeL_callbacks


# =========================
# |  EVALUATE THE MODELS  |
# =========================

def plot_fold_eval_scores(fold_eval_df):

    """
    Create a plot of the evaluation scores and the time for each cross validation fold during the model training and evaluation process for a specific model

    Params:
        fold_eval_df: dataframe - a dataframe containing a row for each fold tested and then a column for the time taken and the accuracy score achived

    Returns:
        None
    """

    fig, (axes1, axes2, axes3) = plt.subplots(figsize=(16, 12), nrows=3, ncols=1, sharex=True)

    # set the axes names
    axes1.set_ylabel('Time (s)')
    axes2.set_ylabel('Val Accuracy')
    axes3.set_ylabel('Test Accuracy & Loss')
    axes3.set_xlabel('Fold No.')

    # plot the evaluation scores
    axes1.plot(fold_eval_df['fold_no'], fold_eval_df['fold_time'])
    axes2.plot(fold_eval_df['fold_no'], fold_eval_df['fold_validation_accuracy'])
    axes3.plot(fold_eval_df['fold_no'], fold_eval_df['fold_test_accuracy'])
    axes3.plot(fold_eval_df['fold_no'], fold_eval_df['fold_test_loss'])
    #plt.subplots_adjust(hspace=0.5)
    plt.show()


# =================================
# |  SAVE THE BEST TRAINED MODEL  |
# =================================

def save_best_model(classifier, trainer, fold_num):

    # Explicitly load the classifiers best checkpoint - batch size was saved in the checkpoint automatically
    best_model = Classifier.load_from_checkpoint(checkpoint_path=trainer.checkpoint_callback.best_model_path)

    # Wrap the model into a new trainer to be able to save a checkpoint
    new_trainer = pl.Trainer(resume_from_checkpoint=trainer.checkpoint_callback.best_model_path,
                             gpus=-1,  # avoid warnings (-1 = automatic selection)
                             logger=pl.loggers.TensorBoardLogger(os.path.abspath('lightning_logs')),
                             )

    # set the model of this trainer to be the trained best model
    new_trainer.model = best_model  # @model.setter in plugins/training_type/training_type_plugin.py

    # -----------------------
    # | save the best model |
    # -----------------------

    # this version contains absolute paths and training parameters
    new_trainer.save_checkpoint("fold-{}-best-model.ckpt".format(fold_num))

    # this version saves only the weights - second parameter is 'save_weights_only'
    new_trainer.save_checkpoint("fold-{}-best-model-weights-only.ckpt".format(fold_num), True)

    # this version saves the bert model in pytorch format and without the classification head
    best_model.bert.save_pretrained('fold-{}-best-bert-encoder.pt'.format(fold_nu))

    # this version saves the full network in pytorch format - can be done since the lightning module inherits from pytorch
    torch.save(best_model.state_dict(), 'fold-{}-best-model.pt'.format(fold_num))


# ===================================
# |  ANALYSE THE MODELS PREDICTION  |
# ===================================

def test_model_and_get_results(model, test_dataset):

    # define a dataframe to house the datas predictions
    prediction_df = pd.DataFrame(columns=['index', 'real_class', 'predicted_class', 'correct'], index=list(range(len(test_dataset))))

    # iterate through the test dataset and populate the dataframe with the predictions
    for i, item in enumerate(test_dataset):
        
        # get the prediction key for this item
        input_token_ids = model.prepare_sample([item])[0]['input_ids']
        key = input_token_ids.tolist()[0]

        # truncate the zeros from this key
        while key and key[-1] == 0:
            del key[-1]
        
        # turn the key to a tuple
        key = tuple(key)

        # get the models prediction for this item
        try:
            prediction = model.seq2label[key]
        except KeyError:
            prediction = 'unknown'

        # populate the dataframe
        prediction_df.loc[i, "real_class"] = item['label']
        prediction_df.loc[i, "predicted_class"] = prediction
        prediction_df.loc[i, "correct"] = "yes" if prediction == item['label'] else "no"

    return prediction_df


