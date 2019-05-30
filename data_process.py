import sys, pickle, os, random
import numpy as np


tag2label = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6
             }


def read_corpus(data_path):

    input_data = []
    with open(data_path, encoding='utf-8') as f:
        lines = f.readlines()
    sent, tag = [], []
    for line in lines:
        if line != '\n':
            [word,label] = line.strip().split()
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            sent.append(word)
            tag.append(label)
        else:
            input_data.append((sent, tag))
            sent, tag = [], []

    return input_data


def pad_sequences(sequences, pad_mark=0):
    """
    :param sequences:
    :param pad_mark:
    :return:
    """
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))
    seq_list = np.array(seq_list)
    return seq_list, seq_len_list

def batch_iter(data, label, batch_size, num_epochs):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param label: a list of labels
    :param batch_size: the size of mini-batch
    :param num_epochs: number of epochs
    :return: a mini-batch iterator
    """

    assert len(data) == len(label)
    # data = np.array(data)
    # data_size = data.shape[0]
    data_size =len(data)
    epoch_length = data_size // batch_size
    if epoch_length==0:
        epoch_length=1
    for _ in range(num_epochs):
        for batch_num in range(epoch_length):
            start_index = batch_num * batch_size
            end_index = start_index + batch_size

            xdata = data[start_index: end_index]
            ydata = label[start_index: end_index]

            yield xdata, ydata