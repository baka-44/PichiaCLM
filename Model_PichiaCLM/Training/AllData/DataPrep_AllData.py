
#==========================================
#Data Prep for Encoder-Decoder Language Model
#==========================================

import logging
import time
import os

import numpy as np
import matplotlib.pyplot as plt

#import tensorflow_datasets as tfds
import tensorflow as tf

# Import tf_text to load the ops used by the tokenizer saved model
#import tensorflow_text  # pylint: disable=unused-import
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib as plt

from sklearn.model_selection import train_test_split


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model,  Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Dropout, Input, TimeDistributed, Dense, Activation, RepeatVector, Embedding, Concatenate
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam, Adagrad
from keras.losses import sparse_categorical_crossentropy
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings
import random


def tokenize_AA(sequences):
    AA_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
               'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19,
               'Y': 20, 'X': 21, 'Z': 21, 'B': 21, 'U': 21, 'O': 21, '*': 22}

    seq_tokenized = []
    for s in range(len(sequences)):
        seq = sequences[s]
        temp = [24]  # Start token
        for i in range(len(seq)):
            temp.append(AA_dict[seq[i]])
        temp.append(23)

        seq_tokenized.append(temp)

    return seq_tokenized, AA_dict




def AA_Codon_list():
    dic_AA_codon = {'A': ['GCT', 'GCC', 'GCA', 'GCG'],
                    'C': ['TGT', 'TGC'],
                    'D': ['GAT', 'GAC'],
                    'E': ['GAA', 'GAG'],
                    'F': ['TTT', 'TTC'],
                    'G': ['GGT', 'GGA', 'GGC', 'GGG'],
                    'H': ['CAT', 'CAC'],
                    'I': ['ATT', 'ATC', 'ATA'],
                    'K': ['AAA', 'AAG'],
                    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
                    'M': ['ATG'],
                    'N': ['AAT', 'AAC'],
                    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
                    'Q': ['CAA', 'CAG'],
                    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
                    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
                    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
                    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
                    'W': ['TGG'],
                    'Y': ['TAT', 'TAC'],
                    '*': ['TAA', 'TAG', 'TGA']}

    codon_list = []
    AA_list = []
    for key in dic_AA_codon:
        for i in range(len(dic_AA_codon[key])):
            AA_list.append(key)
        for i in dic_AA_codon[key]:
            codon_list.append(i)
    return AA_list, codon_list

def tokenize_Codon(sequences):
    AA_list, codon_list = AA_Codon_list()
    keys = codon_list
    values = range(1, len(codon_list) + 1)
    Codon_dict = dict(zip(keys, values))
    seq_tokenized = []
    for s in range(len(sequences)):
        seq = sequences[s]
        temp = [65]  # Start token
        for i in range(int(len(seq) / 3)):
            temp.append(Codon_dict[seq[3 * i: 3 * (i + 1)]])
        
        temp.append(66)
        seq_tokenized.append(temp)

    return seq_tokenized, Codon_dict

def data_prep(N_organisms, Data_dict): #
    Data_dict_New = {}
    for o in range(N_organisms):
        index = []
        for i in range(len(Data_dict[o]['CDS_Seq'])):
            if Data_dict[o]['CDS_Seq'][i].find('N') == -1:
                index.append(i)

        Data_dict_New[o] = Data_dict[o].iloc[index, :]

    AA_Seq_org = {}
    CDS_Seq_org = {}
    AA_Seq_list = []
    CDS_Seq_list = []

    for o in range(N_organisms):
        AA_Seq_org[o] = Data_dict_New[o]['AA_Seq'].tolist()
        CDS_Seq_org[o] = Data_dict_New[o]['CDS_Seq'].tolist()
        AA_Seq_list = AA_Seq_list + AA_Seq_org[o]
        CDS_Seq_list = CDS_Seq_list + CDS_Seq_org[o]

    Nreps = 25
    AA_list, codon_list = AA_Codon_list()
    AA_list_rep = AA_list
    codon_list_rep = codon_list

    for k in range(Nreps):
        AA_list_rep = AA_list_rep + AA_list
        codon_list_rep = codon_list_rep + codon_list

    AA_Seq_list = AA_list_rep + AA_Seq_list
    CDS_Seq_list = codon_list_rep + CDS_Seq_list

    # temp = list(zip(AA_Seq_list, CDS_Seq_list))
    # random.shuffle(temp)
    # res1, res2 = zip(*temp)
    # # res1 and res2 come out as tuples, and so must be converted to lists.
    # AA_Seq_list_shuffle, CDS_Seq_list_shuffle = list(res1), list(res2)

    AA_seq_tokenized, AA_seq_tokenizer = tokenize_AA(AA_Seq_list)
    Cds_seq_tokenized, Cds_seq_tokenizer = tokenize_Codon(CDS_Seq_list)

    AA_pad_seq = pad_sequences(AA_seq_tokenized, maxlen=10001, dtype='int32', padding="post", truncating="post")
    Cds_pad_seq = pad_sequences(Cds_seq_tokenized, maxlen=1000, dtype='int32', padding="post", truncating="post")

    #AA_tr, AA_ts, Cds_tr, Cds_ts = train_test_split(AA_pad_seq, Cds_pad_seq, test_size=0.10, random_state=42)
    AA_tr = AA_pad_seq[len(AA_list):, :]
    Cds_tr = Cds_pad_seq[len(codon_list):, :]
    AA_ts = AA_pad_seq[0:len(AA_list), :]
    Cds_ts = Cds_pad_seq[0:len(codon_list), :]

    return AA_tr, Cds_tr, AA_ts, Cds_ts
