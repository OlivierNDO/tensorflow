# Packages
###############################################################################
import numpy as np, pandas as pd, bs4 as bs, matplotlib.pyplot as plt, seaborn as sns
import nltk, urllib.parse, urllib.request, string, gc, random
from nltk.corpus import stopwords
from urllib.error import URLError
from bs4 import BeautifulSoup as bs
from selenium import webdriver
import time, re, string
from time import sleep
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

# Phantom JS Path
###############################################################################
ph_path = '.../phantomjs.exe'

# Save Info
###############################################################################
save_folder = '.../aggregate/'
save_prefix = 'aggregate_remarks'

# Data Paths
###############################################################################
interview_path = '.../interview_remarks.txt'
rallies_path = '.../rally_remarks.txt'
vlogs_path = '.../vlog_remarks.txt'
press_path = '.../wh_remarks.txt'

# Define Functions
###############################################################################
def read_file(path, return_lines = True):
    """read text file - optional return list of lines"""
    doc = open(path, 'r', encoding="utf8")
    txt = doc.read()
    doc.close()
    if return_lines:
        return txt.split('\n')
    else:
        return txt

def rem_transc_notes(txt):
    """remove transcriber notes, e.g. (<text>) | [<text>]"""
    return re.sub("[\(\[].*?[\)\]]", "", txt)

def rem_custom(txt, remove_list):
    """remove list of substrings"""
    txt_copy = txt
    for rl in remove_list:
        txt_copy = txt_copy.replace(rl, '')
    return txt_copy

def remove_punct_except_apost(txt):
    """remove all punctuation except apostrophes"""
    return re.sub("[^\w\d'\s]+", '', txt)
    
def remove_punct_exc_apost_custom(txt_list, custom_remove):
    """apply rem_transc_notes, rem_custom, remove_punct_except_apost iteratively"""
    txt_list_copy = txt_list
    temp_list = []
    for tl in txt_list_copy:
        new_tl = rem_transc_notes(tl.lower())
        new_tl = rem_custom(new_tl, custom_remove)
        temp_list.append(remove_punct_except_apost(new_tl))
    return temp_list

def str_to_sequences(my_str, seq_len, filler_word = 'fillerword'):
    """convert single string to sequence of n (zero-padded) words"""
    word_list = my_str.split()
    seq_list = []
    for i in range(seq_len, len(word_list) + seq_len, seq_len):
        seq = word_list[i - (seq_len):(i + seq_len)]
        if len(seq) < seq_len:
            seq = word_list[(len(word_list) - seq_len) : len(word_list)]
            if len(seq) < seq_len:
                seq = [filler_word] * (seq_len - len(seq)) + seq
            else:
                pass
        else:
            pass
        seq_list.append(' '.join(seq))
    return '\n'.join(seq_list)
        
def str_to_sentences_mult(my_str_list, seq_len, filler_word = 'fillerword'):
    """apply str_to_sentences iteratively"""
    temp_list = []
    for msl in tqdm(my_str_list):
        temp_list.append(str_to_sequences(my_str = msl,
                                          seq_len = seq_len,
                                          filler_word = filler_word))
    return '\n'.join(temp_list)

# Execute Functions
###############################################################################
# Import and Aggregate Data
interviews = read_file(interview_path)
rallies = read_file(rallies_path)
vlogs = read_file(vlogs_path)
press = read_file(press_path)
agg = interviews + rallies + vlogs + press

# Clean Up Aggregate Data
agg_clean = remove_punct_exc_apost_custom(txt_list = agg, custom_remove = ['applause'])
# Reshape Text into 30-word Sequences
text_sequences = str_to_sentences_mult(my_str_list = agg_clean, seq_len = 30)

# Saving
###############################################################################
# n-Word Sequences Separated by Line Breaks
file = open('D:/individual_trump_stmts/aggregate/sequence_strs_30_words.txt', 'w', encoding="utf-8")
file.write(text_sequences)
file.close()

# All Text Separated by Spaces
all_space_sep = ' '.join(' '.join(text_sequences.split('\n')).split()).replace('fillerword', '')
all_space_sep = ' '.join(all_space_sep.split())
file = open('D:/individual_trump_stmts/aggregate/block_strs.txt', 'w', encoding="utf-8")
file.write(all_space_sep)
file.close()