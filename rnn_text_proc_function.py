# Import Packages
##############################################################################
import numpy as np, pandas as pd
import gc, random, tqdm, re, string, itertools, time
from string import punctuation
from __future__ import division
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# User Input
##############################################################################
glove_txt_file = 'D:/glove/glove.840B.300d.txt'
train_file = 'D:/quora/data/train.csv'
test_file = 'D:/quora/data/test.csv'

# Define Functions
##############################################################################
def load_glove(glove_file_path, progress_print = 5000, encoding_type = 'utf8'):
    """load glove (Stanford NLP) file and return dictionary"""
    num_lines = sum(1 for line in open(glove_txt_file, encoding = encoding_type))
    embed_dict = dict()
    line_errors = []
    f = open(glove_file_path, encoding = encoding_type)
    for i, l in enumerate(f):
        l_split = l.split()
        try:
            embed_dict[l_split[0]] = np.asarray(l_split[1:], dtype = 'float32')
        except:
            line_errors.append(1)
        if ((i / progress_print) > 0) and (float(i / progress_print) == float(i // progress_print)):
            print(str(int(i / 1000)) + ' K of ' + str(int(num_lines / 1000)) + ' K lines completed')
        else:
            pass
    f.close()
    print('failed lines in file: ' + str(int(np.sum(line_errors))))
    return embed_dict
    
def clean_tokenize(some_string):
    """split on punct / whitespace, make lower case, join back together"""
    pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
    clean_lower = ' '.join([part.lower() for part in pattern.split(some_string) if part.strip()])
    return clean_lower

def glove_tokenize_proc(csv_file_path, csv_txt_col, glove_file_path, vocab_size, maxlen, y_col = None):
    """ - read csv file with text in 'csv_txt_col' column"""
    """ - process into 300-dimension embeddings based on Stanford glove embeddings"""
    glove_dict = load_glove(glove_file_path)
    df = pd.read_csv(csv_file_path)
    df[csv_txt_col] = df[csv_txt_col].apply(clean_tokenize)
    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(df[csv_txt_col])
    sequences = tokenizer.texts_to_sequences(df[csv_txt_col])
    x_data = pad_sequences(sequences, maxlen = maxlen)
    embed_wt_matrix = np.zeros((vocab_size, 300))
    for x, i in tokenizer.word_index.items():
        if i > (vocab_size - 1):
            break
        else:
            embed_vec = glove_dict.get(x)
            if embed_vec is not None:
                embed_wt_matrix[i] = embed_vec
    if y_col:
        y_data = [y for y in df[y_col]]
        return y_data, x_data, embed_wt_matrix
    else:
        return x_data, embed_wt_matrix

# Execute Functions
##############################################################################
train_y, train_x, embed_wts = glove_tokenize_proc(csv_file_path = train_file,
                                                  csv_txt_col = 'question_text',
                                                  glove_file_path = glove_txt_file,
                                                  vocab_size = 50000,
                                                  maxlen = 50,
                                                  y_col = 'target')