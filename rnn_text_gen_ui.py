# Packages
import tensorflow as tf, xgboost as xgb
print("Using Tensorflow Version: " + tf.__version__)
print("Using XGBoost Version: " + xgb.__version__)
tf.enable_eager_execution()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle as pkl, tkinter as tk
import os, time, re, string, gc, random
from __future__ import division
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers
from keras.optimizers import SGD
from random import randint

# User Input
path_to_file = '.../block_strs.txt'
model_save_loc = '.../tf_models/'
vocab_save_loc = '.../lstm_serve/'
filler_word = 'abcdefgh'

# Open Vocab File
with open(vocab_save_loc + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

# Define Functions
class rnn_spec(tf.keras.Model):
    """CuDNNGRU model class"""
    def __init__(self, dict_len, embed_dim, num_units):
        super(rnn_spec, self).__init__()
        self.num_units = num_units
        self.embedding = tf.keras.layers.Embedding(dict_len, embed_dim)
        self.gru = tf.keras.layers.CuDNNGRU(self.num_units,
                                            return_sequences = True,
                                            recurrent_initializer = 'glorot_uniform',
                                            kernel_initializer='he_normal',
                                            stateful = False)
        self.fc = tf.keras.layers.Dense(dict_len)
    def call(self, x):
        embedding = self.embedding(x)
        output = self.gru(embedding)
        prediction = self.fc(output)
        return prediction

def map_wordlist_2_int(word_list, vocab_list):
    """returns encoded words based on unique list"""
    word_to_index = {u:i for i, u in enumerate(vocab_list)}
    text_to_num = np.array([word_to_index[c] for c in word_list])
    return text_to_num

def text_pred(input_str, vocab, model_obj, num_words_gen = 10, temperature = 1.0,
              rnn_sep = True, print_txt = True):
    """predict next n words in sequence using trained RNN"""
    clean_input_str = input_str.split()
    input_eval = map_wordlist_2_int(word_list = [w.lower() if w.lower() in vocab else filler_word for w in clean_input_str],
                                                 vocab_list = vocab)
    input_eval = tf.expand_dims(input_eval, 0)
    idx2char = {i:u for i, u in enumerate(vocab)}
    text_generated = []
    model_obj.reset_states()
    for i in range(num_words_gen):
        predictions = model_obj(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    text_gen_sep = ' '.join(text_generated)
    if print_txt:
        if rnn_sep:
            concat_txt = input_str + " ~~~ RNN -> ~~~ " + text_gen_sep
        else:
            concat_txt = input_str + " " + text_gen_sep
        print(concat_txt)
    return text_gen_sep

def window_init(window_title = 'Recurrent Neural Net - Trump Text Generation',
                window_dim = '520x600',
                label_txt = 'Trump Neural Net Text Generation',
                bg_col = '#f0f8ff',
                accent_col = '#75a3e7'):
                """open window for interactive rnn text generation"""
                def submit():
                    """submit button in chat window"""
                    entered_text = user_input.get() + '\n'
                    bot_reply = text_pred(input_str = entered_text,
                                          vocab = vocab,
                                          model_obj = model,
                                          num_words_gen = randint(7,22),
                                          temperature = 1.0,
                                          print_txt = False)
                    chat_output = bot_reply + '\n'
                    user_input.delete(0,tk.END)
                    output.insert(tk.INSERT, '\n\n\n\n\nACTUAL TRUMP: \n' + entered_text + '\n', "You")
                    output.insert(tk.INSERT,  'TRUMP BOT: \n' + chat_output + '\n\n\n', "Trump") 
                # Window Configuration
                window = tk.Tk()
                window.title(window_title)
                window.configure(background = bg_col)
                window.geometry()
                window.resizable(0, 0)
                # Labels
                tk.Label(window, text = label_txt, width = 62,
                         bg = accent_col, font = 'none 12 bold', anchor = 'w').\
                         grid(row = 1, column = 0, sticky = tk.W)
                tk.Label(window, width = 50, bg = bg_col, height = 1).\
                grid(row = 2, column = 0)
                # Output
                output = tk.Text(window, width = 50, wrap = tk.WORD, background = 'white',font=("Helvetica", 16))
                output.grid(row = 3,column = 0, sticky = tk.W, padx = (10,10))
                output.tag_config("Trump", justify = tk.LEFT)
                output.tag_config("You", justify = tk.LEFT)
                # Padding & Label Formats
                tk.Label(window, width = 50, bg = bg_col, height = 1).\
                grid(row = 4, column = 0)
                tk.Label(window, text = '', width = 62, bg = accent_col,
                         font = 'none 12 bold', anchor ='e').\
                         grid(row = 5, column = 0, sticky =tk.W)
                tk.Label(window, width = 50,bg = bg_col, height = 1).\
                grid(row = 6, column = 0)
                # User Input & Execution
                user_input = tk.Entry(window, width = 50, bg = 'white')
                user_input.grid(row = 7, column = 0, sticky = tk.E,padx = (0,0))
                tk.Label(window, width = 50, bg = bg_col, height = 1).\
                grid(row = 8, column = 0)
                tk.Button(window, text = 'SUBMIT', width = 6, command = submit).\
                grid(row = 9, column = 0, sticky = tk.E)
                window.mainloop()

# Restore Trained Model
model = rnn_spec(dict_len = len(vocab),
                 embed_dim = 250,
                 num_units = 800)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(model_save_loc))
model.build(tf.TensorShape([1, None]))
  
# Open Window
window_init()