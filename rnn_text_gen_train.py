 
# Packages
import tensorflow as tf, xgboost as xgb
print("Using Tensorflow Version: " + tf.__version__)
print("Using XGBoost Version: " + xgb.__version__)
tf.enable_eager_execution()
import numpy as np, pandas as pd, matplotlib.pyplot as plt, pickle as pkl
import os, time, re, string, gc, random
from __future__ import division
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers
from keras.optimizers import SGD

# User Input
path_to_file = '.../block_strs.txt'
model_save_loc = '.../tf_models/'
vocab_save_loc = '.../lstm_serve/'

n_epochs = 500
early_stop = 25
lr = .00005
batch_size = 150
seq_length = 50
rnn_units = 800
embed_dim = 250
filler_word = 'abcdefgh'

# Define Functions
def seconds_to_time(sec):
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def txt_lower_alpha_only(text):
    """returns string without punctuation or numbers, all lower case"""
    text_nopunct = ''.join([w.lower() for w in re.sub(r'[^\w\s]', ' ', re.sub('['+string.punctuation+']', ' ', text))])
    text_nonums = ''.join([i for i in text_nopunct if not i.isdigit()])
    return text_nonums

def txt_to_word_list(text):
    """returns list of space-separated words"""
    return [w for w in text.split()]

def filter_list_freq(lst, min_freq):
    """filter list/array by removing low frequency elements"""
    arr = np.array(lst)
    items, count = np.unique(np.array(arr), return_counts=True)
    rem_items = items[count < min_freq]
    return [i for i in arr[~np.in1d(np.array(arr), rem_items)]]

def imp_txt_word_list_vocab(txt_file_loc, min_word_freq = 1):
    """imports text, returns list of lowercase words without punctuation or numbers"""
    txt = open(txt_file_loc).read()
    #clean_txt = txt_lower_alpha_only(txt)
    #clean_txt_list = txt_to_word_list(clean_txt)
    clean_txt_list = txt_to_word_list(txt)
    vocab = sorted(set(clean_txt_list))
    return clean_txt_list, vocab
    
def map_wordlist_2_int(word_list, vocab_list):
    """returns encoded words based on unique list"""
    word_to_index = {u:i for i, u in enumerate(vocab_list)}
    text_to_num = np.array([word_to_index[c] for c in word_list])
    return text_to_num

def map_int_2_wordlist(word_list, vocab_list):
    """returns encoded words based on unique list"""
    word_to_index = {u:i for i, u in enumerate(vocab_list)}
    text_to_num = np.array([word_to_index[c] for c in word_list])
    return word_to_index

def sep_x_y_words(words):
    """splits chunk of words into sequence"""
    x_words = words[:-1]
    y_words = words[1:]
    return x_words, y_words

def slice_by_index(lst, indices):
    """slice a list with a list of indices"""
    slicer = itemgetter(*indices)(lst)
    if len(indices) == 1:
        return [slicer]
    return list(slicer)

def batch_order_pos(y, batch_size):
    """positions for batch iteration"""
    idx = [i for i in range(0, len(y))]
    n_batches = len(y) // batch_size
    batch_list = []
    for batch_idx in np.array_split(idx, n_batches):
        batch_list.append([z for z in batch_idx])
    return batch_list

def tf_train_seq_data_proc(num_txt, batch_size, max_seq):
    """creates shuffled tensorflow data object from matrix"""
    x_and_y = tf.data.Dataset.from_tensor_slices(num_txt).apply(tf.contrib.data.batch_and_drop_remainder(max_seq + 1))
    x_y_mapped = x_and_y.map(sep_x_y_words)
    output = x_y_mapped.shuffle(10000).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    return output

def tf_train_seq_data_proc_nobatch(num_txt, batch_size, max_seq):
    """creates shuffled tensorflow data object from matrix"""
    x_and_y = tf.data.Dataset.from_tensor_slices(num_txt).apply(tf.contrib.data.batch_and_drop_remainder(max_seq + 1))
    x_y_mapped = x_and_y.map(sep_x_y_words)
    output = x_y_mapped.shuffle(10000)
    return output

class rnn_spec(tf.keras.Model):
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

def split_lst_x_perc(lst, perc):
    """train test split without reordering or shuffling"""
    tst_idx = [i for i in range(0, int(len(lst) * 0.2))]
    trn_idx = [i for i in range(int(len(lst) * 0.2), len(lst))]
    tst = slice_by_index(lst, tst_idx)
    trn = slice_by_index(lst, trn_idx)
    return trn, tst

def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)

def train_txt_gen_rnn(train_dat, valid_dat, vocab, embed_dim, units, batch_size, seq_len,
                      learn_rt, n_epochs, early_stop_epochs, save_loc):
    """train rnn to predict next words in the sequence"""
    start_tm = time.time()
    # Model Specification
    rnn = rnn_spec(dict_len = len(vocab),
                   embed_dim = embed_dim,
                   num_units = units)
    optimizer = tf.train.AdamOptimizer(learning_rate = learn_rt)
    rnn.build(tf.TensorShape([batch_size, seq_len]))
    valid_x, valid_y = next(iter(valid_dat))
    # Early Stopping Placeholders
    best_val_loss = 999999
    epoch_ph = []; epoch_tm_ph = [start_tm];
    trn_loss_ph = []; val_loss_ph = []; break_ph = []
    # Iterative Training
    for epoch in range(n_epochs):
        # Train
        for (batch, (inp, target)) in enumerate(train_dat):
            with tf.GradientTape() as tape:
                train_predictions = rnn(inp)
                train_loss = loss_function(target, train_predictions)
            grads = tape.gradient(train_loss, rnn.variables)
            optimizer.apply_gradients(zip(grads, rnn.variables))
        # Validation
        for (batch, (inp, target)) in enumerate(valid_dat):
            with tf.GradientTape() as tape:
                valid_predictions = rnn(valid_x)
                valid_loss = loss_function(valid_y, valid_predictions)
        # Record Epoch Results
        epoch_ph.append(epoch + 1)
        trn_loss_ph.append(train_loss)
        val_loss_ph.append(valid_loss)
        epoch_sec_elapsed = str(int((np.float64(time.time()) - np.float64(epoch_tm_ph[-1]))))
        pr_str1 = str('Ep. {} Loss: Train {:.4f} Val {:.4f}'.format(epoch + 1, train_loss, valid_loss))
        print(pr_str1 + '  ' + epoch_sec_elapsed + ' sec.')
        epoch_tm_ph.append(time.time())
        # Early Stopping
        best_val_loss = min(val_loss_ph)
        if (valid_loss > best_val_loss):
            break_ph.append(1)
        else:
            break_ph = []
            # Model Saving
            checkpoint_prefix = os.path.join(save_loc, "ckpt")
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=rnn)
            checkpoint.save(file_prefix = checkpoint_prefix)
        if sum(break_ph) >= early_stop_epochs:
            print("Stopping after " + str(int(epoch + 1)) + " epochs.")
            print("Validation cross entropy hasn't improved in " + str(int(early_stop_epochs)) + " rounds.")
            break
    # Output Training Progress
    output_df = pd.DataFrame({'Epoch': epoch_ph,
                              'Train Loss': trn_loss_ph,
                              'Validation Loss': val_loss_ph})
    end_tm = time.time()
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    print('Execution Time: ' + seconds_to_time(sec_elapsed))
    return output_df

def text_pred(input_str, vocab, model_obj, num_words_gen = 10, temperature = 1.0,
              rnn_sep = True, print_txt = True):
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
    return input_str, text_gen_sep

def text_pred_reccurent(input_str, vocab, model_obj, max_word_gen = 6, min_word_gen = 2, n_repeats = 10,
                        temperature = 1.0, rnn_sep = True, print_txt = True):
    possible_n_words = [i for i in range(min_word_gen, max_word_gen)]
    temp_list = []
    for i in range(n_repeats):
        input_i, txt_gen_i = text_pred(input_str = input_str,
                                       vocab = vocab,
                                       model_obj = model_obj,
                                       num_words_gen = random.choice(possible_n_words),
                                       temperature = temperature,
                                       print_txt = False)
        temp_list.append(txt_gen_i)
    text_gen_sep = ' '.join(temp_list)
    if print_txt:
        if rnn_sep:
            concat_txt = input_str + " ~~~ RNN -> ~~~ " + text_gen_sep
        else:
            concat_txt = input_str + " " + text_gen_sep
        print(concat_txt)
    return input_str, text_gen_sep

# Execute Data Prep Functions
word_list, vocab = imp_txt_word_list_vocab(txt_file_loc = path_to_file, min_word_freq = 10)
vocab.append(filler_word)
train_word_list, valid_word_list = split_lst_x_perc(word_list, 0.15)
train_word_num_list = map_wordlist_2_int(word_list = train_word_list, vocab_list = vocab)
valid_word_num_list = map_wordlist_2_int(word_list = valid_word_list, vocab_list = vocab)

# Save Vocab
with open(vocab_save_loc + 'vocab.pkl', 'wb') as fp:
    pkl.dump(vocab, fp)

# Create Tensorflow Graph & Fit RNN Model
tf.reset_default_graph()
train = tf_train_seq_data_proc(num_txt = train_word_num_list, batch_size = batch_size, max_seq = seq_length)
valid = tf_train_seq_data_proc(num_txt = valid_word_num_list, batch_size = 1, max_seq = seq_length)
train_progress = train_txt_gen_rnn(train_dat = train,
                                   valid_dat = valid,
                                   vocab = vocab,
                                   embed_dim = embed_dim,
                                   units = rnn_units,
                                   batch_size = batch_size,
                                   seq_len = seq_length,
                                   learn_rt = lr,
                                   n_epochs = n_epochs,
                                   early_stop_epochs = early_stop,
                                   save_loc = model_save_loc)
