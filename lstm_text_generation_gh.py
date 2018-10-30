# Packages
import tensorflow as tf
print("Using Tensorflow Version: " + tf.__version__)
tf.enable_eager_execution()
import numpy as np, pandas as pd
import os, time, re, string, gc
from __future__ import division
from operator import itemgetter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# User Input
path_to_file = '.../rally_pres_tweet_agg.txt'
model_save_loc = '.../tf_models/'
n_epochs = 150
early_stop = 6
lr = 0.0001
batch_size = 30
seq_length = 100
rnn_units = 2000
embed_dim = 600

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

def imp_txt_word_list_vocab(txt_file_loc):
    """imports text, returns list of lowercase words without punctuation or numbers"""
    txt = open(txt_file_loc).read()
    clean_txt = txt_lower_alpha_only(txt)
    clean_txt_list = txt_to_word_list(clean_txt)
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
        self.gru = tf.keras.layers.CuDNNLSTM(self.num_units,
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
    epoch_ph = []; trn_loss_ph = []; val_loss_ph = []; break_ph = []
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
        print ('Ep. {} Loss: Train {:.4f} Val {:.4f}'.format(epoch + 1, train_loss, valid_loss))
        # Record Epoch Results
        epoch_ph.append(epoch + 1)
        trn_loss_ph.append(train_loss)
        val_loss_ph.append(valid_loss)
        # Early Stopping
        best_val_loss = min(val_loss_ph)
        if (valid_loss > best_val_loss):
            break_ph.append(1)
        else:
            break_ph = []
        if sum(break_ph) >= early_stop_epochs:
            print("Stopping after " + str(int(epoch + 1)) + " epochs.")
            print("Validation cross entropy hasn't improved in " + str(int(early_stop_epochs)) + " rounds.")
            break
    # Model Saving
    checkpoint_prefix = os.path.join(save_loc, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=rnn)
    checkpoint.save(file_prefix = checkpoint_prefix)
    # Output Training Progress
    output_df = pd.DataFrame({'Epoch': epoch_ph,
                              'Train Loss': trn_loss_ph,
                              'Validation Loss': val_loss_ph})
    end_tm = time.time()
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    print('Execution Time: ' + seconds_to_time(sec_elapsed))
    return output_df

def text_pred(input_str, vocab, model_obj, num_words_gen = 10, temperature = 1.0, print_txt = True):
    clean_input_str = txt_lower_alpha_only(input_str).split()
    input_eval = map_wordlist_2_int(word_list = [w.lower() if w.lower() in vocab else vocab[0] for w in clean_input_str],
                                                 vocab_list = vocab)
    input_eval = tf.expand_dims(input_eval, 0)
    idx2char = {i:u for i, u in enumerate(vocab)}
    text_generated = []
    model_obj.reset_states()
    for i in range(num_words_gen):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    text_gen_sep = ' '.join(text_generated)
    if print_txt:
        concat_txt = input_str + " ~~~ RNN -> ~~~ " + text_gen_sep
        print(concat_txt)
    return input_str, text_gen_sep

# Execute Data Prep Functions
word_list, vocab = imp_txt_word_list_vocab(txt_file_loc = path_to_file)
train_word_list, valid_word_list = split_lst_x_perc(word_list, 0.1)
train_word_num_list = map_wordlist_2_int(word_list = train_word_list, vocab_list = vocab)
valid_word_num_list = map_wordlist_2_int(word_list = valid_word_list, vocab_list = vocab)

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

# Restore Saved Model & Predict Words
model = rnn_spec(dict_len = len(vocab),
                 embed_dim = embed_dim,
                 num_units = rnn_units)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(model_save_loc))
model.build(tf.TensorShape([1, None]))

# Predict Next N Words of Input String
input_str = ["This is the text that is given."]

for s in input_str:
    text_pred(input_str = s,
              vocab = vocab,
              model_obj = model,
              num_words_gen = 10,
              temperature = 1.0,
              print_txt = True)
    print("_________\n\n")