# Import Packages
#########################################################################################################################
import pandas as pd, numpy as np, tensorflow as tf
import pyodbc
from sqlalchemy import create_engine, MetaData, Table, select
import os, time
import math
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import datetime
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, log_loss
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import KFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from random import choices

# Data Import
#########################################################################################################################
agg_scaled_data = pd.read_csv('C:/Users/user/Desktop/school/dddm/trf_fatality_scaled_07142018.csv')

# Train, Test, Validation Split
######################################################################################
y = agg_scaled_data.iloc[:,1]
x = agg_scaled_data.iloc[:,2:447]
trn_y, tst_y, trn_x, tst_x = train_test_split(y, x, test_size = .25, random_state = 72018)
tst_y, val_y, tst_x, val_x = train_test_split(tst_y, tst_x, test_size = .5, random_state = 72018)
trn_x = trn_x.astype(np.float32)
tst_x = tst_x.astype(np.float32)
val_x = val_x.astype(np.float32)
trn_y = trn_y.as_matrix()
tst_y = tst_y.as_matrix()
val_y = val_y.as_matrix()

# Neural Net Parameters
######################################################################################
tf.reset_default_graph()
n_inputs = trn_x.shape[1]
learn_rt = 0.02
n_classes = 2
n_hidden1 = 120
n_hidden2 = 120
dropout = 0.3
n_epochs = 100
batch_size = 1000
early_stop_epocs = 3

# Balanced Batch Iterations
######################################################################################
def balanced_batch_pos(train_y, size_batch):
    pos_0 = []
    pos_1 = []
    for x, i in enumerate(train_y):
        if i == 0:
            pos_0.append(x)
        else:
            pos_1.append(x)
    num_batch = (len(pos_0) + len(pos_1)) // size_batch
    batch_list = []
    for i in range(0, num_batch):
        sample0 = choices(pos_0, k = size_batch//2)
        sample1 = choices(pos_1, k = size_batch//2)
        batch_list.append(shuffle(sample0 + sample1))
    return batch_list
    
batch_i = balanced_batch_pos(trn_y, batch_size)

# Neural Net Structure
######################################################################################
x = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'x')
y = tf.placeholder(tf.int64)

# Define Layers
with tf.name_scope('dnn'):
    kernel_init = tf.contrib.layers.variance_scaling_initializer()
    training = tf.placeholder_with_default(False, shape=(), name='training')
    hidden1 = tf.layers.dense(x, n_hidden1, name = 'hidden1', activation = tf.nn.relu, kernel_initializer = kernel_init)
    batch_norm = tf.layers.batch_normalization(hidden1, training = training, momentum = 0.9)
    hidden2 = tf.layers.dense(batch_norm, n_hidden2, name = 'hidden2', activation = tf.nn.relu, kernel_initializer = kernel_init)
    d_out = tf.layers.dropout(hidden2, rate=dropout)
    logits = tf.layers.dense(d_out, n_classes, activation=tf.nn.sigmoid)

# Define Loss Function : tf.nn.sparse_softmax_cross_entropy_with_logits
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = 'loss')

# Define Gradient Descent Optimizer
with tf.name_scope('train'): 
    optimizer = tf.train.GradientDescentOptimizer(learn_rt)
    training_op = optimizer.minimize(loss)
    
# Define Performance Evaluation
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Initialize TF Session
init = tf.global_variables_initializer()
saver = tf.train.Saver()
pred_list = []
epoch_list = []
trn_acc = []
val_acc = []
val_auc = []

# Execute
start_tm = time.time()
with tf.Session() as sess:
    init.run()
    best_auc = 0
    auc_list = []
    break_list = []
    for epoch in range(n_epochs):
        for b in range(0,len(batch_i)):
            batch_x = trn_x.iloc[batch_i[b]]
            batch_y = trn_y[batch_i[b]]
            sess.run(training_op, feed_dict = {x: batch_x, y: batch_y})
        acc_train = accuracy.eval(feed_dict = {x: batch_x, y: batch_y})
        acc_test = accuracy.eval(feed_dict = {x: val_x, y: val_y})
        prob_test = logits.eval(feed_dict={x: val_x, y: val_y})
        epoch_auc = roc_auc_score(val_y, prob_test[:,1])
        print(epoch, 'Training Accuracy: ', acc_train, ' Validation Accuracy: ', acc_test, ' Validation AUC: ', epoch_auc)
        epoch_list.append(epoch)
        trn_acc.append(acc_train)
        val_acc.append(acc_test)
        val_auc.append(epoch_auc)
        auc_list.append(epoch_auc)
        best_auc = max(auc_list)
        if epoch_auc < best_auc:
            break_list.append(1)
        else:
            break_list = []
        if sum(break_list) >= early_stop_epocs:
            print("Stopping after " + str(epoch) + "\
                  epochs because AUC hasn't improved in \
                  " + str(early_stop_epocs) + " rounds.")
            break
    make_pred = logits.eval(feed_dict = {x: tst_x})
    pred_list.append(make_pred)
end_tm = time.time()
secs_elapsed = np.round(end_tm - start_tm,0)
print('Execution Time (sec.): ' + str(secs_elapsed))  

# Plot ROC Curve of Test Set
######################################################################################
y_pred = [i for i in pred_list[0][:,1]]
pred_df = pd.DataFrame({'Actual': tst_y, 'Predicted': y_pred})
tf_roc = roc_auc_score(pred_df['Actual'], pred_df['Predicted'])
tf_fpr, tf_tpr, tf_thresholds = roc_curve(pred_df['Actual'], pred_df['Predicted'])
tf_auc = auc(tf_fpr, tf_tpr)
plt.title('Test Set ROC Curve - Tensorflow Model')
plt.plot(tf_fpr, tf_tpr, 'b', label='AUC = %0.2f'% tf_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])
plt.legend(loc='lower right')
plt.show()

# Plot Training Progress
######################################################################################
fig = plt.figure()
fig.show()
ax = fig.add_subplot(111)
ax.plot(epoch_list, trn_acc, label = 'Training Accuracy')
ax.plot(epoch_list, val_acc, 'b', label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
axes.set_ylim([min(trn_acc + val_acc),1])
plt.draw() 

"""
Notes:

    .91 AUC
learn_rt = 0.025
n_classes = 2
n_hidden1 = 100
n_hidden2 = 100
dropout = 0.3
n_epochs = 40
batch_size = 1000
early_stop_epocs = 4

"""

