# Import Packages
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.datasets import fetch_california_housing
import time

# Example 1: pg. 234 - 235
###############################################################
# Perform Basic Computation with Tensorflow
x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')
f = x*x*y + y + 2

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# Easier Way Without Executing 'sess.run' on Every Line
x = tf.Variable(3, name = 'x')
y = tf.Variable(4, name = 'y')
f = x*x*y + y + 2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

# Even Easier Way Without Running Initializer for Every Variable
init = tf.global_variables_initializer()    # prepare an init node

with tf.Session() as sess:
    init.run()
    result = f.eval()

# Managing Graphs: pg. 236
###############################################################
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

# Linear Regression in Tensorflow: pg. 238
###############################################################
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype = tf.float32, name = 'X')         # independent variables
y = tf.constant(housing.target.reshape(-1,1), dtype = tf.float32, name = 'Y')   # dependent variables
XT = tf.transpose(X)                                                
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()         # beta coefficients
    #IVs = XT.eval()
    #DV = y.eval()

# Training a DNN Using Plain Tensorflow: pg. 267
###############################################################

# This example isn't necessary - tensorflow has functions for this - for demonstration purposes
def gen_neuron_layer(x, n_neurons, name, activation = None):
    with tf.name_scope(name):
        # no. inputs determined by input matrix size
        n_inputs = int(x.get_shape()[1]) 
        # create weights for the matrix
        stddev = 2 / np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev = stddev)
        w = tf.Variable(init, name = 'kernel')
        # create variable for bias
        b = tf.Variable(tf.zeroes([n_neurons]), name = 'bias')
        # create variable for weighted sums of inputs plus bias term for each neuron
        z = tf.matmul(x, w) + b
        # allow for user-specified activation 
        if activation is not None:
            return activation(z)
        else:
            return z
        
# Instead:
############################################################### 
# MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')
        
# Parameters
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01
n_epochs = 40
batch_size = 50

# Placeholders
x = tf.placeholder(tf.float32, shape = (None, n_inputs), name = 'x') # placeholder for input layer
y = tf.placeholder(tf.int64, shape = (None), name = 'y')
        
# Define Layers
with tf.name_scope('dnn'):
    hidden1 = tf.layers.dense(x, n_hidden1, name = 'hidden1', activation = tf.nn.relu)
    hidden2 = tf.layers.dense(x, n_hidden2, name = 'hidden2', activation = tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name = 'outputs')
    
# Define Loss Function
with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = 'loss')
    
# Define Gradient Descent Optimizer
with tf.name_scope('train'): 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
# Define Performance Evaluation
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
# Initialize TF Session
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execute
start_tm = time.time()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {x: x_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict = {x: x_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict = {x: mnist.validation.images,
                                             y: mnist.validation.labels})
        print(epoch, 'Training Accuracy: ', acc_train, ' Validation Accuracy: ', acc_val)
end_tm = time.time()
secs_elapsed = end_tm - start_tm
print(secs_elapsed)      


