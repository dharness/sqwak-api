import tensorflow as tf
from math import floor
import numpy as np
from file_reader import read_and_decode_single_example

###############################################################
# Train the model
###############################################################

# get single examples
label, sound = read_and_decode_single_example("sqwak_sounds.tfrecords")
sound = tf.cast(sound, tf.float32) / 255.
# groups examples into batches randomly
sounds_batch, labels_batch = tf.train.shuffle_batch(
    [sound, label], batch_size=128,
    capacity=2000,
    min_after_dequeue=1000)

# simple model
w = tf.get_variable("w1", [275, 10])
y_pred = tf.matmul(sounds_batch, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

for i in range(1000):
  # pass it in through the feed_dict
  _, loss_val = sess.run([train_op, loss_mean])
  print loss_val