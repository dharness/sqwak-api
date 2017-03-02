import tensorflow as tf
from math import floor
import numpy as np
import os
import glob
from file_reader import read_and_decode_single_example, get_all_samples
import random

samples = get_all_samples("sqwak_sounds.train.tfrecords")

def make_batches(data, batch_size=-1):
    sounds = data['sounds']
    labels = data['labels']
    num_batches = len(sounds)/batch_size
    if batch_size == -1:
        num_batches = 2
    batches = []
    for i in range(num_batches-1):
        sounds_batch = sounds[i*batch_size:i*batch_size+batch_size]
        labels_batch = labels[i*batch_size:i*batch_size+batch_size]
        batches.append({
            'sounds': sounds_batch,
            'labels': labels_batch
        })
    return batches

batches = make_batches(samples, 20)
random.shuffle(batches)

###############################################################
# Train the model
###############################################################

x = tf.placeholder(tf.float32, [None, 275])
W = tf.Variable(tf.zeros([275, 3]))
b = tf.Variable(tf.zeros([3]))
y = tf.matmul(x, W) + b

y_true = tf.placeholder(tf.float32, [None, 3])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
files = glob.glob('./logs/train/*')
for f in files:
    os.remove(f)
train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
tf.global_variables_initializer().run()
tf.train.start_queue_runners(sess=sess)

# Train

for i, batch in enumerate(batches):
  batch_xs = batch['sounds']
  batch_ys = batch['labels']
  _, loss, summary = sess.run([train_step, cross_entropy, merged], feed_dict={
    y_true: batch_ys,
    x: batch_xs
  })
#   print(i, loss)
  train_writer.add_summary(summary, i)


# Test the model
test_samples = get_all_samples("sqwak_sounds.test.tfrecords")
test_batch = make_batches(test_samples)[0]

test_labels = test_batch['labels']
test_sounds = test_batch['sounds']


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# divide by 100 to scale the results for more significant decimal places
# int the result
probabilities = tf.nn.softmax(tf.div(y, 100))

accuracy_computed, probs_computed = sess.run([accuracy, probabilities], feed_dict={
  x: test_sounds, 
  y_true: test_labels
})

# print(accuracy_computed)
print(np.sum(probs_computed[0]))