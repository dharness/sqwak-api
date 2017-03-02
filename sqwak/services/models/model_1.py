import tensorflow as tf
from math import floor
import numpy as np
import os
import glob
import random
import tempfile


###############################################################
# Prepare the data
###############################################################
def make_batches(samples, keys, batch_size=-1):
  feature_sets = []
  labels = []
  num_classes = len(keys.keys())
  
  for sample in samples:
    class_id = keys[sample['label']]
    one_hot = [0]*num_classes
    one_hot[class_id] = 1

    labels.append(one_hot)
    feature_sets.append(sample['features'])

  num_batches = len(feature_sets)/batch_size
  
  if batch_size == -1:
      num_batches = 2
  batches = []
  
  for i in range(num_batches-1):
      from_pos = i*batch_size
      to_pos = i*batch_size+batch_size
      
      features_batch = feature_sets[from_pos : to_pos]
      labels_batch = labels[from_pos : to_pos]
      batches.append({
          'feature_sets': features_batch,
          'labels': labels_batch
      })
  return batches


###############################################################
# Train the model
###############################################################
def train(ml_classes):
  
  samples = []
  keys = {}
  for i, ml_class in enumerate(ml_classes):
    keys[ml_class['class_name']] = i
    samples += ml_class['audio_samples']

  random.shuffle(samples)
  batches = make_batches(samples, keys, 20)
  num_classes = len(keys.keys())

  x = tf.placeholder(tf.float32, [None, 275])
  W = tf.Variable(tf.zeros([275, num_classes]))
  b = tf.Variable(tf.zeros([num_classes]))
  y = tf.matmul(x, W) + b
  y_true = tf.placeholder(tf.float32, [None, num_classes])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

  train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  for i, batch in enumerate(batches):
    batch_xs = batch['feature_sets']
    batch_ys = batch['labels']
    _, loss = sess.run([train_step, cross_entropy], feed_dict={
      y_true: batch_ys,
      x: batch_xs
    })

  return {
    'W': W.eval(),
    'b': b.eval(),
    'class_labels': keys
  }

###############################################################
# Predict new samples
###############################################################
def predict(model_paramaters, features):
  n = len(model_paramaters['class_labels'].keys())
  x = tf.placeholder(tf.float32, [None, 275])
  W = tf.placeholder(tf.float32, [275, n])
  b = tf.placeholder(tf.float32, [n])
  y = tf.matmul(x, W) + b

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  probabilities = tf.nn.softmax(tf.div(y, 1000))

  predictions, computed_probabilities = sess.run([y, probabilities], feed_dict={
    W: model_paramaters['W'],
    b: model_paramaters['b'],
    x: [features]
  })


  np.set_printoptions(suppress=True)
  computed_probabilities = np.around(computed_probabilities[0]*100, decimals=6)
  results = {}

  for i, class_label in enumerate(model_paramaters['class_labels'].keys()):
    position = model_paramaters['class_labels'][class_label]
    results[class_label] = str(computed_probabilities[position])
  
  return results