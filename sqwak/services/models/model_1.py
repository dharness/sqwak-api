import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random


###############################################################
# Prepare the data
###############################################################
def make_batches(samples, keys, batch_size=-1):

  one_hot_labels = []
  feature_sets = []
  num_classes = len(list(keys.keys()))

  for i, sample in enumerate(samples):
    label = sample[0]
    feature_set = sample[1]
    one_hot = [0] * num_classes
    one_hot[keys[label]] = 1
    one_hot_labels.append(one_hot)
    feature_sets.append(feature_set)

  num_batches = int(len(feature_sets) / batch_size)

  if batch_size == -1:
    num_batches = 2
  batches = []

  for i in range(num_batches - 1):
    from_pos = i * batch_size
    to_pos = i * batch_size + batch_size

    features_batch = feature_sets[from_pos: to_pos]
    labels_batch = one_hot_labels[from_pos: to_pos]
    batches.append({
        'feature_sets': features_batch,
        'labels': labels_batch
    })
  return batches


###############################################################
# Train the model
###############################################################
def train(samples):

  keys = {}

  unique_ml_class_names = set(samples[:, 0])
  for i, ml_class_name in enumerate(unique_ml_class_names):
    keys[ml_class_name] = i

  random.shuffle(samples)
  batches = make_batches(samples, keys, 20)
  num_classes = len(unique_ml_class_names)

  x = tf.placeholder(tf.float32, [None, 275])
  W = tf.Variable(tf.zeros([275, num_classes]))
  b = tf.Variable(tf.zeros([num_classes]))
  y = tf.matmul(x, W) + b
  y_true = tf.placeholder(tf.float32, [None, num_classes])

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

  train_step = tf.train.GradientDescentOptimizer(
      0.0001).minimize(cross_entropy)

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
  n = len(list(model_paramaters['class_labels'].keys()))
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
  computed_probabilities = np.around(
      computed_probabilities[0] * 100, decimals=6)
  results = {}

  for i, class_label in enumerate(model_paramaters['class_labels'].keys()):
    position = model_paramaters['class_labels'][class_label]
    results[class_label] = str(computed_probabilities[position])

  return results
