import tensorflow as tf
import random
import json
import numpy as np
from math import floor
import argparse


###############################################################
# Prepare the data
###############################################################

def main():
  with open('premade_classes.json') as json_data:
      sounds = json.load(json_data)

  random.shuffle(sounds)
  n = len(sounds)

  cutoff = int(floor(n * 0.7))

  train_sounds = sounds[:cutoff]
  test_sounds = sounds[cutoff:]

  write_records(train_sounds, "sqwak_sounds.train.tfrecords")
  write_records(test_sounds, "sqwak_sounds.test.tfrecords")

def write_records(data, filename):
  writer = tf.python_io.TFRecordWriter(filename)
  for sound in data:
    features = np.array(sound['features'])
    label_name = sound['label']
    if FLAGS.classes == None or label_name in FLAGS.classes:
      num_classes = len(FLAGS.classes) + 1
      label = [0]*num_classes
      label[keys[label_name]] = 1
      label = np.array(label)
      # label = keys[sound['label']]
      example = tf.train.Example(
          features=tf.train.Features(
            feature={
              'label': tf.train.Feature(float_list=tf.train.FloatList(value=label.astype("float"))),
              'sound': tf.train.Feature(float_list=tf.train.FloatList(value=features.astype("float")))
            }
          )
      )
      serialized = example.SerializeToString()
      writer.write(serialized)

if __name__ == '__main__':
  keys = {
    "jackhammer": 0,
    "siren": 1,
    "gun_shot": 2,
    "car_horn": 3,
    "air_conditioner": 4,
    "street_music": 5,
    "engine_idling": 6,
    "children_playing": 7,
    "dog_bark": 8,
    "drilling": 9
  }
  parser = argparse.ArgumentParser()
  parser.add_argument('--classes',
    type=str, 
    nargs='+', 
    default=None, 
    choices=list(keys.keys())
  )
  FLAGS, unparsed = parser.parse_known_args()
  main()