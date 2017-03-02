import tensorflow as tf

def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.VarLenFeature(tf.float32),
            'sound': tf.FixedLenFeature([275], tf.float32)
        })
    label = features['label']
    sound = features['sound']
    return label, sound

def get_all_samples(filename):
    sounds = []
    labels = []
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        sound = example.features.feature['sound'].float_list.value
        label = example.features.feature['label'].float_list.value
        sounds.append(sound)
        labels.append(label)
    return {
        'labels': labels,
        'sounds': sounds
    }
    
if __name__ == "__main__":
    filename = "sqwak_sounds.train.tfrecords"

    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        sound = example.features.feature['sound'].float_list.value
        label = example.features.feature['label'].float_list.value

        print label