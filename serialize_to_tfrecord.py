import tensorflow as tf
from json import loads
#import IPython.display as display

# FEATURE_DESCRIPTION = {
#     'eye_left': tf.io.FixedLenFeature([], tf.string),
#     'eye_right': tf.io.FixedLenFeature([], tf.string),
#     'l_l1': tf.io.FixedLenFeature([2], tf.int64),
#     'l_l2': tf.io.FixedLenFeature([2], tf.int64),
#     'r_l1': tf.io.FixedLenFeature([2], tf.int64),
#     'r_l2': tf.io.FixedLenFeature([2], tf.int64),
#     'label': tf.io.FixedLenFeature([2], tf.float32)
# }

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value[0], value[1]]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value[0], value[1]]))

def serialize_example(eye_left, eye_right, left_lm: list[list[int, int], list[int, int]], 
                      right_lm: list[list[int, int], list[int, int]], label: list[ int, int]):
    '''Serialize features of a single example and return a serialized string'''
    feature = {
        'eye_left' : _bytes_feature(eye_left),
        'eye_right' : _bytes_feature(eye_right),
        'l_l1' : _int64_feature(left_lm[0]),
        'l_l2' : _int64_feature(left_lm[1]),
        'r_l1' : _int64_feature(right_lm[0]),
        'r_l2' : _int64_feature(right_lm[1]),
        'label' : _float_feature(label)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# def read_tfrecord():
#     raw_data = tf.data.TFRecordDataset('train.tfrecords')

#     data = raw_data.map(_parse_fun)

#     for example in data.take(1):
#         img = example['eye_left'].numpy()
#         display.display(display.Image(data=img))

# def _parse_fun(example_proto):
#     return tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)

def write_tfrecord():
    # Set total counters 
    train_co, test_co, valid_co, filt_out = 0, 0, 0, 0

    # Open the tfrecords files
    train_writer = tf.io.TFRecordWriter('train.tfrecords')
    test_writer = tf.io.TFRecordWriter('test.tfrecords')
    valid_writer = tf.io.TFRecordWriter('valid.tfrecords')

    with open('processed/info.json', 'r') as jsonl:
        participant_data = loads(jsonl.read())
        for pid in participant_data:
            for example in participant_data[pid]:

                raw_example_line = example

                # Filter out arrays with number of landmarks != 2
                # if (len(raw_example_line['left_landmarks']) != 2 
                #     or len(raw_example_line['right_landmarks']) != 2): filt_out += 1; continue
                
                # Read the images from the processed_frames folder
                eye_left = open('processed/' + raw_example_line['file_name_left'], 'rb').read()
                eye_right = open('processed/' + raw_example_line['file_name_right'], 'rb').read()

                # Serialize a single example
                example = serialize_example(eye_left, eye_right, raw_example_line['left_landmarks'], 
                                            raw_example_line['right_landmarks'], raw_example_line['label'])
                
                # Decide which sample to write to based on the split in the dataset
                if raw_example_line['split'] == 'train': train_writer.write(example); train_co += 1
                elif raw_example_line['split'] == 'test': test_writer.write(example); test_co += 1
                elif raw_example_line['split'] ==  'valid': valid_writer.write(example); valid_co += 1
                else: print('Split not recognized')

    print('Done serializing.')
    print(f'Split totals: train: {train_co}, test: {test_co}, valid: {valid_co}')
    print(f'Total filtered out {filt_out}')


if __name__ == '__main__':
    write_tfrecord()