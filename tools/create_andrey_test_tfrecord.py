import os
import io
import copy
import random
import re
import string

from PIL import Image
import tensorflow as tf

from aster.utils import dataset_util
from aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_string('output_path', 'data/andrey_test.tfrecord', 'output path')
flags.DEFINE_integer('mixed_case', 0, 'mc or not')
FLAGS = flags.FLAGS

random.seed(1)


def create_andrey_test(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  files_list = os.listdir(FLAGS.data_dir)
  count = 0
  skipped = 0
  lengths = set()
  for image in files_list:
    image_rel_path = image
    image_path = os.path.join(FLAGS.data_dir, image_rel_path)

    groundtruth_text = image_rel_path.split('_')[0]
    if (FLAGS.mixed_case and
        len(list(filter(lambda x: x not in string.ascii_lowercase, groundtruth_text.lower()))) != 0):
        print('skipped {}'.format(groundtruth_text))
        skipped += 1
        continue

    if FLAGS.mixed_case:
        groundtruth_text = groundtruth_text.lower()

    im = Image.open(image_path)
    lengths.add(len(groundtruth_text))
    im_buff = io.BytesIO()
    im.save(im_buff, format='jpeg')
    word_jpeg = im_buff.getvalue()
    crop_name = '{}'.format(image_rel_path)
  
    lexicon = image[3]
    

    example = tf.train.Example(features=tf.train.Features(feature={
      fields.TfExampleFields.image_encoded: \
        dataset_util.bytes_feature(word_jpeg),
      fields.TfExampleFields.image_format: \
        dataset_util.bytes_feature('jpeg'.encode('utf-8')),
        fields.TfExampleFields.filename: \
        dataset_util.bytes_feature(crop_name.encode('utf-8')),
      fields.TfExampleFields.channels: \
        dataset_util.int64_feature(3),
      fields.TfExampleFields.colorspace: \
        dataset_util.bytes_feature('rgb'.encode('utf-8')),
      fields.TfExampleFields.transcript: \
        dataset_util.bytes_feature(groundtruth_text.encode('utf-8')),
    }))
    writer.write(example.SerializeToString())
    count += 1

  writer.close()
  print('{} examples created'.format(count))
  print('{} examples skipped'.format(skipped))
  print(repr(lengths))


if __name__ == '__main__':
  create_andrey_test(FLAGS.output_path)
