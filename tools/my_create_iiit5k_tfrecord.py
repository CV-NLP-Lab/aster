import logging
import os
import io
import random
import PIL.Image

import tensorflow as tf
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from aster.utils import dataset_util
from aster.core import standard_fields as fields
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_string('is50', 'True', '50 lexicon or 1k')
FLAGS = flags.FLAGS


def create_iiit5k_subset(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  gt_path = os.path.join(FLAGS.data_dir, 'iiit5k_50_test.txt' if FLAGS.is50 == 'True' else 'iiit5k_1k_test.txt')

  
  with open(gt_path, 'r') as gt:
    images = [tline.rstrip('\n').split(' ') for tline in gt.readlines()]
    for image in images:
        image[2] = int(image[2])
        image[3] = image[3].split(',')
  count = 0
  for image in images:
    image_rel_path = image[0]
    image_path = os.path.join(FLAGS.data_dir, image_rel_path)
    im = Image.open(image_path)

    groundtruth_text = image[1]

    im_buff = io.BytesIO()
    im.save(im_buff, format='jpeg')
    image_jpeg = im_buff.getvalue()

    lexicon = image[3]


    example = tf.train.Example(features=tf.train.Features(feature={
      fields.TfExampleFields.image_encoded: \
        dataset_util.bytes_feature(image_jpeg),
      fields.TfExampleFields.image_format: \
        dataset_util.bytes_feature('jpeg'.encode('utf-8')),
      fields.TfExampleFields.filename: \
        dataset_util.bytes_feature(image_rel_path.encode('utf-8')),
      fields.TfExampleFields.channels: \
        dataset_util.int64_feature(3),
      fields.TfExampleFields.colorspace: \
        dataset_util.bytes_feature('rgb'.encode('utf-8')),
      fields.TfExampleFields.transcript: \
        dataset_util.bytes_feature(groundtruth_text.encode('utf-8')),
      fields.TfExampleFields.lexicon: \
        dataset_util.bytes_feature(('\t'.join(lexicon)).encode('utf-8'))
    }))
    writer.write(example.SerializeToString())
    count += 1

  print(count)
  writer.close()


if __name__ == '__main__':
  # create_iiit5k_subset('data/iiit5k_train.tfrecord', train_subset=True)
  create_iiit5k_subset('data/iiit5k_test_50.tfrecord' if FLAGS.is50 == 'True' else 'data/iiit5k_test_1k.tfrecord')
  # create_iiit5k_subset('data/iiit5k_test_1k.tfrecord', train_subset=False, lexicon_index=3)
