import logging
import os
import io
import random
from PIL import Image


import tensorflow as tf
import numpy as np
from tqdm import tqdm

from aster.utils import dataset_util
from aster.core import standard_fields as fields


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_integer('start_index', 0, 'Start image index.')
flags.DEFINE_integer('num_images', -1, 'Number of images to create. Default is all remaining.')
flags.DEFINE_integer('shuffle', 0, 'Shuffle images.')
flags.DEFINE_string('output_path', 'data/andrey_train.tfrecord', 'Path to output TFRecord.')
FLAGS = flags.FLAGS


def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # load groundtruth file
  files_list = os.listdir(FLAGS.data_dir)

  #groundtruth_file = os.path.join(FLAGS.data_dir, 'annotation.txt')
  #with open(groundtruth_file, 'r') as f:
  #  groundtruth_lines = f.readlines()

  num_images = len(files_list) - FLAGS.start_index
  if FLAGS.num_images > 0:
    num_images = min(num_images, FLAGS.num_images)

  indices = list(range(FLAGS.start_index, FLAGS.start_index + num_images))
  if FLAGS.shuffle:
    random.shuffle(indices)

  # a test decode pipeline for validating image
  image_jpeg_input = tf.placeholder(
    dtype=tf.string,
    shape=[]
  )
  image = tf.image.decode_jpeg(
    image_jpeg_input,
    channels=3,
    try_recover_truncated=False,
    acceptable_fraction=1
  )
  count = 0
  skipped = 0
  with tf.Session() as sess:
    for index in tqdm(indices):
      image_rel_path = files_list[index]
      image_path = os.path.join(FLAGS.data_dir, image_rel_path)

      # validate image
      valid = True
      image_jpeg = None
      try:
        #with open(image_path, 'rb') as f:
        pil_image = Image.open(image_path)
        im_buff = io.BytesIO()
        pil_image.save(im_buff, format='jpeg')
        image_jpeg = im_buff.getvalue()
        image_output = sess.run(image, feed_dict={
          image_jpeg_input: image_jpeg
        })
        if (image_output.ndim != 3 or
            image_output.shape[0] == 0 or
            image_output.shape[1] == 0 or
            image_output.shape[2] != 3):
#          print(image_output.shape, image_output.ndim)
          valid = False
      except Exception as e:
#        print(e)
        valid = False
      
      if not valid:
        logging.warn('Skip invalid image {}'.format(image_path))
        skipped += 1
        continue

      # extract groundtruth
      groundtruth_text = image_rel_path.split('_')[0]

      # write example
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
          dataset_util.bytes_feature(groundtruth_text.encode('utf-8'))
      }))
      writer.write(example.SerializeToString())
      count += 1
  print('Images written: ', count)
  print('Images skipped: ', skipped)
  writer.close()

if __name__ == '__main__':
  tf.app.run()
