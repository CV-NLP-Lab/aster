import os
import io
import copy
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image
import tensorflow as tf

from aster.utils import dataset_util
from aster.core import standard_fields as fields

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw SynthText dataset.')
flags.DEFINE_bool('ignore_difficult', True, 'Ignore words shorter than 3 or contain non-alphanumeric symbols')
flags.DEFINE_float('crop_margin', 0.2, 'Margin in percentage of word height')
FLAGS = flags.FLAGS

lexicon_size = 50
random.seed(1)


def _random_lexicon(lexicon_list, groundtruth_text, lexicon_size):
  lexicon = copy.deepcopy(lexicon_list)
  del lexicon[lexicon.index(groundtruth_text.lower())]
  random.shuffle(lexicon)
  lexicon = lexicon[:(lexicon_size-1)]
  lexicon.insert(0, groundtruth_text)
  return lexicon

def _is_difficult(word):
  assert isinstance(word, str)
  return len(word) < 3 or not re.match('^[\w]+$', word)

def create_ic03(output_path):
  writer = tf.python_io.TFRecordWriter(output_path)

  gt_path = os.path.join(FLAGS.data_dir, 'gt_lex50.txt')
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
      fields.TfExampleFields.lexicon: \
        dataset_util.bytes_feature(('\t'.join(lexicon)).encode('utf-8')),
    }))
    writer.write(example.SerializeToString())
    count += 1

  writer.close()
  print('{} examples created'.format(count))


if __name__ == '__main__':
  create_ic03('data/ic03_test.tfrecord')
