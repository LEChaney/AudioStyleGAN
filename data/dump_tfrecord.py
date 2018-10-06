import os
import sys

import numpy as np
from scipy.io.wavfile import write as wavwrite
import tensorflow as tf

out_dir, tfrecord_fps = sys.argv[1], sys.argv[2:]

if not os.path.isdir(out_dir):
  os.makedirs(out_dir)

def pitch_speed_aug(wav):
  audio_sample = wav.copy()
  length_change = np.random.uniform(low=0.9,high=1.1)
  speed_fac = 1.0  / length_change
  tmp = np.interp(np.arange(0,len(audio_sample),speed_fac),np.arange(0,len(audio_sample)),audio_sample)
  minlen = min(audio_sample.shape[0], tmp.shape[0])
  audio_sample *= 0
  audio_sample[0:minlen] = tmp[0:minlen]
  return audio_sample

def _mapper(serialized_data):
  context_features = {
      'samples': tf.VarLenFeature(dtype=tf.float32),
      'label': tf.FixedLenFeature([], dtype=tf.string),
      'id': tf.FixedLenFeature([], dtype=tf.string),
      'conditioning_texts': tf.VarLenFeature(dtype=tf.string),
  }
  sequence_features = {
      'cond_text_embeds': tf.FixedLenSequenceFeature([1024], tf.float32, allow_missing=True)
  }

  context_data, sequence_data = tf.parse_single_sequence_example(serialized_data, context_features=context_features, sequence_features=sequence_features)

  wav = tf.sparse_tensor_to_dense(context_data['samples'])

  wav = wav[:16384]
  wav_len = tf.shape(wav)[0]
  wav = tf.pad(wav, [[0, 16384 - wav_len]])

  # Data Augmentation Test
  wav_aug = tf.py_func(pitch_speed_aug, [wav], tf.float32)
  wav_aug *= tf.random_uniform([1], minval=0.5, maxval=1.1) # Change dynamic range

  label = context_data['label']
  id = context_data['id']

  cond_texts = tf.sparse_tensor_to_dense(context_data['conditioning_texts'], default_value='')
  cond_text_embeds = sequence_data['cond_text_embeds']
  # probs = tf.reshape(tf.ones_like(cond_text_embeds, tf.float32)[:, 0], [-1])
  # samples = tf.multinomial(tf.log([probs]), 1) # note log-prob
  # cond_text = cond_texts[tf.cast(samples[0][0], tf.int32)]
  # cond_text_embed = cond_text_embeds[tf.cast(samples[0][0], tf.int32)]

  return wav, wav_aug, id, cond_texts, cond_text_embeds

dataset = tf.data.TFRecordDataset(tfrecord_fps)
dataset = dataset.map(_mapper)
dataset = dataset.batch(1, drop_remainder=True)
wav, wav_aug, id, cond_texts, cond_text_embeds = dataset.make_one_shot_iterator().get_next()
wav, wav_aug, id, cond_texts, cond_text_embeds = wav[0], wav_aug[0], id[0], cond_texts[0], cond_text_embeds[0]

with tf.Session() as sess:
  i = 0
  while True:
    try:
      _wav, _wav_aug, _id, _cond_texts, _cond_text_embeds = sess.run([wav, wav_aug, id, cond_texts, cond_text_embeds])
      # print(_wav.shape)
      # print(_id)
      # print(_cond_texts)
      # print(_cond_text_embeds)
      # print()
    except tf.errors.OutOfRangeError:
      break

    _wav *= 32767.
    _wav_aug *= 32767.
    _wav = np.clip(_wav, -32767., 32767.)
    _wav_aug = np.clip(_wav_aug, -32767., 32767.)
    _wav = _wav.astype(np.int16)
    _wav_aug = _wav_aug.astype(np.int16)
    out_path = os.path.join(out_dir, '{}_{}.wav'.format(_id, str(i).zfill(5)))
    aug_out_path = os.path.join(out_dir, '{}_{}_aug.wav'.format(_id, str(i).zfill(5)))
    wavwrite(out_path, 16000, _wav)
    wavwrite(aug_out_path, 16000, _wav_aug)
    with open('{}_cond.txt'.format(out_path), 'w') as cond_out:
      cond_out.write('\n'.join(map(lambda x: x.decode(), _cond_texts)))
    i += 1
