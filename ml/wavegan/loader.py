import tensorflow as tf
import numpy as np
from multiprocessing import cpu_count

"""
  Augment wav audio data by varying the pitch.
  Use tf.py_func to process 1D tensorflow wavforms
"""
def pitch_speed_aug(wav):
  audio_sample = wav.copy()
  length_change = np.random.uniform(low=0.9,high=1.1)
  speed_fac = 1.0  / length_change
  tmp = np.interp(np.arange(0,len(audio_sample),speed_fac),np.arange(0,len(audio_sample)),audio_sample)
  minlen = min(audio_sample.shape[0], tmp.shape[0])
  audio_sample *= 0
  audio_sample[0:minlen] = tmp[0:minlen]
  return audio_sample

"""
  Data loader
  fps: List of tfrecords
  batch_size: Resultant batch size
  window_len: Size of slice to take from each example
  first_window: If true, always take the first window in the example, otherwise take a random window
  repeat: If false, only iterate through dataset once
  labels: If true, return audio label as the second output (or first is wavs kwarg is False)
  buffer_size: Number of examples to queue up (larger = more random)
  conditionals: If true, return conditional text as an aditional final output
  wavs: If true, return the audio sample as the first output
"""
def get_batch(
    fps,
    batch_size,
    window_len=16384,
    first_window=False,
    repeat=True,
    labels=False,
    buffer_size=8192,
    conditionals=False,
    wavs=True,
    name=None):
  
  def _mapper(example_proto):
    context_features = {'samples': tf.VarLenFeature(dtype=tf.float32)}
    if labels:
      context_features['id'] = tf.FixedLenFeature([], dtype=tf.string)
    if conditionals:
      context_features['conditioning_texts'] = tf.VarLenFeature(dtype=tf.string)

    sequence_features = {
      'cond_text_embeds': tf.FixedLenSequenceFeature([1024], tf.float32, allow_missing=True)
    }

    context_data, sequence_data = tf.parse_single_sequence_example(example_proto, context_features=context_features, sequence_features=sequence_features)
    
    if labels:
      label = tf.identity(context_data['id'], name='audio_name')
    
    # Construct elmo embedding of conditional text
    if conditionals:
      cond_texts = tf.sparse_tensor_to_dense(context_data['conditioning_texts'], default_value='', name='conditional_texts')
      cond_text_embeds = tf.identity(sequence_data['cond_text_embeds'], name='cond_text_embeds')
      probs = tf.identity(tf.reshape(tf.ones_like(cond_text_embeds, tf.float32)[:, 0], [-1]), name='cond_text_selection_probs')
      samples = tf.identity(tf.multinomial(tf.log([probs]), 1), name='cond_text_sample_indices') # note log-prob
      cond_text = tf.identity(cond_texts[tf.cast(samples[0][0], tf.int32)], name='conditional_text')
      cond_text_embed = tf.identity(cond_text_embeds[tf.cast(samples[0][0], tf.int32)], name='cond_text_embed')


    if wavs:
      wav = tf.sparse_tensor_to_dense(context_data['samples'])
      if first_window:
        # Use first window
        wav = wav[:window_len]
      else:
        # Select random window
        wav_len = tf.shape(wav)[0]

        start_max = wav_len - window_len
        start_max = tf.maximum(start_max, 0)

        start = tf.random_uniform([], maxval=start_max + 1, dtype=tf.int32)

        wav = wav[start:start+window_len]

      wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]]])

      # Runtime Data Augmentation
      wav = tf.py_func(pitch_speed_aug, [wav], tf.float32)
      wav *= tf.random_uniform([1], minval=0.5, maxval=1.1) # Change dynamic range

      wav = tf.reshape(wav, [-1, 1], name='real_audio')
      wav.set_shape([window_len, 1])

    if wavs and labels and conditionals:
      return wav, label, cond_text, cond_text_embed
    elif wavs and labels:
      return wav, label
    elif wavs and conditionals:
      return wav, cond_text, cond_text_embed
    elif labels and conditionals:
      return labels, cond_text, cond_text_embed
    elif labels:
      return label
    elif conditionals:
      return cond_text, cond_text_embed
    else:
      return wav
    

  dataset = tf.data.TFRecordDataset(fps)
  dataset = dataset.map(_mapper, num_parallel_calls=cpu_count() // 2)
  if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.batch(batch_size, drop_remainder=True)
  if repeat:
    dataset = dataset.repeat()
  dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0', 10))
  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next(name=name)
