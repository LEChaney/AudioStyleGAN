import tensorflow as tf

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
    features = {'samples': tf.FixedLenSequenceFeature([1], tf.float32, allow_missing=True)}
    if labels:
      features['label'] = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)
    if conditionals:
      features['conditional_text'] = tf.FixedLenSequenceFeature([], tf.string, allow_missing=True)

    example = tf.parse_single_example(example_proto, features)
    
    if labels:
      label = tf.reduce_join(example['label'], 0)
    
    # Construct elmo embedding of conditional text
    if conditionals:
      cond_text = tf.reduce_join(example['conditional_text'], 0, name='conditional_text')

    if wavs:
      wav = example['samples']
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

      wav = tf.pad(wav, [[0, window_len - tf.shape(wav)[0]], [0, 0]], name='audio_sample')

      wav.set_shape([window_len, 1])

    if wavs and labels and conditionals:
      return wav, label, cond_text
    elif wavs and labels:
      return wav, label
    elif wavs and conditionals:
      return wav, cond_text
    elif labels and conditionals:
      return labels, cond_text
    elif labels:
      return label
    elif conditionals:
      return cond_text
    else:
      return wav
    

  dataset = tf.data.TFRecordDataset(fps)
  dataset = dataset.map(_mapper)
  if repeat:
    dataset = dataset.shuffle(buffer_size=buffer_size)
  dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  if repeat:
    dataset = dataset.repeat()
  iterator = dataset.make_one_shot_iterator()

  return iterator.get_next(name=name)
