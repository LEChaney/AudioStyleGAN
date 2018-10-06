if __name__  == '__main__':
  import argparse
  import glob
  import os
  import random
  import sys

  import numpy as np
  import tensorflow as tf
  from tqdm import tqdm
  from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
  import tensorflow_hub as hub

  parser = argparse.ArgumentParser()

  parser.add_argument('in_dir', type=str)
  parser.add_argument('out_dir', type=str)
  parser.add_argument('--name', type=str)
  parser.add_argument('--ext', type=str)
  parser.add_argument('--fs', type=int)
  parser.add_argument('--nshards', type=int)
  parser.add_argument('--slice_len', type=float)
  parser.add_argument('--first_only', action='store_true', dest='first_only')
  parser.add_argument('--nrg_top_k', action='store_true', dest='nrg_top_k')
  parser.add_argument('--nrg_one_every', type=float)
  parser.add_argument('--nrg_min_per', type=int)
  parser.add_argument('--nrg_max_per', type=int)
  parser.add_argument('--labels', action='store_true', dest='labels')
  parser.add_argument('--labels_whitelist', type=str)

  parser.set_defaults(
      name='examples',
      ext='wav',
      fs=16000,
      nshards=1,
      slice_len=None,
      first_only=False,
      nrg_top_k=False,
      nrg_one_every=5.,
      nrg_min_per=1,
      nrg_max_per=4,
      labels=False,
      labels_whitelist=None)

  args = parser.parse_args()

  labels_whitelist = None
  if args.labels_whitelist is not None:
    labels_whitelist = set([l.strip() for l in args.labels_whitelist.split(',')])

  audio_fps = glob.glob(os.path.join(args.in_dir, '*.{}'.format(args.ext)))
  random.shuffle(audio_fps)
  cond_fps = [audio_fp + '_cond.txt' for audio_fp in audio_fps]

  if args.nshards > 1:
    npershard = max(int(len(audio_fps) // (args.nshards - 1)), 1)
  else:
    npershard = len(audio_fps)

  slice_len_samps = None
  if args.slice_len is not None:
    slice_len_samps = int(args.slice_len * args.fs)

  # Conditioning text
  cond_fp = tf.placeholder(tf.string, [])
  cond_dataset = tf.data.TextLineDataset([cond_fp]) # Multiple conditional texts per audio file
  cond_texts_iter = cond_dataset.make_initializable_iterator()
  cond_text = cond_texts_iter.get_next()

  # Conditional text embedding
  embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False, name='embed')
  cond_texts_batch = tf.placeholder(tf.string, [None])
  cond_text_embeds = embed(cond_texts_batch)

  audio_fp = tf.placeholder(tf.string, [])
  audio_bin = tf.read_file(audio_fp)
  samps = contrib_audio.decode_wav(audio_bin, 1).audio[:, 0]

  if slice_len_samps is not None:
    if args.first_only:
      pad_end = True
    else:
      pad_end = False

    slices = tf.contrib.signal.frame(samps, slice_len_samps, slice_len_samps, axis=0, pad_end=pad_end)

    if args.nrg_top_k:
      nsecs = tf.cast(tf.shape(samps)[0], tf.float32) / args.fs
      k = tf.cast(nsecs / args.nrg_one_every, tf.int32)
      k = tf.maximum(k, args.nrg_min_per)
      k = tf.minimum(k, args.nrg_max_per)

      nrgs = tf.reduce_mean(tf.square(slices), axis=1)
      _, top_k = tf.nn.top_k(nrgs, k)

      slices = tf.gather(slices, top_k, axis=0)

    if args.first_only:
      slices = slices[:1]
  else:
    slices = tf.expand_dims(samps, axis=0)

  sess = tf.Session()
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess, coord=coord)

  for i, start_idx in tqdm(enumerate(range(0, len(audio_fps), npershard))):
    shard_name = '{}-{}-of-{}.tfrecord'.format(args.name, str(i).zfill(len(str(args.nshards))), args.nshards)
    shard_fp = os.path.join(args.out_dir, shard_name)

    if not os.path.isdir(args.out_dir):
      os.makedirs(args.out_dir)

    writer = tf.python_io.TFRecordWriter(shard_fp)

    # Read and process conditional text
    _cond_texts_batch = []
    for _cond_fp in cond_fps[start_idx:start_idx+npershard]:
      # Read conditional text
      sess.run(cond_texts_iter.initializer, {cond_fp: _cond_fp})
      single_sample_cond_texts = []
      try:
        while True:
          single_sample_cond_texts.append(sess.run(cond_text))
      except tf.errors.OutOfRangeError:
        _cond_texts_batch.append(single_sample_cond_texts)
      
    # Embed conditional text
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    flat_cond_texts_batch = [item for sublist in _cond_texts_batch for item in sublist]
    _cond_text_embeds = sess.run(cond_text_embeds, {cond_texts_batch: flat_cond_texts_batch})

    embedding_start_idx = 0
    for offset, _audio_fp in enumerate(audio_fps[start_idx:start_idx+npershard]):
      num_embeddings = len(_cond_texts_batch[offset])
      embedding_end_idx = embedding_start_idx + num_embeddings
      sample_text_embeds = _cond_text_embeds[embedding_start_idx:embedding_end_idx]
      embedding_start_idx += num_embeddings

      audio_name = os.path.splitext(os.path.split(_audio_fp)[1])[0]
      if args.labels:
        audio_label, audio_id = audio_name.split('_', 1)

        if labels_whitelist is not None:
          if audio_label not in labels_whitelist:
            continue
      else:
        audio_id = audio_name
        audio_label = ''

      try:
        _slices = sess.run(slices, {audio_fp: _audio_fp})
      except:
        continue

      if _slices.shape[0] == 0 or _slices.shape[1] == 0:
        continue

      for j, _slice in enumerate(_slices):
        context = features=tf.train.Features(feature={
          'id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_id.encode()])),
          'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio_label.encode()])),
          'slice': tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
          'samples': tf.train.Feature(float_list=tf.train.FloatList(value=_slice)),
          'conditioning_texts': tf.train.Feature(bytes_list=tf.train.BytesList(value=_cond_texts_batch[offset])),
        })

        text_embed_features = []
        for cond_text_embed in sample_text_embeds:
          text_embed_feature = tf.train.Feature(float_list=tf.train.FloatList(value=cond_text_embed))
          text_embed_features.append(text_embed_feature)
        text_embeds = tf.train.FeatureList(feature=text_embed_features)
        feature_lists = tf.train.FeatureLists(feature_list={'cond_text_embeds': text_embeds})

        example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)

        writer.write(example.SerializeToString())

    writer.close()

  sess.close()
