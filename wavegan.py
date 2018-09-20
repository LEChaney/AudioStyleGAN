import tensorflow as tf


def conv1d_transpose(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample='zeros'):
  if upsample == 'zeros':
    return tf.layers.conv2d_transpose(
        tf.expand_dims(inputs, axis=1),
        filters,
        (1, kernel_width),
        strides=(1, stride),
        padding='same'
        )[:, 0]
  elif upsample == 'nn':
    batch_size = tf.shape(inputs)[0]
    _, w, nch = inputs.get_shape().as_list()

    x = inputs

    x = tf.expand_dims(x, axis=1)
    x = tf.image.resize_nearest_neighbor(x, [1, w * stride])
    x = x[:, 0]

    return tf.layers.conv1d(
        x,
        filters,
        kernel_width,
        1,
        padding='same')
  else:
    raise NotImplementedError

def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)

def compress_embedding(c, out_size):
  """
  c: Embedded context to reduce, this should be a [batch_size x N] tensor
  out_size: The size of the new embedding
  returns: [batch_size x out_size] tensor
  """
  with tf.variable_scope('reduce_embed'):
    return lrelu(tf.layers.dense(c, out_size))

"""
  Input: [None, 100]
  Output: [None, 16384, 1]
"""
def WaveGANGenerator(
    z,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    upsample='zeros',
    train=False,
    context_embedding=None,
    embedding_dim=128):
  batch_size = tf.shape(z)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=train)
  else:
    batchnorm = lambda x: x

  if (context_embedding is not None):
    # Reduce or expand context embedding to be size [embedding_dim]
    c = compress_embedding(context_embedding, embedding_dim)
    output = tf.concat([z, c], 1)
  else:
    output = z

  # FC and reshape for convolution
  # [100 + context_embedding size] -> [16, 1024]
  with tf.variable_scope('z_project'):
    output = tf.layers.dense(output, 4 * 4 * dim * 16)
    output = tf.reshape(output, [batch_size, 16, dim * 16])
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = conv1d_transpose(output, dim * 8, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = conv1d_transpose(output, dim * 4, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = conv1d_transpose(output, dim * 2, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = conv1d_transpose(output, dim, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  output = tf.nn.relu(output)

  # Layer 4
  # [4096, 64] -> [16384, 1]
  with tf.variable_scope('upconv_4'):
    output = conv1d_transpose(output, 1, kernel_len, 4, upsample=upsample)
  output = tf.nn.tanh(output)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if len(update_ops) != 10:
      raise Exception('Other update ops found in graph')
    with tf.control_dependencies(update_ops):
      output = tf.identity(output)

  return output


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    kernel_len=25,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    context_embedding=None,
    embedding_dim=128):
  batch_size = tf.shape(x)[0]

  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  # Layer 0
  # [16384, 1] -> [4096, 64]
  output = x
  with tf.variable_scope('downconv_0'):
    output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
    output = batchnorm(output)
  output = lrelu(output)

  if (context_embedding is not None):
    # Reduce size of context embedding
    # Context dims: [1024] -> [128]
    c = compress_embedding(context_embedding, embedding_dim)

    # Replicate context 
    # Context dims: [128] -> [1, 128]
    c = tf.expand_dims(c, 1)
    # Context dims: [1, 128] -> [16, 128]
    c = tf.tile(c, [1, 16, 1])

    # Concat context with encoded audio along the channels dimension
    # [16, 1024] -> [16, 1152]
    output = tf.concat([output, c], 2)

    # 1x1 Convolution over combined features
    # [16, 1152] -> [16, 1024]
    output = phaseshuffle(output)
    with tf.variable_scope('1_by_1_conv'):
      output = tf.layers.conv1d(output, dim * 16, kernel_len, 1, padding='SAME')
      output = batchnorm(output)
    output = lrelu(output)

  # Flatten
  # [16, 1024] -> [16384]
  output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])

  # if (context_embedding is not None):
  #   # Concat context embeddings
  #   # [16384] -> [16384 + embedding_dim]
  #   c = compress_embedding(context_embedding, embedding_dim)
  #   output = tf.concat([output, c], 1)

  #   # FC
  #   # [16384 + embedding_dim] -> [1024]
  #   with tf.variable_scope('FC'):
  #     output = tf.layers.dense(output, 1024)
  #     output = tf.nn.relu(output)

  # Connect to single logit
  # [16384] -> [1]
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
