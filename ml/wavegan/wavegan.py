import tensorflow as tf


def custom_conv1d(
    inputs,
    filters,
    kernel_width,
    stride=4,
    padding='same',
    upsample=None):
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
    return tf.layers.conv1d(
        inputs, 
        filters, 
        kernel_width, 
        strides=stride, 
        padding='same')

def residual_unit(    
    inputs,
    filters,
    kernel_width=25,
    stride=1,
    padding='same',
    upsample=None,
    activation=tf.nn.tanh,
    is_gated=True):
  # Shortcut connection
  if (upsample is not None) or (inputs.shape[-1] != filters) or (stride != 1):
    shortcut = custom_conv1d(inputs, filters, 1, stride, padding, upsample)
  else:
    shortcut = inputs

  # Up-Conv + Gated Activation
  z1 = custom_conv1d(inputs, filters, kernel_width, stride, padding, upsample)
  output = activation(z1)
  if is_gated:
    z2 = custom_conv1d(inputs, filters, kernel_width, stride, padding, upsample)
    gate = tf.sigmoid(z2)
    output = gate * output

  return output + shortcut


def dense_block(
    inputs,
    num_units,
    filters_per_unit=32,
    kernel_width=25,
    out_dim=None,
    activation=tf.tanh,
    batchnorm_fn=lambda x: x):
  """
  input: Input tensor
  num_units: Number of internal convolution units in the dense block
  batchnorm_fn: A function for normalizing each layer
  filters_per_unit: The number of filters produced by each unit, these are stacked together
  so the final output filters will be num_units * filters_per_unit + input filters
  out_dim: Settings this will override the output dimension using 1 by 1 convolution at end of block
  kernel_width: The size of the kernel used by each convolutional unit
  """
  output = inputs
  for i in range(num_units):
    with tf.variable_scope("unit_{}".format(i)):
      bn = batchnorm_fn(output)
      unit_out = residual_unit(bn, filters_per_unit, kernel_width, activation=activation)
      output = tf.concat([output, unit_out], 2)

  # Resize out dimensions on request
  if out_dim is not None:
    with tf.variable_scope("1_by_1"):
      return residual_unit(output, out_dim, 1, activation=activation)
  else:
    return output

def inception_block(inputs):
  filter1 = tf.layers.conv1d(inputs, 32, 1, padding="SAME")

  filter2 = tf.layers.conv1d(inputs, 32, 1, padding="SAME")
  filter2 = tf.layers.conv1d(filter2, 32, 9, padding="SAME")

  filter3 = tf.layers.conv1d(inputs, 32, 1, padding="SAME")
  filter3 = tf.layers.conv1d(filter3, 32, 25, padding="SAME")

  concat = tf.concat([filter1, filter2, filter3], 2)
  final_conv = tf.layers.conv1d(concat, inputs.shape[-1], 1, padding="SAME")

  return final_conv + inputs


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

  # Inception Blocks
  with tf.variable_scope('incept_0'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_1'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Up Conv 0
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_0'):
    output = residual_unit(output, dim * 8, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_2'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_3'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Up Conv 1
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_1'):
    output = residual_unit(output, dim * 4, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_4'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_5'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Up Conv 2
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_2'):
    output = residual_unit(output, dim * 2, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_6'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_7'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Up Conv 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_3'):
    output = residual_unit(output, dim, kernel_len, 4, upsample=upsample)
    output = batchnorm(output)
  #output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_8'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Inception Blocks
  with tf.variable_scope('incept_9'):
    output = inception_block(output)
    batchnorm(output)
  output = tf.nn.relu(output)

  # Up Conv 4
  # [4096, 64] -> [16384, 1]
  with tf.variable_scope('upconv_4'):
    output = residual_unit(output, 1, kernel_len, 4, upsample=upsample)
  #output = tf.nn.tanh(output)

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
    output = residual_unit(output, dim, kernel_len, 4, padding='SAME', activation=lrelu, is_gated=False)
  #output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 1
  # [4096, 64] -> [1024, 128]
  with tf.variable_scope('downconv_1'):
    output = residual_unit(output, dim, kernel_len, 4, padding='SAME', activation=lrelu, is_gated=False)
    output = batchnorm(output)
  #output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 2
  # [1024, 128] -> [256, 256]
  with tf.variable_scope('downconv_2'):
    output = residual_unit(output, dim * 2, kernel_len, 4, padding='SAME', activation=lrelu, is_gated=False)
    output = batchnorm(output)
  #output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 3
  # [256, 256] -> [64, 512]
  with tf.variable_scope('downconv_3'):
    output = residual_unit(output, dim * 2, kernel_len, 4, padding='SAME', activation=lrelu, is_gated=False)
    output = batchnorm(output)
  #output = lrelu(output)
  output = phaseshuffle(output)

  # Layer 4
  # [64, 512] -> [16, 1024]
  with tf.variable_scope('downconv_4'):
    output = residual_unit(output, dim * 4, kernel_len, 4, padding='SAME', activation=lrelu, is_gated=False)
    output = batchnorm(output)
  #output = lrelu(output)

  # Flatten
  # [16, 1024] -> [16384]
  output = tf.reshape(output, [batch_size, -1])

  if (context_embedding is not None):
    # Concat context embeddings
    # [16384] -> [16384 + embedding_dim]
    c = compress_embedding(context_embedding, embedding_dim)
    output = tf.concat([output, c], 1)

    # FC
    # [16384 + embedding_dim] -> [512]
    with tf.variable_scope('FC'):
      output = tf.layers.dense(output, dim * 4)
    output = tf.nn.relu(output)
    output = tf.layers.dropout(output)

  # Connect to single logit
  # [16384] -> [1]
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)[:, 0]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  return output
