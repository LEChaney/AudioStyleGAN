import tensorflow as tf


def lerp_clip(a, b, t): 
  return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)


def nn_upsample(inputs, stride=4):
  '''
  Upsamples an audio clip using nearest neighbor upsampling.
  Output is of size 'audio clip length' x 'stride'
  '''
  with tf.variable_scope('upsample'):
    w = tf.shape(inputs)[1]
    output = tf.expand_dims(inputs, axis=1)
    output = tf.image.resize_nearest_neighbor(output, [1, w * stride])
    output = output[:, 0]
    return output


def avg_downsample(inputs, stride=4):
  with tf.variable_scope('downsample'):
    return tf.layers.average_pooling1d(inputs, pool_size=stride, strides=stride, padding='same')


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
    x = nn_upsample(inputs, stride)

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


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)
  

def to_audio(inputs):
  '''
  Converts 2d feature map into an audio clip.
  '''
  with tf.variable_scope('to_audio'):
    return tf.layers.conv1d(inputs, filters=1, kernel_size=1, strides=1, padding='same')

def from_audio(inputs, out_feature_maps):
  '''
  Converts an input audio clips into a 2d feature maps.
  :param out_feature_maps: The number of feature maps to output.
  '''
  with tf.variable_scope('from_audio'):
    return tf.layers.conv1d(inputs, filters=out_feature_maps, kernel_size=1, strides=1, padding='same')


def up_block(inputs, audio_lod, filters, on_amount, kernel_size=9, stride=4, activation=lrelu):
  '''
  Up Block
  '''
  with tf.variable_scope('up_block'):
    def skip():
      with tf.variable_scope('skip'):
        skip_connection_audio = nn_upsample(audio_lod, stride)
        skip_connection_code = tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, filters], dtype=tf.float32)
        return skip_connection_code, skip_connection_audio

    def transition():
      with tf.variable_scope('transition'):
        skip_connection_audio = nn_upsample(audio_lod, stride)

        # Shortcut
        with tf.variable_scope('shortcut'):
          shortcut = nn_upsample(inputs, stride)
          if shortcut.get_shape().aslist()[2] != filters:
            shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

        code = inputs

        # Convolution layers
        # <-- TODO: Normalization goes here 
        with tf.variable_scope('conv_0'):
          code = activation(inputs) # Pre-Activation
          code = nn_upsample(code, stride) # Upsample
          code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')
        with tf.variable_scope('conv_1'):
          # <-- TODO: Normalization goes here 
          code = activation(code)  # Pre-Activation
          code = tf.layers.conv1d(code, filters, kernel_size, strides=1, padding='same')
        
        # Add shortcut connection
        code = shortcut + code
      
        # Blend this LOD block in over time
        audio_lod_ = to_audio(code)
        audio_lod_ = lerp_clip(skip_connection_audio, audio_lod_, on_amount)
        return code, audio_lod_

    code, audio_lod = tf.cond(on_amount <= 0.0, skip, transition)
    code.set_shape([inputs.shape[0], inputs.shape[1] * stride, filters])
    audio_lod.set_shape([inputs.shape[0], inputs.shape[1] * stride, 1])

    return code, audio_lod


def down_block(inputs, audio_lod, filters, on_amount, kernel_size=9, stride=4, activation=lrelu, use_minibatch_stddev=False):
  '''
  Down Block
  '''
  with tf.variable_scope('down_block'):
    audio_lod = avg_downsample(audio_lod, stride)
    def skip():
      with tf.variable_scope('skip'):
        skip_connection_code = from_audio(audio_lod, filters)
        return skip_connection_code

    def transition():
      with tf.variable_scope('transition'):
        skip_connection_code = from_audio(audio_lod, filters)

        # Shortcut
        with tf.variable_scope('shortcut'):
          shortcut = avg_downsample(inputs, stride)
          if shortcut.get_shape().aslist()[2] != filters and use_minibatch_stddev:
            shortcut = tf.layers.conv1d(shortcut, filters + 1, kernel_size=1, strides=1, padding='same')
          elif shortcut.get_shape().aslist()[2] != filters:
            shortcut = tf.layers.conv1d(shortcut, filters, kernel_size=1, strides=1, padding='same')

        code = inputs

         # Convolution layers
        with tf.variable_scope('conv_0'):
           # <-- TODO: Normalization goes here 
          code = activation(code)  # Pre-Activation
          
          # Minibatch std deviation
          if use_minibatch_stddev:
            code = minibatch_stddev_layer(code)

          code = tf.layers.conv1d(code, inputs.get_shape().as_list()[2], kernel_size, strides=1, padding='same')
        with tf.variable_scope('conv_1'):
          # <-- TODO: Normalization goes here 
          code = activation(code)  # Pre-Activation
          code = tf.layers.conv1d(code, filters, kernel_size, strides=stride, padding='same')

        # Add shortcut connection
        code = shortcut + code

        # Blend this LOD block in over time
        return lerp_clip(skip_connection_code, code, on_amount)
      
    code = tf.cond(on_amount <= 0.0, skip, transition)
    code.set_shape([inputs.shape[0], inputs.shape[1] // stride, filters])

    return code, audio_lod


def residual_unit(    
    inputs,
    filters,
    kernel_width=24,
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
    kernel_width=24,
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
      bn = batchnorm_fn(output) if i != 0 else output
      unit_out = residual_unit(bn, filters_per_unit, kernel_width, activation=activation)
      output = tf.concat([output, unit_out], 2)

  # Resize out dimensions on request
  if out_dim is not None:
    with tf.variable_scope("1_by_1"):
      return residual_unit(output, out_dim, 1, activation=activation)
  else:
    return output


def inception_block(inputs, filters_internal=64, kernel_width=24):
  shortcut = inputs

  filter1 = tf.layers.conv1d(inputs, filters_internal, 1, padding="SAME")
  filter1 = tf.layers.conv1d(filter1, filters_internal, kernel_width // 4, padding="SAME")

  filter2 = tf.layers.conv1d(inputs, filters_internal, 1, padding="SAME")
  filter2 = tf.layers.conv1d(filter2, filters_internal, kernel_width // 2, padding="SAME")

  filter3 = tf.layers.conv1d(inputs, filters_internal, 1, padding="SAME")
  filter3 = tf.layers.conv1d(filter3, filters_internal, kernel_width, padding="SAME")

  concat = tf.concat([filter1, filter2, filter3], 2)
  output = tf.layers.conv1d(concat, inputs.shape[-1], 1, padding="SAME")

  return shortcut + output


def compress_embedding(embedding, embed_size):
  """
  Return compressed embedding for discriminator
  c: Embedded context to reduce, this should be a [batch_size x N] tensor
  embed_size: The size of the new embedding
  returns: [batch_size x embed_size] tensor
  """
  with tf.variable_scope('reduce_embed'):
    embedding = lrelu(tf.layers.dense(embedding, embed_size))
    # embedding = tf.layers.dropout(embedding)
    return embedding


def generate_context_dist_params(embedding, embed_size, train=False):
  """
  Generates the parameters for a gaussian distribution derived from a
  supplied context embedding.
    embedding - The input context from which to derive the sampling distribution parameters
    embed_size - The size of the embedding vector we are using in this program
    train - Flag to tell the generator whether to use dropout or not
    Returns - [batch_size, 2 * embed_size] sized tensor containing distribution parameters
              (mean, log(sigma)) where sigma is the diagonal entries for the covariance matrix
  """
  with tf.variable_scope('gen_context_dist'):
      params = lrelu(tf.layers.dense(embedding, 2 * embed_size))
      # params = tf.layers.dropout(params, 0.5 if train else 0)
  mean = params[:, :embed_size]
  log_sigma = params[:, embed_size:]
  return mean, log_sigma


def KL_loss(mu, log_sigma):
  with tf.name_scope("KL_divergence"):
    loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
    loss = tf.reduce_mean(loss)
    return loss


def sample_context_embeddings(embedding, embed_size, train=False):
  """
  Resamples the context embedding from a normal distribution derived 
  from the supplied context embedding.
    embedding - The input context from which to derive the sampling distribution
    embed_size - The size of output embedding vector
    train - Whether to do resample or just reduce the embedding
  """
  mean, log_sigma = generate_context_dist_params(embedding, embed_size, train)
  if train:
    epsilon = tf.truncated_normal(tf.shape(mean))
    stddev = tf.exp(log_sigma)
    c = mean + stddev * epsilon

    kl_loss = KL_loss(mean, log_sigma)
  else:
    c = mean # This is just the unmodified compressed embedding.
    kl_loss = 0

  TRAIN_COEFF_KL = 2.0
  return c, TRAIN_COEFF_KL * kl_loss


def minibatch_stddev_layer(x, group_size=4):
  with tf.variable_scope('minibatch_stddev'):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NWC]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2]])         # [GMWC] Split minibatch into G groups of size M.
    y = tf.cast(y, tf.float32)                              # [GMWC] Cast to FP32.
    y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMWC] Subtract mean over group.
    y = tf.reduce_mean(tf.square(y), axis=0)                # [MWC]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MWC]  Calc stddev over group.
    y = tf.reduce_mean(y, axis=[1,2], keepdims=True)        # [M11]  Take average over fmaps and samples.
    y = tf.cast(y, x.dtype)                                 # [M11]  Cast back to original data type.
    y = tf.tile(y, [group_size, s[1], 1])                   # [NW1]  Replicate over group and samples.
    return tf.concat([x, y], axis=2)                        # [NWC]  Append as new fmap.


"""
  Input: [None, 100]
  Output: [None, 16384, 1], kl_loss for regularizing context embedding sample distribution
"""
def WaveGANGenerator(
    z,
    lod,
    kernel_len=24,
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
    kl_loss = 0
    x_code = tf.concat([z, c], 1)
  else:
    x_code = z
    kl_loss = 0

  # FC and reshape for convolution
  # [100 + context_embedding size] -> [16, 1024]
  with tf.variable_scope('z_project'):
    x_code = tf.layers.dense(x_code, 4 * 4 * dim * 16)
    x_code = tf.reshape(x_code, [batch_size, 16, dim * 16])
    x_code = tf.nn.relu(x_code)
    # x_code = batchnorm(x_code)

  with tf.variable_scope('layer_0'):
    x_code = residual_unit(x_code, filters=dim * 16, kernel_width=kernel_len)
    x_code = batchnorm(x_code)
    x_code = residual_unit(x_code, filters=dim * 16, kernel_width=kernel_len)
    x_code = batchnorm(x_code)

  # Layer 1
  # [16, 1024] -> [64, 512]
  with tf.variable_scope('upconv_1'):
    on_amount = lod-0
    x_code, audio_lod = up_block(x_code, audio_lod=to_audio(x_code), filters=dim * 8, kernel_size=kernel_len, on_amount=on_amount)
    tf.summary.scalar('on_amount', on_amount)

  # Layer 2
  # [64, 512] -> [256, 256]
  with tf.variable_scope('upconv_2'):
    on_amount = lod-1
    x_code, audio_lod = up_block(x_code, audio_lod=audio_lod, filters=dim * 4, kernel_size=kernel_len, on_amount=on_amount)
    tf.summary.scalar('on_amount', on_amount)

  # Layer 3
  # [256, 256] -> [1024, 128]
  with tf.variable_scope('upconv_3'):
    on_amount = lod-2
    x_code, audio_lod = up_block(x_code, audio_lod=audio_lod, filters=dim * 2, kernel_size=kernel_len, on_amount=on_amount)
    tf.summary.scalar('on_amount', on_amount)

  # Layer 3
  # [1024, 128] -> [4096, 64]
  with tf.variable_scope('upconv_4'):
    on_amount = lod-3
    x_code, audio_lod = up_block(x_code, audio_lod=audio_lod, filters=dim, kernel_size=kernel_len, on_amount=on_amount)
    tf.summary.scalar('on_amount', on_amount)

  # Layer 4
  # [4096, 64] -> [16384, 1]
  with tf.variable_scope('upconv_5'):
    on_amount = lod-4
    x_code, audio_lod = up_block(x_code, audio_lod=audio_lod, filters=1, kernel_size=kernel_len, on_amount=on_amount)
    tf.summary.scalar('on_amount', on_amount)

  # Automatically update batchnorm moving averages every time G is used during training
  if train and use_batchnorm:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # if len(update_ops) != 10:
    #   raise Exception('Other update ops found in graph')
    with tf.control_dependencies(update_ops):
      audio_lod = tf.identity(audio_lod)

  return audio_lod, kl_loss


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


def encode_audio_stage_1(x,
    lod,
    kernel_len=24,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    embedding_dim=128):
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  with tf.variable_scope('audio_encode_stage_1'):
    # Layer 0
    # [16384, 1] -> [4096, 64]
    tf.summary.audio('input_audio', x, 16000, max_outputs=1)
    output = x
    with tf.variable_scope('downconv_0'):
      on_amount = lod-4
      output, audio_lod = down_block(output, audio_lod=x, filters=dim, kernel_size=kernel_len, on_amount=on_amount)
      tf.summary.audio('audio_downsample', nn_upsample(audio_lod), 16000, max_outputs=1)
      tf.summary.scalar('on_amount', on_amount)
      

    # Layer 1
    # [4096, 64] -> [1024, 128]
    with tf.variable_scope('downconv_1'):
      on_amount = lod-3
      output, audio_lod = down_block(output, audio_lod=audio_lod, filters=dim * 2, kernel_size=kernel_len, on_amount=on_amount)
      tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(audio_lod)), 16000, max_outputs=1)
      tf.summary.scalar('on_amount', on_amount)

    return output, audio_lod


def encode_audio_stage_2(x,
    audio_lod,
    lod,
    kernel_len=24,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    embedding_dim=128):
  if use_batchnorm:
    batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
  else:
    batchnorm = lambda x: x

  if phaseshuffle_rad > 0:
    phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
  else:
    phaseshuffle = lambda x: x

  with tf.variable_scope('audio_encode_stage_2'):
    # Layer 2
    # [1024, 128] -> [256, 256]
    output = x
    with tf.variable_scope('downconv_2'):
      on_amount = lod-2
      output, audio_lod = down_block(output, audio_lod=audio_lod, filters=dim * 4, kernel_size=kernel_len, on_amount=on_amount)
      tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(nn_upsample(audio_lod))), 16000, max_outputs=1)
      tf.summary.scalar('on_amount', on_amount)

    # Layer 3
    # [256, 256] -> [64, 512]
    with tf.variable_scope('downconv_3'):
      on_amount = lod-1
      output, audio_lod = down_block(output, audio_lod=audio_lod, filters=dim * 8, kernel_size=kernel_len, on_amount=on_amount)
      tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(nn_upsample(nn_upsample(audio_lod)))), 16000, max_outputs=1)
      tf.summary.scalar('on_amount', on_amount)

    # Layer 4
    # [64, 512] -> [16, 1024]
    with tf.variable_scope('downconv_4'):
      on_amount = lod-0
      output, audio_lod = down_block(output, audio_lod=audio_lod, filters=dim * 16, kernel_size=kernel_len, use_minibatch_stddev=False, on_amount=on_amount)
      tf.summary.audio('audio_downsample', nn_upsample(nn_upsample(nn_upsample(nn_upsample(nn_upsample(audio_lod))))), 16000, max_outputs=1)
      tf.summary.scalar('on_amount', on_amount)

      # Flatten
    # [16, 1024] -> [16384]
    batch_size = tf.shape(x)[0]
    output = tf.reshape(output, [batch_size, dim * 16 * 16])

    return output, audio_lod


"""
  Input: [None, 16384, 1]
  Output: [None] (linear output)
"""
def WaveGANDiscriminator(
    x,
    lod,
    kernel_len=24,
    dim=64,
    use_batchnorm=False,
    phaseshuffle_rad=0,
    context_embedding=None,
    embedding_dim=128,
    use_extra_uncond_output=False):

  stage_1, audio_lod_stage_1 = encode_audio_stage_1(x, lod, kernel_len, dim, use_batchnorm, phaseshuffle_rad, embedding_dim)

  with tf.variable_scope('unconditional'):
    uncond_out, _ = encode_audio_stage_2(stage_1, audio_lod_stage_1, lod, kernel_len, dim, use_batchnorm, phaseshuffle_rad, embedding_dim)
  
  if (context_embedding is not None):
    with tf.variable_scope('conditional'):
      cond_out, _ = encode_audio_stage_2(stage_1, audio_lod_stage_1, lod, kernel_len, dim, use_batchnorm, phaseshuffle_rad, embedding_dim)

      # Concat context embeddings
      # [16384] -> [16384 + embedding_dim]
      c = compress_embedding(context_embedding, embedding_dim)
      cond_out = tf.concat([cond_out, c], 1)

      # FC
      # [16384 + embedding_dim] -> [1024]
      with tf.variable_scope('FC'):
        cond_out = tf.layers.dense(cond_out, dim * 16)
        cond_out = lrelu(cond_out)
        cond_out = tf.layers.dropout(cond_out)
        output = cond_out
  else:
    output = uncond_out

  # Connect to single logit
  # [16384] -> [1]
  with tf.variable_scope('output'):
    output = tf.layers.dense(output, 1)
    if (use_extra_uncond_output) and (context_embedding is not None):
      uncond_out = tf.layers.dense(uncond_out, 1)
      return [output, uncond_out]
    else:
      return [output]

  # Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training
