import tensorflow as tf

class HistoryBuffer(object):
    def __init__(self, clip_length, max_size, batch_size):
        """
        Initialize the class's state.
        :param clip_length: Length of audio clips that will be placed in the buffer
        :param max_size: Maximum number of audio clips that can be stored in the history buffer. Note: This should a multiple of the batch size
        :param batch_size: Batch size used to train GAN.
        """
        self.history_buffer = tf.get_variable('history_buffer', initializer=tf.zeros([max_size, clip_length * 2 + 1024, 1]), trainable=False, dtype=tf.float32)
        #self.cond_history_buffer = tf.get_variable('cond_history_buffer', initializer=tf.zeros([max_size, 1024]), trainable=False, dtype=tf.float32)
        self.current_size = tf.get_variable('cur_history_buffer_size', initializer=0, trainable=False, dtype=tf.int32)
        self.max_size = max_size
        self.clip_length = clip_length
        self.batch_size = batch_size

    def add_to_history_buffer(self, g_audio_clips, r_audio_clips, cond_embeds, nb_to_add=None):
        """
        To be called during training of GAN. By default add batch_size // 2 audio clips to the history buffer each
        time the generator generates a new batch of audio clips.
        :param g_audio_clips: Array of generated audio clips (usually a batch) to be added to the history buffer.
        :param r_audio_clips: Array of real audio clips (usually a batch) to be added to the history buffer.
        :param nb_to_add: The number of audio clips from `g_audio_clips` to add to the history buffer
                          (batch_size / 2 by default).
        :return: Ops to execute (use tf.control_dependencies)
        """
        if not nb_to_add:
            nb_to_add = self.batch_size // 2

        # Reshape to match packed history
        cond_embeds = tf.expand_dims(cond_embeds, axis=2)

        def not_full_fn():
            slice_start = self.current_size
            slice_end   = slice_start + nb_to_add
            slice_assign_op = self.history_buffer[slice_start:slice_end, 0:self.clip_length].assign(g_audio_clips[:nb_to_add])
            with tf.control_dependencies([slice_assign_op]):
                slice_assign_op = self.history_buffer[slice_start:slice_end, self.clip_length:self.clip_length*2].assign(r_audio_clips[:nb_to_add])
                with tf.control_dependencies([slice_assign_op]):
                    slice_assign_op = self.history_buffer[slice_start:slice_end, self.clip_length*2:].assign(cond_embeds[:nb_to_add])
                    with tf.control_dependencies([slice_assign_op]):
                        cur_size_incr_op = self.current_size.assign_add(nb_to_add)
                        with tf.control_dependencies([cur_size_incr_op]):
                            return tf.identity(slice_assign_op)
        def full_fn():
            slice_assign_op = self.history_buffer[:nb_to_add, 0:self.clip_length].assign(g_audio_clips[:nb_to_add])
            with tf.control_dependencies([slice_assign_op]):
                slice_assign_op = self.history_buffer[:nb_to_add, self.clip_length:self.clip_length*2].assign(r_audio_clips[:nb_to_add])
                with tf.control_dependencies([slice_assign_op]):
                    slice_assign_op = self.history_buffer[:nb_to_add, self.clip_length*2:].assign(cond_embeds[:nb_to_add])
                    return slice_assign_op

        dep_op = tf.cond(self.current_size < self.max_size, not_full_fn, full_fn)

        with tf.control_dependencies([dep_op]):
            shuffle_op = tf.random_shuffle(self.history_buffer[:self.current_size])
            shuffle_assign_op = self.history_buffer[:self.current_size].assign(shuffle_op)
            
            return shuffle_assign_op

    def get_from_history_buffer(self, nb_to_get=None):
        """
        Get a random sample of audio clips from the history buffer.
        :param nb_to_get: Number of audio clips to get from the history buffer (batch_size / 2 by default).
        :return: A random sample of `nb_to_get` audio clips from the history buffer, or an tensor if the
                 history buffer is empty and `nb_to_get` conditional text embeddings for each audio clip.
        """
        if not nb_to_get:
            nb_to_get = self.batch_size // 2

        nb_to_get = tf.minimum(nb_to_get, self.current_size)
        return (self.history_buffer[:nb_to_get, 0:self.clip_length],
                self.history_buffer[:nb_to_get, self.clip_length:self.clip_length*2],
                tf.squeeze(self.history_buffer[:nb_to_get, self.clip_length*2:]))