import tensorflow as tf
import collections
from .cells import build_rnn_layers, create_attention_mechanism
from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import MultiRNNCell
import numpy as np

from tensorflow.python.layers.core import Dense


class EncoderData(collections.namedtuple("EncoderData", ("outputs", "final_state"))):
    pass


class DualSeq2SeqEncoder(object):

    def __init__(self,
                 audio_data,
                 video_data,
                 mode,
                 hparams,
                 audio_num_units_per_layer,
                 video_num_units_per_layer,
                 ):

        self._audio_data = audio_data
        self._video_data = video_data
        self._mode = mode
        self._hparams = hparams
        self._audio_num_units_per_layer = audio_num_units_per_layer
        self._video_num_units_per_layer = video_num_units_per_layer

        self._init_data()
        self._init_encoder()

    def _init_data(self):
        self._audio_inputs = self._audio_data.inputs
        self._video_inputs = self._video_data.inputs
        self._audio_inputs_len = self._audio_data.inputs_length
        self._video_inputs_len = self._video_data.inputs_length

        # self._labels = self._data.labels
        # self._labels_len = self._data.labels_length

        if self._hparams.batch_normalisation is True:
            self._audio_inputs = tf.layers.batch_normalization(
                inputs=self._audio_inputs,
                axis=-1,
                training=(self._mode == 'train'),
                fused=True,
            )
            self._video_inputs = tf.layers.batch_normalization(
                inputs=self._video_inputs,
                axis=-1,
                training=(self._mode == 'train'),
                fused=True,
            )

        if self._hparams.instance_normalisation is True:
            from tensorflow.contrib.layers import instance_norm
            self._audio_inputs = instance_norm(
                inputs=self._audio_inputs,
            )
            self._video_inputs = instance_norm(
                inputs=self._video_inputs,
            )

    def _init_encoder(self):
        if self._hparams.encoder_type not in ('unidirectional','bidirectional',):
            raise Exception('Allowed encoder types: `unidirectional`, `bidirectional`')

    def _maybe_add_dense_layers(self):
        r"""
        Optionally passes self._input through several Fully Connected (Dense) layers
        with the configuration defined by the self._input_dense_layers tuple

        Returns
        -------
        The output of the network of Dense layers
        """
        audio_layer_inputs = self._audio_inputs
        video_layer_inputs = self._video_inputs
        
        if self._hparams.input_dense_layers[0] > 0:

            fc_a = [Dense(units,
                        activation=tf.nn.selu,
                        use_bias=False,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
                  for units in self._hparams.input_dense_layers]
            fc_v = [Dense(units,
                        activation=tf.nn.selu,
                        use_bias=False,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001))
                  for units in self._hparams.input_dense_layers]

            for layer in fc_a:
                audio_layer_inputs = layer(audio_layer_inputs)
            for layer in fc_v:
                video_layer_inputs = layer(video_layer_inputs)
        else:
            pass
        return audio_layer_inputs, video_layer_inputs
    
    def get_audio_data(self):
        pass
    
    def get_video_data(self):
        pass

class DualAttentiveEncoder(DualSeq2SeqEncoder):

    def __init__(self,
                 audio_data,
                 video_data,
                 mode,
                 hparams,
                 audio_num_units_per_layer,
                 video_num_units_per_layer,
                 ):
        r"""
        Implements https://arxiv.org/abs/1809.01728
        (V->A) + (A->V) cross modal attention 
        """

        super(DualAttentiveEncoder, self).__init__(
            audio_data,
            video_data,
            mode,
            hparams,
            audio_num_units_per_layer,
            video_num_units_per_layer
        )

    def _init_encoder(self):
        with tf.variable_scope("Encoder") as scope:

            audio_encoder_inputs, video_encoder_inputs = self._maybe_add_dense_layers()

            if self._hparams.encoder_type == 'unidirectional':
                with tf.variable_scope("audio_rnn") as scope:
                    self._audio_encoder_cells = build_rnn_layers(
                        cell_type=self._hparams.cell_type,
                        num_units_per_layer=self._audio_num_units_per_layer,
                        use_dropout=self._hparams.use_dropout,
                        dropout_probability=self._hparams.dropout_probability,
                        mode=self._mode,
                        as_list=True,
                        dtype=self._hparams.dtype)
    
                    self._audio_encoder_inter_outputs, self._audio_encoder_inter_state = tf.nn.dynamic_rnn(
                        cell=MultiRNNCell(self._audio_encoder_cells[0:-1]),
                        inputs=audio_encoder_inputs,
                        sequence_length=self._audio_inputs_len,
                        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        scope=scope,
                        )

                with tf.variable_scope("video_rnn") as scope:
                    self._video_encoder_cells = build_rnn_layers(
                        cell_type=self._hparams.cell_type,
                        num_units_per_layer=self._video_num_units_per_layer,
                        use_dropout=self._hparams.use_dropout,
                        dropout_probability=self._hparams.dropout_probability,
                        mode=self._mode,
                        as_list=True,
                        dtype=self._hparams.dtype)
    
                    self._video_encoder_inter_outputs, self._video_encoder_inter_state = tf.nn.dynamic_rnn(
                        cell=MultiRNNCell(self._video_encoder_cells[0:-1]),
                        inputs=video_encoder_inputs,
                        sequence_length=self._video_inputs_len,
                        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        scope=scope,
                        )
                
                with tf.variable_scope("audio_mech") as scope:
                    audio_attention_mechanism, audio_output_attention = create_attention_mechanism(
                        attention_type=self._hparams.attention_type[0][0],
                        num_units=self._audio_num_units_per_layer[-1],
                        memory=self._video_encoder_inter_outputs,
                        memory_sequence_length=self._video_inputs_len,
                        mode=self._mode,
                        dtype=self._hparams.dtype
                    )
 
                with tf.variable_scope("video_mech") as scope:
                    video_attention_mechanism, video_output_attention = create_attention_mechanism(
                        attention_type=self._hparams.attention_type[0][0],
                        num_units=self._video_num_units_per_layer[-1],
                        memory=self._audio_encoder_inter_outputs,
                        memory_sequence_length=self._audio_inputs_len,
                        mode=self._mode,
                        dtype=self._hparams.dtype
                    )               

                with tf.variable_scope("audio_wrap") as scope:
                    audio_attention_cells = seq2seq.AttentionWrapper(
                        cell=self._audio_encoder_cells[-1],
                        attention_mechanism=audio_attention_mechanism,
                        attention_layer_size=self._hparams.decoder_units_per_layer[-1]/2,
                        alignment_history=self._hparams.write_attention_alignment,
                        output_attention=audio_output_attention,
                    )

                with tf.variable_scope("video_wrap") as scope:
                    video_attention_cells = seq2seq.AttentionWrapper(
                        cell=self._video_encoder_cells[-1],
                        attention_mechanism=video_attention_mechanism,
                        attention_layer_size=self._hparams.decoder_units_per_layer[-1]/2,
                        alignment_history=self._hparams.write_attention_alignment,
                        output_attention=video_output_attention,
                    )

                self._audio_encoder_cells[-1] = audio_attention_cells
                self._video_encoder_cells[-1] = video_attention_cells

                with tf.variable_scope("audio_last") as scope:
                    self._audio_encoder_outputs, self._audio_encoder_final_state = tf.nn.dynamic_rnn(
                        cell=self._audio_encoder_cells[-1],
                        inputs=self._audio_encoder_inter_outputs,
                        sequence_length=self._audio_inputs_len,
                        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        scope=scope,
                        )

                with tf.variable_scope("video_last") as scope:
                    self._video_encoder_outputs, self._video_encoder_final_state = tf.nn.dynamic_rnn(
                        cell=self._video_encoder_cells[-1],
                        inputs=self._video_encoder_inter_outputs,
                        sequence_length=self._video_inputs_len,
                        parallel_iterations=self._hparams.batch_size[0 if self._mode == 'train' else 1],
                        swap_memory=False,
                        dtype=self._hparams.dtype,
                        scope=scope,
                        )

                if self._hparams.write_attention_alignment is True:
                    self.attention_summary = self._create_attention_alignments_summary(self._video_encoder_final_state)
                
                self._tr_a = tf.identity(self._audio_encoder_final_state.alignment_history.stack())
                self._tr_v = tf.identity(self._video_encoder_final_state.alignment_history.stack())
                self._ga_a = tf.identity(self._tr_a)
                self._ga_v = tf.identity(self._tr_v)

    def get_transpose_loss(self):
        tens_v = tf.transpose(self._tr_v, perm=[2,1,0])
        tens_a = self._tr_a
        len_a = tf.identity(self._audio_inputs_len)
        len_v = tf.identity(self._video_inputs_len)
        
        l1_loss = 0
        batch_count = 0
        for i in range(self._hparams.batch_size[0 if self._mode == 'train' else 1]):
            try:
                tens_v_b = tens_v[0:len_a[i]-1, i, 0:len_v[i]-1]
                tens_a_b = tens_a[0:len_a[i]-1, i ,0:len_v[i]-1]
                
                l1_loss += tf.reduce_mean(tf.abs(tens_v_b - tens_a_b))

                batch_count += 1

            except tf.errors.InvalidArgumentError as e:
                pass
        
        return l1_loss / batch_count

    def get_guided_attention_loss(self):
        resize_att_a2v_array = tf.TensorArray(dtype=self._hparams.dtype,
                size=self._hparams.batch_size[0 if self._mode == 'train' else 1]
                )
        resize_att_v2a_array = tf.TensorArray(dtype=self._hparams.dtype,
                size=self._hparams.batch_size[0 if self._mode == 'train' else 1]
                )
        tens_v = tf.transpose(self._ga_v, perm=[2,1,0])
        tens_a = self._ga_a
        
        len_a = tf.identity(self._audio_inputs_len)
        max_len_a = tf.reduce_max(self._audio_inputs_len)
        len_v = tf.identity(self._video_inputs_len)
        max_len_v = tf.reduce_max(self._video_inputs_len)
        
        batch_count = 0
        for i in range(self._hparams.batch_size[0 if self._mode == 'train' else 1]):
            try:
                tens_v_b = tens_v[0:len_a[i]-1, i, 0:len_v[i]-1]
                tens_a_b = tens_a[0:len_a[i]-1, i ,0:len_v[i]-1]
                tens_v_b = tens_v_b[:,:,tf.newaxis]
                tens_a_b = tens_a_b[:,:,tf.newaxis]

                resize_att_a2v = tf.reshape(
                        tf.image.resize_images(
                            tens_v_b,
                            [max_len_a, max_len_a]
                            ),
                        [max_len_a, -1]
                        )
    
                resize_att_v2a = tf.reshape(
                        tf.image.resize_images(
                            tens_a_b,
                            [max_len_a, max_len_a]
                            ),
                        [max_len_a, -1]
                        )
                resize_att_a2v_array = resize_att_a2v_array.write(i, resize_att_a2v)
                resize_att_v2a_array = resize_att_v2a_array.write(i, resize_att_v2a)
                
                batch_count += 1

            except tf.errors.InvalidArgumentError as e:
                pass
        
        att_v2a = resize_att_v2a_array.stack()
        att_a2v = resize_att_a2v_array.stack()

        len_eye = 120
        eye = tf.eye(
                num_rows = len_eye,
                batch_shape = [batch_count],
                dtype = self._hparams.dtype
                )
    
        gauss_kernel = self.gaussian_kernel(size = 10, mean = 0.0, std = 10.0)
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
        eye = eye[:, :, :, tf.newaxis]    
        N_filter = 5
        for n in range(N_filter):
            eye = tf.nn.conv2d(eye, gauss_kernel, strides=[1,1,1,1], padding="SAME")

        eye_resize = tf.image.resize_images(
                eye[:, 40:80, 40:80, :],
                [max_len_a, max_len_a],
                )
        
        weight_map = tf.reshape(
                tf.ones_like(eye_resize) - eye_resize/tf.reduce_max(eye_resize),
                [batch_count, max_len_a, max_len_a],
                )
    
        ga_loss = tf.reduce_mean(tf.abs(att_v2a * weight_map)) + tf.reduce_mean(tf.abs(att_a2v * weight_map))

        return ga_loss
    
    def gaussian_kernel(self,
            size: int,
            mean: float,
            std: float,
            ):

        d = tf.distributions.Normal(mean, std)
        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
        gauss_kernel = tf.einsum('i,j->ij',
                vals,
                vals,
                )

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def _create_attention_alignments_summary(self, states):
        r"""
        Generates the alignment images, useful for visualisation/debugging purposes
        """
        attention_alignment = states.alignment_history.stack()

        attention_images = tf.expand_dims(tf.transpose(attention_alignment, [1, 2, 0]), -1)

        # attention_images_scaled = tf.image.resize_images(1-attention_images, (256,128))
        attention_images_scaled = 1 - attention_images

        attention_summary = tf.summary.image("attention_images_cm_video", attention_images_scaled,
                                             max_outputs=self._hparams.batch_size[1])

        return attention_summary


    def get_audio_data(self):

        return EncoderData(
            outputs=self._audio_encoder_outputs,
            final_state=(self._audio_encoder_inter_state, self._audio_encoder_final_state.cell_state),
        )
    
    def get_video_data(self):

        return EncoderData(
            outputs=self._video_encoder_outputs,
            final_state=(self._video_encoder_inter_state, self._video_encoder_final_state.cell_state),
        )

