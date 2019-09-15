from tensorflow.contrib.rnn import MultiRNNCell, DeviceWrapper, DropoutWrapper, \
    LSTMCell, GRUCell, LSTMBlockCell, UGRNNCell, NASCell, GRUBlockCellV2, \
    HighwayWrapper, ResidualWrapper
import tensorflow as tf
from tensorflow.contrib import seq2seq


def _build_single_cell(cell_type, num_units, use_dropout, mode, dropout_probability, dtype, device=None):
    r"""

    :param num_units: `int`
    :return:
    """
    if cell_type == 'lstm':
        cells = LSTMCell(num_units=num_units,
                         use_peepholes=False,
                         cell_clip=1.0,
                         initializer=tf.variance_scaling_initializer(),
                         dtype=dtype)
    elif cell_type == 'gru':
        cells = GRUCell(num_units=num_units,
                        kernel_initializer=tf.variance_scaling_initializer(),
                        bias_initializer=tf.variance_scaling_initializer(),
                        dtype=dtype
                        )
    elif cell_type == 'ugrnn':
        cells = UGRNNCell(num_units)
    elif cell_type == 'lstm_block':
        cells = LSTMBlockCell(num_units=num_units,
                              use_peephole=True,
                              cell_clip=None)
    elif cell_type == 'gru_block':
        cells = GRUBlockCellV2(num_units=num_units)
    elif cell_type == 'nas':
        cells = NASCell(num_units=num_units)
    elif cell_type == 'lstm_masked':
        from tensorflow.contrib.model_pruning import MaskedLSTMCell
        cells = MaskedLSTMCell(num_units=num_units)
    else:
        raise Exception('cell type not supported: {}'.format(cell_type))

    if use_dropout is True and mode == 'train':
        cells = DropoutWrapper(cells,
                               input_keep_prob=dropout_probability[0],
                               state_keep_prob=dropout_probability[1],
                               output_keep_prob=dropout_probability[2],
                               variational_recurrent=False,
                               dtype=dtype,
                               # input_size=self._inputs.get_shape()[1:],
                               )
    if device is not None:
        cells = DeviceWrapper(cells, device=device)

    return cells


def build_rnn_layers(
        cell_type,
        num_units_per_layer,
        use_dropout,
        dropout_probability,
        mode,
        dtype,
        residual_connections=False,
        highway_connections=False,
        as_list=False
    ):

    cell_list = []
    for layer, units in enumerate(num_units_per_layer):

        cell = _build_single_cell(
            cell_type=cell_type,
            num_units=units,
            use_dropout=use_dropout,
            dropout_probability=dropout_probability,
            mode=mode,
            dtype=dtype,
        )

        if highway_connections is True and layer > 0:
            cell = HighwayWrapper(cell)
        elif residual_connections is True and layer > 0:
            cell = ResidualWrapper(cell)

        cell_list.append(cell)

    if len(cell_list) == 1:
        return cell_list[0]
    else:
        if as_list is False:
            return MultiRNNCell(cell_list)
        else:
            return cell_list


def create_attention_mechanism(
        attention_type,
        num_units,
        memory,
        memory_sequence_length,
        mode,
        dtype):

    if attention_type == 'bahdanau':
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=False,
            dtype=dtype,
        )
        output_attention = False
    elif attention_type == 'normed_bahdanau':
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=True,
            dtype=dtype,
        )
        output_attention = False
    elif attention_type == 'normed_monotonic_bahdanau':
        attention_mechanism = seq2seq.BahdanauMonotonicAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            normalize=True,
            score_bias_init=-2.0,
            sigmoid_noise=1.0 if mode == 'train' else 0.0,
            mode='hard' if mode != 'train' else 'parallel',
            dtype=dtype,
        )
        output_attention = False
    elif attention_type == 'luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            dtype=dtype,
        )
        output_attention = True
    elif attention_type == 'scaled_luong':
        attention_mechanism = seq2seq.LuongAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=True,
            dtype=dtype,
        )
        output_attention = True
    elif attention_type == 'scaled_monotonic_luong':
        attention_mechanism = seq2seq.LuongMonotonicAttention(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            scale=True,
            score_bias_init=-2.0,
            sigmoid_noise=1.0 if mode == 'train' else 0.0,
            mode='hard' if mode != 'train' else 'parallel',
            dtype=dtype,
        )
        output_attention = True
    else:
        raise Exception('unknown attention mechanism')

    return attention_mechanism, output_attention

