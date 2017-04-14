
import tensorflow as tf

def make_rnn_cell(rnn_layer_sizes,
                  nlayers,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.nn.rnn_cell.BasicLSTMCell,
                  state_is_tuple=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
        RNN.
    dropout_keep_prob: The float probability to keep the output of any given
        sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells, layer = [], 0
  for num_units in rnn_layer_sizes:
    if layer == nlayers:
        break
    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)
    layer += 1

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell


# modified from tensorflow-wavenet
def _causal_conv(value,
        filter_width,
        out_channels=16,
        dilation=1,
        name='causal_conv'):
    """
    Args:
        value has shape [batch, in_height, in_width, in_channels]
        filter_ has shape [filter_height, filter_width, in_channels, out_channels]
    """
    in_channels = value.get_shape()[-1]
    with tf.variable_scope(name):
        weights_filter = tf.get_variable('w', [1, filter_width, in_channels, out_channels], 
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        padding = [[0,0], [0,0], [(filter_width - 1) * dilation, 0], [0,0]]
        padded = tf.pad(value, padding)
        if dilation > 1:
            # instead of explicitly using time_to_batch, batch_to_time, just use atrous_conv2d
            conv = tf.nn.atrous_conv2d(padded, weights_filter, dilation, 
                                padding='VALID')
        else:
            conv = tf.nn.conv2d(padded, weights_filter, strides=[1,1,1,1], padding='VALID')
        return conv

def _create_dilation_layer(input_batch, layer_idx, dilation, global_condition_batch, 
        filter_width, residual_channels, is_training):
    # Each layer is wrapped in a residual block.
    # TODO layer norms, gated activation unit instead of ReLU
    # concatenating z with every word embedding of the decoder input.

    # ReLU 1x1
    relu1 = tf.nn.relu(input_batch, name='relu1_layer{}'.format(layer_idx))
    conv1 = _causal_conv(relu1, 
            filter_width=1, 
            out_channels=residual_channels,
            dilation=1,
            name='conv1d_1_layer{}'.format(layer_idx))
    conv1 = conv1 + _causal_conv(global_condition_batch,
                        filter_width=1,
                        out_channels=residual_channels,
                        dilation=1,
                        name='gc_filter_layer{}'.format(layer_idx))

    # ReLU 1xk
    relu2 = tf.nn.relu(conv1, name='relu2_layer{}'.format(layer_idx))
    conv2 = _causal_conv(relu2, 
            filter_width=filter_width,
            out_channels=residual_channels,
            dilation=dilation,
            name='dilated_conv_layer{}'.format(layer_idx))

    # ReLU 1x1 
    relu3 = tf.nn.relu(conv2, name='relu3_layer{}'.format(layer_idx))
    conv3 = _causal_conv(relu3, 
            filter_width=1,
            out_channels=2*residual_channels,
            dilation=1,
            name='conv1d_2_layer{}'.format(layer_idx))
   
    # "dense output"
    return input_batch + conv3

# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/model.py
# https://github.com/tensorflow/magenta/pull/537/files
def dilated_cnn(inputs, dilations,
        residual_channels=512,
        output_channels=128,
        global_condition=None, 
        filter_width=3,
        dropout_keep_prob=1.0,
        mode='train'):
    '''
    Args
        inputs: A tensor of shape [batch_size, num_steps, input_size] of floats32s.
    '''
    final_state = []
    input_shape = inputs.get_shape().as_list()
    batch_size = input_shape[0]
    input_size = input_shape[2]

    # TODO for completeness, include case where global_condition = None
    # [batch size, 1, 1, channels]
    global_condition_channels = global_condition.get_shape().as_list()[1]
    global_condition_batch = tf.reshape(
            global_condition, [-1, 1, 1, global_condition_channels])

    # Pre-process the input with a regular convolution (create causal layer)
    # Note: num_steps = in_width, input_size = in_channels. That makes sense.
    inputs_batch = tf.reshape(inputs, [batch_size,1,-1,input_size])
    current_layer = _causal_conv(
            inputs_batch,
            filter_width=filter_width,
            out_channels=2*residual_channels,
            dilation=1,
            name='causal_layer')

    # expect shape to be [batch, 1, out_width, out_channels]
    # out_width = in_width - (filter_width - 1) * dilation (I think.) -- this comes from the pyramid structure.

    # TODO Use skip connections, as per Wavenet
    for i, dilation in enumerate(dilations):
        with tf.variable_scope('layer{}'.format(i)):
            '''
            # This padding is for generation. Not sure what it does exactly, but OK.
            pad = initial_state[:,dlt_sum[i]*residual_channels:dlt_sum[i+1]*residual_channels]
            pad = tf.reshape(pad,[batch_size,1,dilation,residual_channels])
            _h = current_layer
            h = tf.concat(2,[pad,current_layer]) # TODO correct axis?
            _fs = tf.reshape(h[:,-dilation:,:,:],[batch_size,dilation*residual_channels])
            final_state.append(_fs)
            '''
            h = current_layer

            layer_output = _create_dilation_layer(
                    h, i, dilation, global_condition_batch, 
                    filter_width, residual_channels, 
                    (mode == 'train'))
            current_layer = layer_output

    # Post-process to be dense
    # Unlike Wavenet paper, no final softmax -- using softmax_cross_entropy_with_logits in train
    dense_output = _causal_conv(layer_output,
            filter_width=1,
            out_channels=output_channels,
            dilation=1,
            name='last_weights')
    dense_output = tf.nn.dropout(dense_output, dropout_keep_prob)
    # Back to 1D
    outputs = tf.reshape(dense_output, [batch_size, -1, output_channels])

    return outputs, final_state

