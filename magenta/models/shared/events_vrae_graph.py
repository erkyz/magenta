# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provides function to build an event sequence RNN model's graph."""

# internal imports
import tensorflow as tf
import magenta

TEMP_LATENT_SIZE = 64

def make_rnn_cell(rnn_layer_sizes,
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
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell


def build_graph(mode, config, sequence_example_file_paths=None):
  """Builds the TensorFlow graph.

  Args:
    mode: 'train', 'eval', or 'generate'. Only mode related ops are added to
        the graph.
    config: An EventSequenceRnnConfig containing the encoder/decoder and HParams
        to use.
    sequence_example_file_paths: A list of paths to TFRecord files containing
        tf.train.SequenceExample protos. Only needed for training and
        evaluation. May be a sharded file of the form.

  Returns:
    A tf.Graph instance which contains the TF ops.

  Raises:
    ValueError: If mode is not 'train', 'eval', or 'generate'.
  """
  if mode not in ('train', 'eval', 'generate'):
    raise ValueError("The mode parameter must be 'train', 'eval', "
                     "or 'generate'. The mode parameter was: %s" % mode)

  hparams = config.hparams
  encoder_decoder = config.encoder_decoder

  tf.logging.info('hparams = %s', hparams.values())

  input_size = encoder_decoder.input_size
  num_classes = encoder_decoder.num_classes
  no_event_label = encoder_decoder.default_event_label

  with tf.Graph().as_default() as graph:
    inputs, labels, lengths, = None, None, None
    state_is_tuple = True

    if mode == 'train' or mode == 'eval':
      inputs, labels, lengths = magenta.common.get_padded_batch(
          sequence_example_file_paths, hparams.batch_size, input_size)

    elif mode == 'generate':
      inputs = tf.placeholder(tf.float32, [hparams.batch_size, None,
                                           input_size])
      # If state_is_tuple is True, the output RNN cell state will be a tuple
      # instead of a tensor. During training and evaluation this improves
      # performance. However, during generation, the RNN cell state is fed
      # back into the graph with a feed dict. Feed dicts require passed in
      # values to be tensors and not tuples, so state_is_tuple is set to False.
      state_is_tuple = False


    with tf.variable_scope('encoder'):
        encoder_cell = make_rnn_cell(hparams.rnn_layer_sizes,
                         dropout_keep_prob=hparams.dropout_keep_prob,
                         attn_length=0,
                         state_is_tuple=state_is_tuple)

        encoder_initial_state = encoder_cell.zero_state(hparams.batch_size, tf.float32)

        # TODO alter so it doesn't output stuff?
        _, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, inputs, initial_state=encoder_initial_state, parallel_iterations=1,
            swap_memory=True)

        # TODO multi-layer
        # We only look at the cell state of the last layer of the encoder LSTM
        # need a z for each input in the batch.
        encoder_final_cell_state = encoder_final_state[-1][0]
        z_mu = tf.contrib.layers.fully_connected(encoder_final_cell_state, 
                num_outputs=TEMP_LATENT_SIZE, activation_fn=None, trainable=True)
        z_logvar = tf.contrib.layers.fully_connected(encoder_final_cell_state, 
                num_outputs=TEMP_LATENT_SIZE, activation_fn=None, trainable=True)

    # decoder
    with tf.variable_scope('decoder'):
        decoder_cell = make_rnn_cell(hparams.rnn_layer_sizes,
                             dropout_keep_prob=hparams.dropout_keep_prob,
                             attn_length=hparams.attn_length,
                             state_is_tuple=state_is_tuple)

        decoder_h0 = []
        
        # sample z using reparameterization trick, do this batch_size times.
        # TODO efficiency
        # TODO alter depending on state_is_tuple
        for c, h in decoder_cell.zero_state(hparams.batch_size, dtype=tf.float32):
            epsilon = tf.random_normal(tf.shape(z_logvar), 0, 1, dtype=tf.float32)
            z = z_mu + tf.mul(tf.sqrt(tf.exp(z_logvar)), epsilon)
            h_state = tf.contrib.layers.fully_connected(z, 
                    num_outputs=hparams.rnn_layer_sizes[1], 
                    trainable=True)
            decoder_h0.append(tf.nn.rnn_cell.LSTMStateTuple(c, h_state))

        # Note: tf.reverse syntax unique to TF 0.12
        outputs, final_state = tf.nn.dynamic_rnn(
            decoder_cell, tf.reverse(inputs, [False, True, False]), initial_state=tuple(decoder_h0),
            parallel_iterations=1, swap_memory=True)

    outputs_flat = tf.reshape(outputs, [-1, decoder_cell.output_size])
    logits_flat = tf.contrib.layers.linear(outputs_flat, num_classes)

    if mode == 'train' or mode == 'eval':
      '''
      if hparams.skip_first_n_losses:
        logits = tf.reshape(logits_flat, [hparams.batch_size, -1, num_classes])
        logits = logits[:, hparams.skip_first_n_losses:, :]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        labels = labels[:, hparams.skip_first_n_losses:]
      '''

      labels_flat = tf.reshape(labels, [-1])
      mask_flat = tf.reshape(tf.sequence_mask(lengths, dtype=tf.float32), [-1])
      num_logits = tf.to_float(tf.reduce_sum(lengths))

      with tf.control_dependencies(
          [tf.Assert(tf.greater(num_logits, 0.), [num_logits])]):
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels_flat, logits=logits_flat)

      # "latent loss" -- KL divergence from N(0,I) 
      # TODO idk how this was derived exactly. what happened to det and trace?
      kld = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), 1)
      # "reconstruction loss": 
      reconstruction_loss = tf.reduce_sum(mask_flat * softmax_cross_entropy) / num_logits
      # VR lower bound -- cross entropy is equivalent to negative log likelihood.
      # average over batch
      loss = tf.reduce_mean(reconstruction_loss + kld)
      perplexity = (tf.reduce_sum(mask_flat * tf.exp(softmax_cross_entropy)) /
                    num_logits)

      correct_predictions = tf.to_float(
          tf.nn.in_top_k(logits_flat, labels_flat, 1)) * mask_flat
      accuracy = tf.reduce_sum(correct_predictions) / num_logits * 100

      event_positions = (
          tf.to_float(tf.not_equal(labels_flat, no_event_label)) * mask_flat)
      event_accuracy = (
          tf.reduce_sum(tf.multiply(correct_predictions, event_positions)) /
          tf.reduce_sum(event_positions) * 100)

      no_event_positions = (
          tf.to_float(tf.equal(labels_flat, no_event_label)) * mask_flat)
      no_event_accuracy = (
          tf.reduce_sum(tf.multiply(correct_predictions, no_event_positions)) /
          tf.reduce_sum(no_event_positions) * 100)

      global_step = tf.Variable(0, trainable=False, name='global_step')

      tf.add_to_collection('loss', loss)
      tf.add_to_collection('perplexity', perplexity)
      tf.add_to_collection('accuracy', accuracy)
      tf.add_to_collection('global_step', global_step)

      summaries = [
          tf.summary.scalar('loss', loss),
          tf.summary.scalar('perplexity', perplexity),
          tf.summary.scalar('accuracy', accuracy),
          tf.summary.scalar(
              'event_accuracy', event_accuracy),
          tf.summary.scalar(
              'no_event_accuracy', no_event_accuracy),
      ]

      if mode == 'train':
        learning_rate = tf.train.exponential_decay(
            hparams.initial_learning_rate, global_step, hparams.decay_steps,
            hparams.decay_rate, staircase=True, name='learning_rate')

        opt = tf.train.AdamOptimizer(learning_rate)
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,
                                                      hparams.clip_norm)
        train_op = opt.apply_gradients(zip(clipped_gradients, params),
                                       global_step)
        tf.add_to_collection('learning_rate', learning_rate)
        tf.add_to_collection('train_op', train_op)

        summaries.append(tf.summary.scalar(
            'learning_rate', learning_rate))

      if mode == 'eval':
        summary_op = tf.summary.merge(summaries)
        tf.add_to_collection('summary_op', summary_op)

    elif mode == 'generate':
      temperature = tf.placeholder(tf.float32, [])
      softmax_flat = tf.nn.softmax(
          tf.div(logits_flat, tf.fill([num_classes], temperature)))
      softmax = tf.reshape(softmax_flat, [hparams.batch_size, -1, num_classes])

      tf.add_to_collection('inputs', inputs)
      tf.add_to_collection('initial_state', initial_state)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)

  return graph