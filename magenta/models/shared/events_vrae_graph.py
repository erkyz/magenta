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

TEMP_LATENT_SIZE = 10
TEMP_HIDDEN_SIZE = 64
TRAIN_BATCH_SIZE = 64 #TODO save properly

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
      # TODO this is hacky af. also where is this happening?
      state_is_tuple = False

    with tf.variable_scope('encoder'):
        encoder_cell = make_rnn_cell(hparams.rnn_layer_sizes, 2,
                         dropout_keep_prob=hparams.dropout_keep_prob,
                         attn_length=0, # do not use attention on the encoder
                         state_is_tuple=True)

        encoder_initial_state = encoder_cell.zero_state(hparams.batch_size, tf.float32)

        # TODO alter so it doesn't output stuff?
        _, encoder_final_state = tf.nn.dynamic_rnn(
            encoder_cell, inputs, initial_state=encoder_initial_state, parallel_iterations=1,
            swap_memory=True)

        # We only look at the cell state of the last layer of the encoder LSTM
        encoder_final_hidden_state = encoder_final_state[-1].h
        z_mu = tf.contrib.layers.fully_connected(encoder_final_hidden_state, 
                num_outputs=TEMP_LATENT_SIZE, activation_fn=None, trainable=True)
        z_logvar = tf.contrib.layers.fully_connected(encoder_final_hidden_state, 
                num_outputs=TEMP_LATENT_SIZE, activation_fn=None, trainable=True)

    # decoder
    with tf.variable_scope('decoder'):
        numDecodingLayers = len(hparams.rnn_layer_sizes)
        # TODO attention
        decoder_cell = make_rnn_cell(hparams.rnn_layer_sizes, numDecodingLayers,
                             dropout_keep_prob=hparams.dropout_keep_prob,
                             attn_length=hparams.attn_length,
                             state_is_tuple=state_is_tuple)
        
        # sample z using reparameterization trick
        if state_is_tuple:
            decoder_h0 = []
            zero_state = decoder_cell.zero_state(hparams.batch_size, dtype=tf.float32)
            zerostate = zero_state[0] if hparams.attn_length > 0 else zero_state
            for c, h, in zerostate:
                # TODO each layer should have its own z_mu, z_logvar?
                epsilon = tf.random_normal(tf.shape(z_logvar), 0, 1, dtype=tf.float32)
                z = z_mu + tf.mul(tf.sqrt(tf.exp(z_logvar)), epsilon)
                hidden1 = tf.contrib.layers.fully_connected(z, 
                        num_outputs=TEMP_HIDDEN_SIZE,
                        trainable=True)
                hidden2 = tf.contrib.layers.fully_connected(hidden1, 
                        num_outputs=TEMP_HIDDEN_SIZE,
                        trainable=True)
                h_state = tf.contrib.layers.fully_connected(hidden2, 
                        num_outputs=hparams.rnn_layer_sizes[len(decoder_h0)],
                        trainable=True)
                decoder_h0.append(tf.nn.rnn_cell.LSTMStateTuple(c, h_state))
            decoder_h0 = tuple(decoder_h0)
            if hparams.attn_length > 0:
                decoder_h0 = (decoder_h0, zero_state[1], zero_state[2])
        else:
            # take z_mu, z_logvar from last batch input since batch_size is 1 for gen
            epsilon = tf.random_normal(tf.shape(z_logvar[-1]), 0, 1, dtype=tf.float32)
            z = z_mu[-1] + tf.mul(tf.sqrt(tf.exp(z_logvar[-1])), epsilon)
            z = tf.reshape(z, [1, TEMP_LATENT_SIZE])
            hidden1 = tf.contrib.layers.fully_connected(z, 
                    num_outputs=TEMP_HIDDEN_SIZE,
                    trainable=True)
            hidden2 = tf.contrib.layers.fully_connected(hidden1, 
                    num_outputs=TEMP_HIDDEN_SIZE,
                    trainable=True)
            # this assumes that all the layer sizes are the same
            h_state = tf.contrib.layers.fully_connected(hidden2, 
                    num_outputs=hparams.rnn_layer_sizes[0], 
                    trainable=True)
            decoder_h0 = decoder_cell.zero_state(hparams.batch_size, dtype=tf.float32)
            for i in range(numDecodingLayers):
                # set initial hidden state of all decoder layers
                lenBefore = sum(hparams.rnn_layer_sizes[:i])*2 + hparams.rnn_layer_sizes[i]
                lenAfter = sum(hparams.rnn_layer_sizes[i+1:])*2
                attn_pad = 0
                if hparams.attn_length > 0:
                    attn_pad = TRAIN_BATCH_SIZE + hparams.attn_length*TRAIN_BATCH_SIZE
                mask = tf.pad(h_state, [[0,0], [lenBefore, lenAfter + attn_pad]])
                decoder_h0 += mask

        # Note: tf.reverse syntax unique to TF 0.12
        # TODO does reverse still make sense in a Melody? continue goes backwards.
        # TODO do you need <EOS>?
        outputs, final_state = tf.nn.dynamic_rnn(
            decoder_cell, tf.reverse(inputs, [False, False, True]), initial_state=decoder_h0,
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

      global_step = tf.Variable(0, trainable=False, name='global_step')
      # KL weight for annealing
      kl_weight = tf.minimum(tf.div(tf.cast(global_step,tf.float32),5000.), 1.)

      # "latent loss" -- KL divergence D[Q(z|x)||P(z)] where z ~ N(0,I)
      # see Doersch tutorial page 9
      # TODO this term should be larger... wanna do KL cost annealing or something.
      kld = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mu) - tf.exp(z_logvar), 1)
      # "reconstruction loss" 
      # VR lower bound -- cross entropy is equivalent to negative log likelihood.
      # TODO scale on this?
      reconstruction_loss = tf.reduce_sum(mask_flat * softmax_cross_entropy) / num_logits
      # average over batch
      loss = tf.reduce_mean(reconstruction_loss + kl_weight*kld)
      # TODO regularize loss
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
          tf.summary.scalar(
              'kl_cost', tf.reduce_mean(kld)),
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
      tf.add_to_collection('initial_state', decoder_h0)
      tf.add_to_collection('final_state', final_state)
      tf.add_to_collection('temperature', temperature)
      tf.add_to_collection('softmax', softmax)
      tf.add_to_collection('z_logvar', z_logvar)
      tf.add_to_collection('z_mu', z_mu)

  return graph
