#todo 1.q to ask, comment contract dict to writeup doc. is the first probability in logits corresponding to START or first vocab
#     2.do we need to mask *unk* etc in part2
#     3.how to reduce training time
#     4.perpelxity part 2 training for 1 epoch, print every 10 batch, only check the final printed perplecity < 10?
#     5.embedding size vs rnn size
import numpy as np
import tensorflow as tf
from preprocess import *

class RNN_Part1(tf.keras.Model):
    def __init__(self, vocab):
        """
        The RNN_Part1 class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(RNN_Part1, self).__init__()

        # TODO: initialize tf.keras.layers!
        # - tf.keras.layers.Embedding for embedding layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
        # - tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN
        self.vocab = vocab
        #print("vocab",vocab)
        self.vocab_size = len(vocab)
        self.embedding_size = 300
        self.rnn_size = 256

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.rnn_layer = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True)
        self.dense_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network 
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor
        """
        # TODO: implement the forward pass calls on your tf.keras.layers!
        #print("inputs", inputs.shape)
        embedding_output = self.embedding_layer(inputs)
        print("embedding output", embedding_output.shape, embedding_output)
        rnn_output = self.rnn_layer(embedding_output)
        print("rnn output", rnn_output.shape,rnn_output)
        dense_output = self.dense_layer(rnn_output)
        #print("dense output", dense_output.shape)

        return dense_output

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        # We recommend using tf.keras.losses.sparse_categorical_crossentropy
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy

        # TODO: implement the loss function with mask as described in the writeup
        mask = tf.less(labels, self.vocab[FIRST_SPECIAL]) | tf.greater(labels, self.vocab[LAST_SPECIAL])
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        loss = tf.boolean_mask(loss, mask)
        loss = tf.reduce_mean(loss)
        return loss


class RNN_Part2(tf.keras.Model):
    def __init__(self, french_vocab, english_vocab):

        super(RNN_Part2, self).__init__()

        french_vocab_size = len(french_vocab)
        english_vocab_size = len(english_vocab)

        self.french_vocab = french_vocab
        self.english_vocab = english_vocab

        # TODO: initialize tf.keras.layers!
        self.embedding_size = 32
        self.rnn_size = 128

        self.french_embedding_layer = tf.keras.layers.Embedding(french_vocab_size, self.embedding_size)
                                                #embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.french_rnn_layer = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        # self.french_dense_layer = tf.keras.layers.Dense(french_vocab_size, activation='softmax')
        self.english_embedding_layer = tf.keras.layers.Embedding(english_vocab_size, self.embedding_size)
                                               # embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.1))
        self.english_rnn_layer = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True)
        self.english_dense_layer = tf.keras.layers.Dense(english_vocab_size, activation='softmax')

    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        # TODO: implement the forward pass calls on your tf.keras.layers!
        # Note 1: in the diagram there are two inputs to the decoder
        #  (the decoder_input and the hidden state of the encoder)
        #  Be careful because we don't actually need the predictive output
        #   of the encoder -- only its hidden state
        # Note 2: If you use an LSTM, the hidden_state will be the last two
        #   outputs of calling the rnn. If you use a GRU, it will just be the
        #   second output.
        #print("encoder input", encoder_input)
        #print("Decoder input", decoder_input)
        french_embedding_output = self.french_embedding_layer(encoder_input)
        french_hidden_state = self.french_rnn_layer(french_embedding_output)[1:]
        english_embedding_output = self.english_embedding_layer(decoder_input)
        english_rnn_output = self.english_rnn_layer(english_embedding_output, initial_state=french_hidden_state)
        english_dense_output = self.english_dense_layer(english_rnn_output)
        #print("dense oupyput", english_dense_output.shape)
        return english_dense_output

    def loss_function(self, probs, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.

        :param probs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :return: the loss of the model as a tensor
        """


        # When computing loss, we need to compare the output probs and labels with a shift
        #  of 1 to ensure a proper alignment. This is because we generated the output by passing
        #  in a *START* token and the encoded French state.
        #
        # - The labels should have the first token removed:
        #	 [*START* COSC440 is the best class. *STOP*] --> [COSC440 is the best class. *STOP*]
        # - The logits should have the last token in the window removed:
        #	 [COSC440 is the best class. *STOP* *PAD*] --> [COSC440 is the best class. *STOP*]

        # TODO: implement the loss function with mask as described in the writeup


        #print("original probls", probs)
        #print("original lables", labels)
        probs = probs[:,:-1,:]
        labels = labels[:, 1:]
        #print("probs",probs)
      #  print("labels",labels)

        english_mask = tf.math.not_equal(labels, self.english_vocab[PAD_TOKEN])
        english_mask = (labels != self.english_vocab[PAD_TOKEN])

        labels = tf.boolean_mask(labels, english_mask)
        probs = tf.boolean_mask(probs, english_mask)
        #print("after lables", labels)
        english_mask = tf.math.not_equal(labels, self.english_vocab[PAD_TOKEN])

        labels = tf.boolean_mask(labels, english_mask)
       # print("labels masks",labels)
        probs = tf.boolean_mask(probs, english_mask)
       # print("probs makss",probs)

        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

        return loss
