import numpy as np

from ..rnn_layers import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=np.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)
        # Self initialization of null, start, and end.
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors. Essentially the word embeddings matrix.
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        # Scaling each element in "W_embed" by 100, making them closer to 
        # zero. Helps with stability, and convergence of the model.
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters.
        # Random gaussian distribution for weights (Xavier's initialization).
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        # Initialization of the biases.
        self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN and/ or LSTM. 
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        # Xavier's initialization of the weights and biases for
        # input x and hidden state h respectively.
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        # Initialization of the biases.
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # Xavier's initialization of output to vocab weights.
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        # Initialization of the biases.
        self.params["b_vocab"] = np.zeros(vocab_size)

        # Cast parameters to correct dtype.
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this. Boolean array that indicates which elements of the 
        # caption_out are valid words and which are NULL token that should be 
        # ignored in the loss computation.
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state. 
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix. 
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN/ LSTM
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the forward and backward passes for the CaptioningRNN.   #
        # In the forward pass you will need to do the following:                   #
        # (1) Use an affine transformation to compute the initial hidden state     #
        #     from the image features. This should produce an array of shape (N, H)#
        # (2) Use a word embedding layer to transform the words in captions_in     #
        #     from indices to vectors, giving an array of shape (N, T, W).         #
        # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
        #     process the sequence of input word vectors and produce hidden state  #
        #     vectors for all timesteps, producing an array of shape (N, T, H).    #
        # (4) Use a (temporal) affine transformation to compute scores over the    #
        #     vocabulary at every timestep using the hidden states, giving an      #
        #     array of shape (N, T, V).                                            #
        # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
        #     the points where the output word is <NULL> using the mask above.     #
        #                                                                          #
        #                                                                          #
        # Do not worry about regularizing the weights or their gradients!          #
        #                                                                          #
        # In the backward pass you will need to compute the gradient of the loss   #
        # with respect to all model parameters. Use the loss and grads variables   #
        # defined above to store loss and gradients; grads[k] should give the      #
        # gradients for self.params[k].                                            #
        #                                                                          #
        # Note also that you are allowed to make use of functions from layers.py   #
        # in your implementation, if needed.                                       #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Affine transformation for the initial hidden state.
        h0, h0_cache = affine_forward(features, W_proj, b_proj)

        # Transformation of the words in 'caption_in' from indices to vectors.
        word_2_vec, word_2_vec_cache = word_embedding_forward(captions_in, W_embed)

        # Forward pass for rnn or lstm.
        if self.cell_type == 'rnn':
          h, h_cache = rnn_forward(word_2_vec, h0, Wx, Wh, b)
        elif self.cell_type == 'lstm':
          h, h_cache = lstm_forward(word_2_vec, h0, Wx, Wh, b)

        # Temporal affine transformation for the raw, unnormalized scores.
        scores, scores_cache = temporal_affine_forward(h, W_vocab, b_vocab)

        # Computation of the loss, gradients of the scores wrt. loss via temporal loss.
        loss, dout = temporal_softmax_loss(scores, captions_out, mask, verbose=False)

        # Gradients of h, W_vocab, b_vocab.
        dout, dW_vocab, db_vocab = temporal_affine_backward(dout, scores_cache)

        # Gradients of word vectors, h0, Wx, Wh, b. Depend on rnn or lstm.
        if self.cell_type == 'rnn':
          dout, dh0, dWx, dWh, db = rnn_backward(dout, h_cache)
        elif self.cell_type == 'lstm':
          dout, dh0, dWx, dWh, db = lstm_backward(dout, h_cache)

        # Gradients of W_embed matrix.
        dW_embed = word_embedding_backward(dout, word_2_vec_cache)

        # Gradients of features, WQ_proj, and b_proj.
        dout, dW_proj, db_proj = affine_backward(dh0, h0_cache)

        # Gradients into grads dict.
        grads = {
            "W_proj": dW_proj,
            "b_proj": db_proj,
            "W_embed": dW_embed,
            "Wx": dWx,
            "Wh": dWh,
            "b": db,
            "W_vocab": dW_vocab,
            "b_vocab": db_vocab
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO: Implement test-time sampling for the model. You will need to      #
        # initialize the hidden state of the RNN by applying the learned affine   #
        # transform to the input image features. The first word that you feed to  #
        # the RNN should be the <START> token; its value is stored in the         #
        # variable self._start. At each timestep you will need to do to:          #
        # (1) Embed the previous word using the learned word embeddings           #
        # (2) Make an RNN step using the previous hidden state and the embedded   #
        #     current word to get the next hidden state.                          #
        # (3) Apply the learned affine transformation to the next hidden state to #
        #     get scores for all words in the vocabulary                          #
        # (4) Select the word with the highest score as the next word, writing it #
        #     (the word index) to the appropriate slot in the captions variable   #
        #                                                                         #
        # For simplicity, you do not need to stop generating after an <END> token #
        # is sampled, but you can if you want to.                                 #
        #                                                                         #
        # HINT: You will not be able to use the rnn_forward or lstm_forward       #
        # functions; you'll need to call rnn_step_forward or lstm_step_forward in #
        # a loop.                                                                 #
        #                                                                         #
        # NOTE: we are still working over minibatches in this function. Also if   #
        # you are using an LSTM, initialize the first cell state to zeros.        #
        ###########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Forward test time pass:
          # Affine transformation for the initial hidden state based on image features.
          # Create word array, which will initially the newly create sequence with 
            # the start token, for each N example.
          # For each time step t from 0 to the max length:
            # embed the word with its corresponding weight matrix.
            # do an rnn/lstm step to get the hidden state
            # apply affine transformation on that hidden state to get scores.
            # get the max score, add that to captions at specified timestep.
            # set word as max_score, and update h0 to be h. Continue loop.

        h, _ = affine_forward(features, W_proj, b_proj)
        word = np.repeat(self._start, N)

        # if using lstm, must initialize the first cell state.
        c = np.zeros_like(h)

        for t in range(max_length):
          word_embed, _ = word_embedding_forward(word, W_embed)
          if self.cell_type == 'rnn':
            h, _ = rnn_step_forward(word_embed, h, Wx, Wh, b)
          # Compute the hidden cell state, assign to c to use for next iteration.
          elif self.cell_type == 'lstm':
            h, c, _ = lstm_step_forward(word_embed, h, c, Wx, Wh, b)
          scores, _ = affine_forward(h, W_vocab, b_vocab)
          max_score = np.argmax(scores, axis=1)
          captions[:, t] = max_score
          word = max_score
          h0 = h


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return captions
