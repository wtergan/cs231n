"""This file defines layer types that are commonly used for recurrent neural networks.
"""

import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)
      
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # ht = tanh(Wh*ht-1 + Wx*xt) + b
    # Do the dot product of previous h and its weights, as well as current input 
    # and its weights, add bias, and then apply tanh to result.
    next_h = np.tanh((prev_h.dot(Wh)) + (x.dot(Wx)) + b) 
    cache = (next_h, x, prev_h, Wx, Wh, b)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Lets unpack the cache tuple to get the variables needed for backprop.
    next_h, x, prev_h, Wx, Wh, b = cache

    # derivative of tanh wrt. its input x = 1 - tanh(x)^2
    # tanh(x) is our output from forward pass, which is next_h, thus:
    dtanh = dnext_h * (1 - np.square(next_h))

    # Now that we have the derivative of tanh wrt. dnext_h, we can easily 
    # compute our other gradients.
    dx = dtanh.dot(Wx.T)
    dprev_h = dtanh.dot(Wh.T)
    dWx = x.T.dot(dtanh)
    dWh = prev_h.T.dot(dtanh)
    db = np.sum(dtanh, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Lets initialize cache as a list, h as a list, with the initial state in first.
    cache = []
    h = [h0]

    for t in range(x.shape[1]):
        # With the slicing technique, we passing in x (N, D) at timestep t.
        next_h, t_cache = rnn_step_forward(x[:,t,:], h[t], Wx, Wh, b)
        cache.append(t_cache)
        h.append(next_h)
    
    # Stack h along the time axis, ignoring the initial state h0.
    h = np.stack(h[1:], axis=1) # first axis is the T axis.

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Lets get the shapes we need to initialize the gradients. We can get most from dh.
    N, T, H = dh.shape[0], dh.shape[1], dh.shape[2]
    # We have to get data dimension from one of the Wx weights in the cache.
    D = cache[0][3].shape[0]

    # Now that we have the dimensions, we can start initializing the gradients.
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)

    # Now that we initialized the gradients, lets loop through the timesteps, 
    # accessing cache that correspond to that step (go in reverse).
    # We will calculate the gradients at that timestep using dh for that specific step.
    for t in range(T-1, -1, -1):
      # Get the gradients from step backward at timestep t.
      dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dh[:,t,:] + dh0, cache[t])
      # Set dx at timestep t as the dx_t that was calculated during the backward pass,
      # so that at each t dimension, it holds (N, D) values at t!
      dx[:,t,:] = dx_t
      # Set dh0 to the current dprev_h, which will be passed to the previous timestep.
      dh0 = dprev_h
      # add other gradients at timestep t to their respective gradients.
      dWx += dWx_t
      dWh += dWh_t
      db += db_t

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # intuition of out = W[x] is as follows:
      # x is sequence of words, where each example sequence (N) is of length T, 
      # and subsequently, each index in x correspond to V (vocabulary word in V).

      # W is the weight matrix, represented as the word embedding, where each row
      # of W is a vector representation (embeddding) of its corresponding word in V.

      # So when we do array indexing of W[x], what we are doing is creating a new
      # array, where each word in each example sequence of words in x will now be
      # replaced by its vector representation (embedding) from W.

    out, cache = W[x], (x, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """Backward pass for word embeddings.
    
    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # get values from cache.
    x, W = cache

    # initialize the weight gradient to be the same shape as W.
    dW = np.zeros_like(W)

    # We can add dout (N, T, D) to dW (V, D) at index x (N, T).
    np.add.at(dW, x, dout)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """A numerically stable version of the logistic sigmoid function."""
    pos_mask = x >= 0
    neg_mask = x < 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Activation vector a, compute a, then split into 4.
    a = np.hsplit(x.dot(Wx) + prev_h.dot(Wh) + b, 4)

    input_gate = sigmoid(a[0]) # input gate
    forget_gate = sigmoid(a[1]) # forget gate
    output_gate = sigmoid(a[2]) # output gate
    g_gate = np.tanh(a[3]) # affine transformation of input x.

    # Computation of the next LSTM cell state, and then the next hidden state.
    next_c = forget_gate * prev_c + input_gate * g_gate
    next_h = output_gate * np.tanh(next_c)

    # Save the values needed for the backward pass.
    cache = (prev_h, prev_c, next_c, input_gate, forget_gate, output_gate, g_gate, x, Wx, Wh)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    """Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Unpack the cache values needed to compute the gradients.
    prev_h, prev_c, next_c, input_gate, forget_gate, output_gate, g_gate, x, Wx, Wh = cache

    # We have the upstream gradient loss wrt. next_c (dnext_c), but we have to 
    # update this by adding the contribution from dnext_h, since next_c contributes to
    # the loss from next_h (dnext_h).

    # dL/dnext_c = dL/dnext_c + dL/dnext_h * dnext_h/dnext_c.
    dnext_c += dnext_h * output_gate * (1 - (np.tanh(next_c)**2))

    # Once dnext_c is computed, we can now compute dprev_c.
    dprev_c = dnext_c * forget_gate

    # After gradients for prev_c is computed, lets get the gradients for each
    # of the gates.
    # For most of the gates, have to compute the sigmoid derivative.
    d_input_gate = dnext_c * g_gate * input_gate * (1-input_gate)
    d_forget_gate = dnext_c * prev_c * forget_gate * (1-forget_gate)
    d_output_gate = dnext_h * np.tanh(next_c) * output_gate * (1-output_gate)
    d_g_gate = dnext_c * input_gate * (1-(g_gate)**2)

    # Since we originally computed the activation vector, and then split the 
    # aformentioned vector into 4 to compute the gates, we will have to join them
    # back together in order to compute the gradients of input x, Wx, Wh, and b.
    da = np.hstack((d_input_gate, d_forget_gate, d_output_gate, d_g_gate))

    # Once joined, do final gradients.
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    db = da.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    cache = []
    # Initialize h as a list, add initial hidden state first.
    h = [h0]
    # Initialize cell state c as 0.
    c0 = np.zeros_like(h0)
    c = [c0]

    # Iterating over the timesteps T (which happens to be the second dimension of x).
    for t in range(x.shape[1]):
      # We will get the (N,D) from x at each timestep t, insert into step_forward.
      next_h, next_c, t_cache = lstm_step_forward(x[:,t,:], h[t], c[t], Wx, Wh, b)
      h.append(next_h)
      c.append(next_c)
      cache.append(t_cache)
    
    # After forward, must stack h, c, along the T axis (second dimension).
    # Omit the initial hidden states for each of them.
    h = np.stack(h[1:], axis=1)
    c = np.stack(c[1:], axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Similar process to the rnn_forward method above.
    # Lets get the dimensions N, T, H, from the upstream gradient.
    (N, T, H) = dh.shape

    # Get the dimension D, 4H from a Wx weight from the cache.
    (D , H4) = cache[0][8].shape

    # Now that we have the dimensions, lets initialize the gradients.
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dc0 = np.zeros((N, H))
    dWx = np.zeros((D, H4))
    dWh = np.zeros((H, H4))
    db = np.zeros(H4)

    # Loop in reverse, compute gradients at each timestep t.
    # Difference between this and rnn_forward is the inclusion of the cell state.
    for t in range(T-1, -1, -1):
      # remember, dh0 and dc0 is dprev_h and dprev_c respectively.
      dx_t, dh0, dc0, dWx_t, dWh_t, db_t = lstm_step_backward(dh[:, t, :] + dh0, dc0, cache[t])
      # Set dx_t inside dx at timestep t.
      dx[:, t, :] = dx_t
      # add computed gradients at timestep t to initialzed gradients
      dWx += dWx_t
      dWh += dWh_t
      db += db_t

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    # Reshape x to be 2D, then perform dot with w, reshape back, add biases.
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    # Same reshaping as the forward pass, but this time we are computing the 
    # gradients.
    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    # Get the dimensions from the input's shape for usage.
    N, T, V = x.shape

    # Reshape x, y, and mask, so that x is 2D and y, mask is 1D respectively.
    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    # Mask, array that determines which scores in each x_i should contribute to
    # the loss. This is changed to be a 1D array.
    mask_flat = mask.reshape(N * T)

    # Compute the predicted labels (probs).
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    # Computation of the average negative log likelihood. (cross entropy).
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    # Create a copy of probs for usage in computing the gradients.
    dx_flat = probs.copy()
    # Computation of the gradients of x wrt the loss. 
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    # Add a second axis to mask array, then compute element-wise multiplication
    # with the gradients of x wrt. the loss.
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
