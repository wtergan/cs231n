from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # compute the dot product of the inputs and the weights.
    W_X = X.dot(W)
    # compute the softmax.
    exp_values = np.exp(W_X - np.max(W_X, axis=1, keepdims=True))
    softmax = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    # compute the loss.
    conf_1 = softmax[range(len(softmax)), y]
    loss_1 = np.mean(-np.log(conf_1)) + 0.5 * reg * np.sum(W*W)
    # compute the one hot equivalent for y.
    one_hot = np.zeros(W_X.shape)
    one_hot[range(X.shape[0]), y] = 1
    # compute loss with one hot encode now:
    conf_2 = np.sum(softmax * one_hot, axis=1, keepdims=True)
    loss_2 = np.mean(-np.log(conf_2)) + 0.5 * reg * np.sum(W*W)
    # loss_1 == loss_2
    print(loss_1 == loss_2)
    
    # compute gradient with the one_hot array.
    '''
    computes the difference between the gradients of the loss function wrt
    the weights of the predicted label class (softmax)
    and the gradients of the loss function wrt to the weights of the true 
    label class (one_hot vector/matrix)

    computing the contributions of each feature to the incorrect predictions and 
    the correct predictions separately.

    X.T.dot(one_hot) is the contribution matrix 
    of each feature to the correct predictions
    in each class.
          X = (N, F), one_hot (N, C), results (F, C)

    X.T.dot(softmax) is the contribution matrix 
    of each feature in the incorrect predictions for each class.
          X = (N, F), softmax = (N, C), results (F, C)
    '''
    
    dW = (X.T.dot(softmax) - X.T.dot(one_hot)) / y.size + 0.5 * W * reg
    return loss_2, dW

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_scores = scores[range(num_train), y].reshape((num_train, 1))
    sum_j = np.sum(np.exp(scores), axis=1).reshape((num_train, 1))

    loss = np.sum(-1 * correct_class_scores + np.log(sum_j)) / num_train + 0.5 * reg * np.sum(W * W)

    correct_matrix = np.zeros(scores.shape)
    correct_matrix[range(num_train), y] = 1

    dW = X.T.dot(np.exp(scores) / sum_j) - X.T.dot(correct_matrix)
    dW = dW / num_train + W * reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
