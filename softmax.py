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

    num_train = X.shape[0]
    num_classes = W.shape[1]

    log_C = -np.max(X.dot(W))

    p = np.zeros((X.shape[0], W.shape[1]))
    p = np.exp(X.dot(W))

    for i in range(num_train):
        p_sum = np.sum(p[i])

        correct_y = y[i]
        loss -= X[i].dot(W[:, y[i]]) + log_C
        temp = 0
        dW[:, y[i]] -= X[i]
        for j in range(num_classes):
            temp += np.exp(X[i].dot(W[:, j]) + log_C)
            dW[:, j] += X[i] * p[i][j] / p_sum
        loss += np.log(temp)
   
    
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


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

    
    """
    틀린 건데 왜 틀린건지 알아야만 함
    X * W[:, y].T 가 무슨 문제 있는 듯.
    log_C = -np.max(X.dot(W))

    L = np.zeros((X.shape[0], W.shape[1]))
    L += np.exp(X.dot(W) + log_C))
    loss += np.sum(np.log(np.sum(L, axis=1)))
    #L[np.arange(X.shape[0]), y] -= np.sum(X * W[:, y].T, axis=1))) + log_C # how?
    loss -= np.sum(np.sum(X * W[:, y].T, axis=1))) + log_C)

    # ---------------------------
    # 이렇게 하니까 맞음. 괄호 잘못 침 ㅋㅋ

    log_C = -np.max(X.dot(W))

    L = np.zeros((X.shape[0], W.shape[1]))
    L += np.exp(X.dot(W) + log_C))
    loss += np.sum(np.log(np.sum(L, axis=1)))
    #L[np.arange(X.shape[0]), y] -= np.sum(X * W[:, y].T, axis=1))) + log_C # how?
    loss -= np.sum(np.sum(X * W[:, y].T, axis=1) + log_C)
    """
    
    """
    # answer
    num_train = X.shape[0]
    num_classes = W.shape[1]

    p = np.zeros((X.shape[0], W.shape[1]))
    L = X.dot(W)
    L -= np.max(L, axis=1).reshape(-1, 1) # numeric instability...
    
    scores = np.exp(L) / np.sum(np.exp(L), axis=1).reshape(-1, 1)
    loss -= np.sum(np.log(scores[np.arange(num_train), y]))
    
    """
    num_train = X.shape[0]
    num_classes = W.shape[1]

    log_C = -np.max(X.dot(W))

    L = np.zeros((num_train, num_classes))
    L = X.dot(W)
    L -= np.max(L, axis=1).reshape(-1, 1) # numeric instability...
    loss += np.sum(np.log(np.sum(np.exp(L), axis=1)))
    #L[np.arange(X.shape[0]), y] -= np.sum(X * W[:, y].T, axis=1))) + log_C # how?
    loss -= np.sum(L[range(num_train), y]) 

    p = np.zeros((num_train, num_classes))
    p += np.exp(L) / np.sum(np.exp(L), axis=1).reshape(-1, 1)
    p[range(num_train), y] -= 1 

    dW += X.T.dot(p)
    
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
