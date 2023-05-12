from ctypes import alignment
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

"""
This file defines layer types that are commonly used for transformers.
"""

class PositionalEncoding(nn.Module):
    """
    Encodes information about the positions of the tokens in the sequence. In
    this case, the layer has no learnable parameters, since it is a simple
    function of sines and cosines.
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)
        ############################################################################
        # TODO: Construct the positional encoding array as described in            #
        # Transformer_Captioning.ipynb.  The goal is for each row to alternate     #
        # sine and cosine, and have exponents of 0, 0, 2, 2, 4, 4, etc. up to      #
        # embed_dim. Of course this exact specification is somewhat arbitrary, but #
        # this is what the autograder is expecting. For reference, our solution is #
        # less than 5 lines of code.                                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        i = torch.arange(max_len)[:, None]
        pows = torch.pow(10000, -torch.arange(0, embed_dim, 2) / embed_dim)

        # in the third dimension of the pe tensor:
          # if position is even, sin, if odd, cosine.
        pe[0, :, 0::2] = torch.sin(i * pows)
        pe[0, :, 1::2] = torch.cos(i * pows)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))
        ############################################################################
        # TODO: Index into your array of positional encodings, and add the         #
        # appropriate ones to the input sequence. Don't forget to apply dropout    #
        # afterward. This should only take a few lines of code.                    #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Add x with the positional encoding along the second axis, up to length of S.
        output = x + self.pe[:, :S]
        output = self.dropout(output)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


class MultiHeadAttention(nn.Module):
    """
    A model layer which implements a simplified version of masked attention, as
    introduced by "Attention Is All You Need" (https://arxiv.org/abs/1706.03762).

    Usage:
      attn = MultiHeadAttention(embed_dim, num_heads=2)

      # self-attention
      data = torch.randn(batch_size, sequence_length, embed_dim)
      self_attn_output = attn(query=data, key=data, value=data)

      # attention using two inputs
      other_data = torch.randn(batch_size, sequence_length, embed_dim)
      attn_output = attn(query=data, key=other_data, value=other_data)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Construct a new MultiHeadAttention layer.

        Inputs:
         - embed_dim: Dimension of the token embedding
         - num_heads: Number of attention heads
         - dropout: Dropout probability
        """
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by
        # implementation).

        # Remember, nn.Linear is a linear transformation layer, so that when you
        # call these, it will implicitly perform the matrix multiplication between
        # the weight matrix of shape (embed_dim, embed_dim) and the input to this 
        # call. 
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head
        # Scale the alignment scores before computing our attention weights.
        self.scale = math.sqrt(embed_dim / num_heads)

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the masked attention output for the provided data, computing
        all attention heads in parallel.

        In the shape definitions below, N is the batch size, S is the source
        sequence length, T is the target sequence length, and E is the embedding
        dimension.

        Inputs:
        - query: Input data to be used as the query, of shape (N, S, E)
        - key: Input data to be used as the key, of shape (N, T, E)
        - value: Input data to be used as the value, of shape (N, T, E)
        - attn_mask: Array of shape (S, T) where mask[i,j] == 0 indicates token
          i in the source should not influence token j in the target.

        Returns:
        - output: Tensor of shape (N, S, E) giving the weighted combination of
          data in value according to the attention weights calculated using key
          and query.
        """
        N, S, E = query.shape
        N, T, E = value.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, E))
        ############################################################################
        # TODO: Implement multiheaded attention using the equations given in       #
        # Transformer_Captioning.ipynb.                                            #
        # A few hints:                                                             #
        #  1) You'll want to split your shape from (N, T, E) into (N, T, H, E/H),  #
        #     where H is the number of heads.                                      #
        #  2) The function torch.matmul allows you to do a batched matrix multiply.#
        #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
        #     shape (N, H, T, T). For more examples, see                           #
        #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
        #  3) For applying attn_mask, think how the scores should be modified to   #
        #     prevent a value from influencing output. Specifically, the PyTorch   #
        #     function masked_fill may come in handy.                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY 2THIS LINE)*****

        # Set H as the number of heads.
        H = self.n_head # Set H as the number of heads

        # Q have to include the shape of the source sequence (S).
        # K, V have to include the shape of the target sequence (T).

        # Do linear transformation of self.query and query, reshape to include H, 
        # E/H, then swap the dimensions of S and H, so that new shape is (N, H, S, E/H) 
        # H should be in the second dimension, S or T in the third dimension.
        Q = self.query(query).view(N, S, H, E//H).moveaxis(1, 2) # H is now in 2nd dim.

        # Do same linear transformation for K, V respectively, just that now S is T.
        K = self.key(key).view(N, T, H, E//H).moveaxis(1, 2)
        V = self.value(value).view(N, T, H, E//H).moveaxis(1, 2)

        # Multi-head scaled dot-product attention.

        # Computation of alignment scores, which is scaled before softmax.
        # Must transpose K so that matrix product applies.
        # (N,H,S,E//H) * (N,H,E//H,T) = (N,H,S,T)
        alignment_scores = (Q.matmul(K.moveaxis(2, 3))) / self.scale

        # If attn_mask is not None, apply mask to the alignment scores (-neg infinity)
        # Then, once softmax is applied, those values will close to 0.
        if attn_mask is not None:
          alignment_scores = alignment_scores.masked_fill(attn_mask==0, float("-inf"))

        # Computation of our attention weights (softmax), with dropout applied.
        # Softmax on last dimension, to normalized probabilities representing 
        # the attention distribution over the source sequence for each target position.
        attn_weights = self.attn_drop(F.softmax(alignment_scores, dim=-1))

        # Computation of our attention output (which is context matrix.)
        # (N,H,S,T) * (N,H,T,E//H) = (N,H,S,E//H)
        attn_output = attn_weights.matmul(V)

        # Linearly project the attention output with the weight self.proj.
        output = self.proj(attn_output.moveaxis(1, 2).reshape((N, S, E)))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return output


