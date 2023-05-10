from matplotlib.pyplot import bar_label
import torch
import random
import torchvision.transforms as T
import numpy as np
from .image_utils import SQUEEZENET_MEAN, SQUEEZENET_STD
from scipy.ndimage.filters import gaussian_filter1d

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode.
    model.eval()

    # Make input tensor require gradient.
    X.requires_grad_()

    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores (we'll combine losses across a batch by summing), and then compute  #
    # the gradients with a backward pass.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # First, lets see the size of the input X.
    print('This is the size of the input X: ', X.size())
    # We will now do the forward pass, computing the unnormalized scores.
    scores = model(X)
    # Lets see the shape of scores (should be (5,1000), 1000 scores for each 
    # image).
    print('Size of the scores: ', scores.size())

    # Now, lets get the correct (confidence) scores for the correct classes.
    # We can use .gather() method in PyTorch.
    correct_scores = scores.gather(1, y.view(-1, 1)).squeeze()
    print('the correct scores: ', correct_scores)

    # Now, lets compue the loss by simply adding up all of the correct scores.
    # This loss will be used to compute the gradient of the saliency map.
    # NOTE: we do not have to use cross entropy loss for this, as simply summing
    # the correct scores and doing backprop of this summation will still give us
    # the gradients of the loss wrt. the input images.
    loss = torch.sum(correct_scores)

    # Now, lets compute the backward pass of the loss wrt. the images X.
    loss.backward()

    # Finally, we can calculate the saliency map by getting the maximum 
    # absolute gradient values over the input's channels. We can do this by 
    # simply using the torch.max() method. We do this over the first dimension
    # (which is the channel's dimension). Finally, since torch.max() returns
    # a tuple (values, indices), we slice the tuple to only get the values.
    saliency = torch.max(torch.abs(X.grad), dim=1)[0]
    print('saliency should be only 2 dimensions (no channel dim): ', saliency.size())
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency

def make_fooling_image(X, target_y, model):
    """
    Generate a fooling image that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN

    Returns:
    - X_fooling: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our fooling image to the input image, and make it require gradient
    X_fooling = X.clone()
    X_fooling = X_fooling.requires_grad_()

    learning_rate = 1
    ##############################################################################
    # TODO: Generate a fooling image X_fooling that the model will classify as   #
    # the class target_y. You should perform gradient ascent on the score of the #
    # target class, stopping when the model is fooled.                           #
    # When computing an update step, first normalize the gradient:               #
    #   dX = learning_rate * g / ||g||_2                                         #
    #                                                                            #
    # You should write a training loop.                                          #
    #                                                                            #
    # HINT: For most examples, you should be able to generate a fooling image    #
    # in fewer than 100 iterations of gradient ascent.                           #
    # You can print your progress over iterations to check your algorithm.       #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # We will create a training loop of 100 iterations, used to update the 
    # gradients until the model is fooled.
    for iter in range(100):
      # Lets compute the unnormalized scores of the fooling image:
      scores = model(X_fooling)
      # Next, lets compute the probabilites, using softmax (norm the raw scores).
      softmax = torch.softmax(scores, dim=1)
      # Now, lets get the predicted class, by retrieving the max from the 
      # softmax result.
      pred = torch.argmax(softmax).item()
      # Lets compare the predicted with the actual. If the same, we can stop
      # iterating.
      if pred == target_y:
        break
      # If not, we will get the target score from the predicted, then do 
      # backward pass.
      target_score = scores[0, target_y]
      target_score.backward()
      # Lets do gradient ascent on the fooling image, but first we will have 
      # to normalize the gradients first. Then we will continue loop...
      with torch.no_grad():
        normalized_grad = torch.norm(X_fooling.grad)
        # Same process as for gradient descent, but its opposite. (MAX, not MIN)
        dX = learning_rate * X_fooling.grad / normalized_grad
        X_fooling += dX
        # zero out the grads before starting next iteration.
        X_fooling.grad.zero_()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return X_fooling

def class_visualization_update_step(img, model, target_y, l2_reg, learning_rate):
    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    ########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Simple: We get the raw, unnormalized scores of the image, get the target 
    # score from the aformentioned scores based on the target, then we compute
    # backward pass on this score. Using this, we then do gradient ascent on the 
    # image, add' a L2 Regulatization term, and then iterate this process until
    # new image is generated!
    
    # Make the image tensor require gradients.
    img.requires_grad_()
    # Forward pass. (the raw, unnormalized scores).
    scores = model(img)
    # Get the score based on the target_y. This is what we will get the gradients
    # from.
    target_score = scores[0, target_y]
    # Backward pass. 
    target_score.backward()
    # Now, do gradient ascent on the input image, making sure L2 is applied.
    with torch.no_grad():
        normalized_grad = torch.norm(img.grad)
        # Same process as for gradient descent, but its opposite. (MAX, not MIN)
        dX = learning_rate * img.grad / normalized_grad
        img += dX + l2_reg
        # zero out the grads before starting next iteration.
        img.grad.zero_()




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################


def preprocess(img, size=224):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / SQUEEZENET_STD).tolist()),
        T.Normalize(mean=(-SQUEEZENET_MEAN).tolist(), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def jitter(X, ox, oy):
    """
    Helper function to randomly jitter an image.

    Inputs
    - X: PyTorch Tensor of shape (N, C, H, W)
    - ox, oy: Integers giving number of pixels to jitter along W and H axes

    Returns: A new PyTorch Tensor of shape (N, C, H, W)
    """
    if ox != 0:
        left = X[:, :, :, :-ox]
        right = X[:, :, :, -ox:]
        X = torch.cat([right, left], dim=3)
    if oy != 0:
        top = X[:, :, :-oy]
        bottom = X[:, :, -oy:]
        X = torch.cat([bottom, top], dim=2)
    return X
