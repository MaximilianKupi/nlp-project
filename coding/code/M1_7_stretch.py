import matplotlib.pyplot as plt
import numpy as np
import torch


def stretch(vector,n,plot=False):
    """Stretches Vectors to desired length by interpolating values


    Example
    stretch([0,0,0,0,20,0,0,1],15)

    with plotting
    stretch([0,0,0,0,20,0,0,1],15,True)

    Args:
        vector (array(float)): The input vector that should be stretched
        n (int): The length of the output vector after stretching

    Returns:
        tensor: stretched vector

    """
    if(len(vector) == 0):
        return torch.zeros(1,n)

    if(n<len(vector)):
        return vector
        #raise ValueError("Shrinking not allowed")
    # Converts input to numpy array
    oldY = np.array(vector)
    # Treates input as y-values of a function on x-values created here
    oldX = np.linspace(0,len(vector)-1,len(vector))

    # Determine non-zero values that we have to keep
    # If we deviate from these points in the interpolation
    # then the result will shrink the dictionary evaluation number
    keep = np.where(oldY != 0)[0]

    # Create new vector with desired length
    newX = np.linspace(0,len(vector)-1,n)

    # Make sure that the nonzero values are kept as is after interpolation
    # by replacing the closest values with the ones we saved in keep
    for value in keep:
        newX[np.abs(np.array(newX) - value).argmin()] = value

    # interpolate to new 
    newY = np.interp(newX,oldX,oldY)

    if(plot):
        plt.plot(oldX,oldY)
        plt.plot(newX,newY)
    return torch.Tensor(newY)