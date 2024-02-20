from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, erosion
from skimage.filters import threshold_otsu


def hw2_walkthrough1():
    
    #----------------- 
    # Convert a grayscale image to a binary image
    #-----------------
    img = Image.open('data/coins.png')
    img = img.convert('L')  # Convert the image to grayscale
    img = np.array(img)
    
    # Convert the image into a binary image by applying a threshold
    
    ### Solution - Devin Bresser ###
    threshold = 77
    ###
    
    bw_img = img > threshold
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    
    ax[1].imshow(bw_img, cmap='gray')
    ax[1].set_title('Binary Image')
    
    fig.savefig('outputs/binary_coins.png')
    plt.show()
    
    #----------------- 
    # Remove noises in the binary image
    #-----------------
    # Clean the image (you may notice some holes in the coins) by using
    # dilation and then erosion
    
    # Specify the size of the structuring element for erosion/dilation
    
    ### Solution - Devin Bresser ###
    k = 5
    
    erosion_element = np.ones((k,k))
    dilation_element = np.ones((k,k))
    
    # Apply erosion then dilation once to remove the noises
    
    fig, ax = plt.subplots(1, 2)
    processed_img = erosion(bw_img, erosion_element)
    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('After Erosion')
    
    processed_img = dilation(processed_img, dilation_element)
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title('After Dilation')
    
    ###
    
    plt.savefig('outputs/noise_removal_coins.png')
    plt.show()
        
    #----------------- 
    # Remove the rices
    #-----------------
    # Apply erosion then dilation once to remove the rices
    
    # Specify the size of the structuring element for erosion/dilation
    
    ### Solution - Devin Bresser
    
    k = 23
    selem = np.ones((k, k))
    
    fig, ax = plt.subplots(1, 2)
    processed_img = erosion(processed_img, selem)
    ax[0].imshow(processed_img, cmap='gray')
    ax[0].set_title('After Erosion')
    
    processed_img = dilation(processed_img, selem)
    ax[1].imshow(processed_img, cmap='gray')
    ax[1].set_title('After Dilation')
    
    ###
    
    fig.savefig('outputs/morphological_operations_coins.png')
    plt.show()
