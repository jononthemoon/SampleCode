''' 

This lab solves three problems: green screening, manual stitching of multiple images to create a composite, and warping a photo onto another given a set of points.



'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import transform as tf
import sys
from scipy import signal
from scipy.ndimage import map_coordinates

######  Auxiliary Methods   #####################
def color_index(image,index):
    if index == 0:
        image = 2*image[:,:,0] - image[:,:,1] - image[:,:,2]
    elif index == 1:
        image = 2*image[:,:,1] - image[:,:,0] - image[:,:,2]
    else:
        image = 2*image[:,:,2] - image[:,:,0] - image[:,:,1]
    return image

def gray_level(image):
    image = .299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
    return image

def correlate2d_scipy(image,filt):
    r,c = filt.shape
    if len(image.shape) == 3:
        result = np.zeros((image.shape[0]-r + 1, image.shape[1]-c + 1, image.shape[2]))
        for z in range(image.shape[2]):
            result[:,:,z] = signal.correlate2d(image[:,:,z], filt,mode='valid')      
    else:
        result = signal.correlate2d(image, filt,mode='valid')
    return result

def gaussian_filter(size, sigma):
    d = ((np.arange(size) - (size-1)/2)**2).reshape(-1,1)
    f = np.exp(-(d+d.T)/2/sigma/sigma)
    return f/np.sum(f)

def projective_warp(img, img2, dest):
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])
    H1 = tf.ProjectiveTransform()
    H1.estimate(src, dest)
    warped = tf.warp(img2, H1.inverse, output_shape=(img.shape[0],img.shape[1]))
    mask = np.expand_dims(np.sum(warped,axis=2)==0,axis=2).astype(np.int)
    combined = warped + img*mask
    return combined

###Problem 1, calls projective warp on 1 image, then calls it again to replace second region.
def region_placement(img, img2):
    plt.close('all')   #close any opened plots
    fig, ax = plt.subplots(figsize=(12,10))   #prepare objects to demonstrate photos
    ax.imshow(img)  #display photo
    fig.suptitle('Original image', fontsize=14) #place title on photo
    # Input the source points
    print("Click four source points")
    first_region = np.asarray(plt.ginput(n=4))
    print("Click four source points")
    second_region = np.asarray(plt.ginput(n=4)) 
    first_region_replaced = projective_warp(img, img2, first_region) #We call projective warp once to project img2 onto img
    first_and_second_regions_replaced = projective_warp(first_region_replaced, img2, second_region) #Then we call projective warp again to project img2 onto the altered version of img
    fig, ax = plt.subplots(ncols=1, figsize=(12, 10))
    ax.imshow(first_and_second_regions_replaced) #We show the final image before returning it
    fig.suptitle('Final image creation', fontsize=14)
    return first_and_second_regions_replaced

##### Problem 2 ########
def stitching(img1, img2): #After speaking with Dr. Fuentes, he helped me understand how to warp both images to a center point instead of warping one image to the other; conceptually this made a lot more sense to me. This function follows that method.
    fig, ax = plt.subplots(ncols = 2, figsize=(12,10))
    ax[0].imshow(img1)
    ax[1].imshow(img2) #We display both images to get points from the user.
    fig.suptitle('Original image', fontsize=14)
    # Input the source points
    print("Click four source points")
    points = np.asarray(plt.ginput(n=8, timeout=60))
    img1_points = points[::2]            #Please note that the program assumes that the source points will be selected at even indices, so that means the first point selected, the third, and so on belong to the source. While the points with the odd indices belong to the dest image, so please make sure to click in this order.
    img2_points = points[1::2]
    target_points = (img1_points+img2_points)/2    #We create the center point that we will warp both images to.
    H0 = tf.ProjectiveTransform()
    H0.estimate(img1_points, target_points)    #We create a homography from the img1 points to the target points to see how much the offset will be using the coordinate map.
    coords0 = tf.warp_coords(H0, img1.shape)    
    H0.estimate(img2_points, target_points)    #We do the same for the img2 points to the target points.
    coords1 = tf.warp_coords(H0, img2.shape)
    #Extract corners from coordinate arrays
    corners0 = coords0[:2,[0,0,-1,-1],[0,-1,-1,0],0]
    corners1 = coords1[:2,[0,0,-1,-1],[0,-1,-1,0],0]
    corners = np.hstack((corners0, corners1))   #We get the corners of both warps.
    offset = -np.min(corners,axis=1)[::-1]      #We compute the offset here.
    size = (np.max(corners,axis=1)-np.min(corners,axis=1)+0.5).astype(np.int) #We compute the final size.
    target_points = target_points + offset  #We add the offset to the target points
    H0.estimate(img1_points, target_points) #We create another homography from the source to the target points, this time, since they are offset, it should fit within the bounds of the image.
    warped0 = tf.warp(img1, H0.inverse, output_shape = size)
    H0.estimate(img2_points, target_points)  #We do the same for the other image, since the offset is in place, it should also fit within the bounds of the image.
    warped1 = tf.warp(img2, H0.inverse, output_shape = size)
    return np.maximum(warped1, warped0) #We return the max because in the areas of overlap, this will result in a less blurry image as opposed to an averaged image. Since we warp both images, the results for both are of the same size.
##### Problem 2 ########
    

##### Problem 3 ########
def greenscreening(img, img2):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12,10))
    ax.imshow(img)
    fig.suptitle('Original image', fontsize=14) #We show the image so the user can click the green screen area
    print('select points where green screen is located')
    dest = np.asarray(plt.ginput(n=4))
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.array([[img2.shape[1],img2.shape[0]]])
    H0 = tf.ProjectiveTransform()
    H0.estimate(src, dest)
    warped = tf.warp(img2, H0.inverse, output_shape=(img.shape[0],img.shape[1])) #We warp the source image to the dest image
    green_indices = color_index(img, 1)  #find the green indices of the img to create the green screen mask
    green_indices = np.where(green_indices>=0.20, 1, 0) #I found 0.20 experimentally. #finalize the green screen mask with threshold
    for channel in range(3):
          warped[:,:,channel] = warped[:,:,channel]*green_indices #I chose to use the green_indices as a mask and apply it to every channel in the warped image to "remove" the sections where there should non-green sections.
    mask = np.expand_dims(np.sum(warped,axis=2)==0,axis=2).astype(np.int)
    combined = warped + img*mask
    fig, ax = plt.subplots(ncols=1, figsize=(12, 10))
    ax.imshow(combined)
    fig.suptitle('Final image creation/', fontsize=14)
    return combined

    
if __name__ == '__main__':
    ##### Uncomment for Problem 1 Results ########
    img = mpimg.imread('mobile_billboard.jpg')
    img = img/np.amax(img)
    img2 = mpimg.imread('utep720.jpg')
    img2 = img2/np.amax(img2)
    gauss_filter = gaussian_filter(15, 9)
    img2 = correlate2d_scipy(img2,gauss_filter)  #We apply a gaussian filter before placing the region.
    region_placement(img, img2)
    ##### Uncomment for Problem 1 Results ########

    ##### Uncomment for Problem 2 Results ########    
    # img = mpimg.imread('WF01.jpg')
    # img = img/np.amax(img)
    # img2 = mpimg.imread('WF02.jpg')
    # img2 = img2/np.amax(img2)
    # img3 = mpimg.imread('WF03.jpg')
    # img3 = img3/np.amax(img3)
    # intermediate_photo = stitching(img, img2)   #We apply the stitching function to the first two photos
    # final_photo = stitching(img3, intermediate_photo)   #Then we apply the stitchign function to the result of the intermediate photo and the third photo we want to stitch.
    # fig, ax = plt.subplots(figsize=(12,4))
    # ax.imshow(final_photo)
    ##### Uncomment for Problem 2 Results ########
    
    ##### Uncomment for Problem 3 Results ########
    # img = mpimg.imread('soto.jpg')
    # img = img/np.amax(img)
    # img2 = mpimg.imread('vll_utep_720.jpg')
    # img2 = img2/np.amax(img2)
    # gauss_filter = gaussian_filter(15, 9)
    # img2 = correlate2d_scipy(img2,gauss_filter)  #Apply a gaussian filter before passing img2 to greescreen function
    # greenscreening(img, img2)
    ##### Uncomment for Problem 3 Results ########