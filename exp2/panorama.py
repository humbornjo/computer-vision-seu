"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above.

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    dx_2 = dx ** 2  #dx squared
    dy_2 = dy ** 2  #dy squared
    dxdy = dx * dy  #dx*dy
    # get M[0][0], M[0][1], M[1][0], M[1][1] in the image scale.
    M_dx_2=convolve(dx_2,window)
    M_dy_2=convolve(dy_2,window)
    M_dxdy=convolve(dxdy,window)
    M = np.zeros((2,2))
    # beg for result
    for i in range(H):
        for j in range(W):
            M=np.array(([M_dx_2[i][j],M_dxdy[i][j]],[M_dxdy[i][j],M_dy_2[i][j]]))
            R = np.linalg.det(M) - k * (np.trace(M)**2)
            response[i][j] = R
    pass
    ### END YOUR CODE

    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        If a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    feature = []
    ### YOUR CODE HERE
    # According to Hint
    denominator=np.std(patch)
    if denominator==0:
        denominator=1
    feature = ((patch - np.mean(patch))/denominator).flatten()
    pass
    ### END YOUR CODE
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    N = desc1.shape[0]
    dists = cdist(desc1, desc2)

    ### YOUR CODE HERE
    # get sorted matrix
    sorted=np.sort(dists,axis=1)
    for i in range(N):
        # get first two closest distances
        closest = sorted[i, 0]
        second = sorted[i, 1]
        # append into matches if meet the demand
        if(closest < threshold * second):    
            matches.append([i, np.argmin(dists[i])])
    matches = np.array(matches)
    pass
    ### END YOUR CODE

    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1

    Hint:
        You can use np.linalg.lstsq function to solve the problem.

    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)

    Return:
        H: a matrix of shape (P, P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)
    ### YOUR CODE HERE
    H = np.linalg.lstsq(p2, p1,rcond=None)[0]
    pass
    ### END YOUR CODE

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    # avoid double pad, make the input matrix N by 2 instead of N by 3
    pts1 = keypoints1[matches[:,0]]
    pts2 = keypoints2[matches[:,1]]
    for _ in range(n_iters):
        # randomly take n_sample matches
        index = np.random.choice(N, n_samples, replace=False)
        # get sample match pairs
        p1 = pts1[index]
        p2 = pts2[index]
        # get fit affine matrix H using the function above
        H = fit_affine_matrix(p1,p2)
        # check inlier matches
        inliers = np.sum((matched2@H - matched1) ** 2,axis=1) < threshold   
        current_n_inlier = np.sum(inliers)
        # update if current num inlier matches exceed the one before 
        if current_n_inlier > n_inliers:        
            max_inliers = inliers.copy()
            n_inliers = current_n_inlier
    # update for the final fit affine matrix H
    H = fit_affine_matrix(pts1[max_inliers], pts2[max_inliers])
    pass
    ### END YOUR CODE
    print(H)
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # Compute histogram per cell
    ### YOUR CODE HERE
    for i in range(rows):
        for j in range(cols):
            for m in range(pixels_per_cell[0]):
                for n in range(pixels_per_cell[1]):
                    # compute which bin this pixel belongs to
                    bin = int(theta_cells[i, j, m, n] // degrees_per_bin)
                    r=int(theta_cells[i, j, m, n] % degrees_per_bin)
                    # take 180 into consideration
                    # add up into corresponding bin by weight
                    cells[i,j,bin % n_bins] += (degrees_per_bin - r) / degrees_per_bin* G_cells[i, j, m, n]
                    cells[i,j,(bin+1) % n_bins] += r / degrees_per_bin * G_cells[i, j, m, n]
    # Flatten block of histograms into a 1D feature vector
    block = cells.flatten()
    # Normalize flattened block
    block = (block - np.mean(block)) / np.std(block)
    pass
    ### YOUR CODE HERE

    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    right_margin = out_W - np.argmax(np.max(np.fliplr(img1_mask),axis=0).reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(np.max(img2_mask,axis=0).reshape(1, out_W), 1)[0]
    ### YOUR CODE HERE
    img1_warped[~img1_mask] = 0
    img2_warped[~img2_mask] = 0
    mask1 = np.array(img1_mask,dtype=np.float64)
    mask2 = np.array(img2_mask,dtype=np.float64)
    # Define a weight matrices for img1_warped and img2_warped
    mask1[:,left_margin:right_margin] = np.tile(np.linspace(1, 0, right_margin-left_margin), (out_H, 1))
    mask2[:,left_margin:right_margin] = np.tile(np.linspace(0, 1, right_margin-left_margin), (out_H, 1))
    # Apply the weight matrices to their corresponding images
    img1 = mask1*img1_warped
    img2 = mask2*img2_warped
    # Combine the images
    merged = img1+img2
    pass
    ### END YOUR CODE

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
    # Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
    # Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    ### YOUR CODE HERE
    imgs_len=len(imgs)
    # get H matrix for each image pair
    Hs=[]
    rbmatches=[]
    for i in range(imgs_len-1):
        H,mtchs = ransac(keypoints[i],keypoints[i+1],matches[i])
        Hs.append(H)
        rbmatches.append(mtchs)

    # get reference middle image
    ref_index=(imgs_len-1)//2
    copy_imgs=imgs.copy()
    ref_img=copy_imgs.pop(ref_index)
    rest_imgs=copy_imgs

    # get the H_gt matrix for stitching the images
    H_gt=[]
    H_piror=np.eye(3)
    for i in range(ref_index):
        H_gt.insert(0,H_piror @ np.linalg.inv(Hs[ref_index-i-1]))
        H_piror=H_piror @ np.linalg.inv(Hs[ref_index-i-1])
    H_piror=np.eye(3)
    for i in range(ref_index,imgs_len-1):
        H_gt.append(H_piror @ Hs[i])
        H_piror=H_piror @ Hs[i]
    # get universial output_shape, offset to wrap each image
    output_shape, offset = get_output_space(ref_img, rest_imgs, H_gt)

    # prepare for the warped reference middle.  
    panorama=warp_image(ref_img, np.eye(3), output_shape, offset)
    # there is 'img1_mask = (img1_warped != 0)' in linear_blend(img1_warped, img2_warped) 
    # instead of 'img1_mask = (img1_warped != -1)' -1 would cause error when stitching multiple images
    mask = (panorama != -1)
    panorama[~mask] = 0

    # same process as above, deal with the image before the reference image first, after later.
    for i in range(ref_index):
        warped=warp_image(rest_imgs[ref_index-i-1], H_gt[ref_index-i-1], output_shape, offset)
        mask = (warped != -1)
        warped[~mask] = 0
        panorama = linear_blend(warped,panorama)
    for i in range(ref_index,imgs_len-1):
        warped=warp_image(rest_imgs[i], H_gt[i], output_shape, offset)
        mask = (warped != -1)
        warped[~mask] = 0
        panorama = linear_blend(panorama,warped)
    pass
    ### END YOUR CODE

    return panorama
