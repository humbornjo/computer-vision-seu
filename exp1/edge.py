"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    #kernel完全对称，无需反转
    for i in range(Hi):
        for j in range(Wi):
            out[i][j]=(padded[i:i+Hk,j:j+Wk]*kernel).sum()
                   
    pass
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    #计算基本参数k，除二向下取整
    k=size//2
    #计算矩阵的上三角部分，再使用转置和减去对角部分完成矩阵减少浮点运算次数
    for i in range(size):
        for j in range(i,size):
            kernel[i][j]=np.power(i-k,2)+np.power(j-k,2)
    kernel += kernel.T - np.diag(kernel.diagonal())
    #使用lambda函数进行简洁的矩阵处理，使其成为高斯核
    param=2*np.power(sigma,2)
    gauss=lambda x:np.exp(-x/param)/(np.pi*param)
    kernel=gauss(kernel)
    
    pass
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    x=np.array([[0, 0, 0],
            [-0.5, 0, 0.5],
            [0, 0, 0]])
    out=conv(img,x)
    pass
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    y=np.array([[0, -0.5, 0],
            [0, 0, 0],
            [0, 0.5, 0]])
    out=conv(img,y)
    pass
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    G=np.sqrt(partial_x(img)**2+partial_y(img)**2)
    theta=np.arctan2(partial_y(img),partial_x(img))*180/np.pi
    #使用where函数将角度从[-180,180]变为[0,360]
    theta=np.where(theta>=0,theta,theta+360)
    pass
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    #print(theta)
    ### BEGIN YOUR CODE
    #对于边缘的像素点，只与不超出索引范围的像素点比较
    #对于可能大于索引范围的索引，使用try except进行处理
    #对于可能小于0的索引，由于-1变成了当前维度的最后一个值，所以使用负索引进行纠正
    sign=False
    for i in range(H):
         for j in range(W):
            neigh_1=0
            neigh_2=0
            if theta[i][j]%180==0:
                try:
                    neigh_1=G[i][j-1-W]
                except:
                    pass
                try:
                    neigh_2=G[i][j+1]
                except:
                    pass
            elif theta[i][j]%90==0:
                try:
                    neigh_1=G[i-1-H][j]
                except:
                    pass
                try:
                    neigh_2=G[i+1][j]
                except:
                    pass
            elif (theta[i][j]-45)%180==0:
                try:
                    neigh_1=G[i-1-H][j-1-W]
                except:
                    pass
                try:
                    neigh_2=G[i+1][j+1]
                except:
                    pass
            else:
                try:
                    neigh_1=G[i+1][j-1-W]
                except:
                    pass
                try:
                    neigh_2=G[i-1-H][j+1]
                except:
                    pass

            out[i][j]=G[i][j] if G[i][j]>=max(neigh_1,neigh_2) else 0
    pass
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    H, W = img.shape
    for i in range(H):
        for j in range(W):
            if img[i][j] >= high:
                strong_edges[i, j] = True
            elif img[i][j] >= low:
                weak_edges[i, j] = True
    pass
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    #每次进入循环，new列表置空，并记录新加入的weak点，同时将weak矩阵中的对应点置0，防止多余的比较
    #bool值可直接相加
    #每次循环更新indices为new，保证不使用已经检查过的强点，产生冗余
    #循环直至某次循环后new中的元素数为0
    while len(indices):
        new=[]
        for i,j in indices:
            edges[i][j]=True
            for ind in get_neighbors(i, j, H, W):
                if weak_edges[ind]==1:
                    edges[ind]+=weak_edges[ind]
                    weak_edges[ind]=0
                    new.append(ind)
        indices=new
    pass
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    #结合上述函数功能
    kernel = gaussian_kernel(kernel_size, sigma)
    img = conv(img, kernel)
    G, theta = gradient(img)
    G = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(G, high, low)
    edge = link_edges(strong_edges, weak_edges)
    pass
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    print(diag_len)
    for i in range(0, len(ys)):
        for j in range(num_thetas):
            rho = int(xs[i] * cos_t[j] + ys[i] * sin_t[j])
            #[-diag_len, diag_len]->[diag_len, 2*diag_len]
            std_rho = rho + diag_len
            accumulator[std_rho][j] += 1    
    pass
    ### END YOUR CODE

    return accumulator, rhos, thetas
