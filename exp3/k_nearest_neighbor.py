import numpy as np
from numpy.core.defchararray import count


def compute_distances(X1, X2):
    """Compute the L2 distance between each point in X1 and each point in X2.
    It's possible to vectorize the computation entirely (i.e. not use any loop).

    Args:
        X1: numpy array of shape (M, D) normalized along axis=1
        X2: numpy array of shape (N, D) normalized along axis=1

    Returns:
        dists: numpy array of shape (M, N) containing the L2 distances.
    """
    M = X1.shape[0]
    N = X2.shape[0]
    assert X1.shape[1] == X2.shape[1]

    dists = np.zeros((M, N))

    # YOUR CODE HERE
    # Compute the L2 distance between all X1 features and X2 features.
    # Don't use any for loop, and store the result in dists.
    #
    # You should implement this function using only basic array operations;
    # in particular you should not use functions from scipy.
    #
    # HINT: Try to formulate the l2 distance using matrix multiplication

    # formulate the l2 distance to x^2-2xy+y^2
    # x^2 is the sum of square of the values in the same column.
    # a.k.a. the diagnoal elements of X1 * X1.T
    # y^2 the same
    # xy is the dot product of each column in X1 and X2.
    # which is X1 * X2.T
    # corresponding to the shape M by N
    mid=X1@X2.T
    # function np.diagnoal return vector, use np.newaxis to extend a dimension
    # with the broadcast mechanism within numpy to realize the reformulated l2 distance
    head=np.diagonal(X1@X1.T)[:,np.newaxis]
    tail=np.diagonal(X2@X2.T)[:,np.newaxis].T
    dists=head+tail-2*mid
    pass
    # END YOUR CODE

    assert dists.shape == (M, N), "dists should have shape (M, N), got %s" % dists.shape

    return dists


def predict_labels(dists, y_train, k=1):
    """Given a matrix of distances `dists` between test points and training points,
    predict a label for each test point based on the `k` nearest neighbors.

    Args:
        dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives
               the distance betwen the ith test point and the jth training point.

    Returns:
        y_pred: A numpy array of shape (num_test,) containing predicted labels for the
                test data, where y[i] is the predicted label for the test point X[i].
    """
    num_test, num_train = dists.shape
    y_pred = np.zeros(num_test, dtype=np.int)
    
    for i in range(num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        closest_y = []
        # Use the distance matrix to find the k nearest neighbors of the ith
        # testing point, and use self.y_train to find the labels of these
        # neighbors. Store these labels in closest_y.
        # Hint: Look up the function numpy.argsort.

        # Now that you have found the labels of the k nearest neighbors, you
        # need to find the most common label in the list closest_y of labels.
        # Store this label in y_pred[i]. Break ties by choosing the smaller
        # label.

        # YOUR CODE HERE

        # get k nearest neighbors of the ith test sample and get the class index using the input y_train
        for idx in np.argsort(dists[i])[:k]:
            closest_y.append(y_train[idx])
        # create a vector with the length of the number of classes. 
        count_list=np.zeros(np.max(y_train))
        # count the number of each class
        for clas in closest_y:
            count_list[clas-1]+=1
        # meet the demand 'Break ties by choosing the smaller label' by using np.argmax()
        y_pred[i]=np.argmax(count_list)+1
        pass
        # END YOUR CODE

    return y_pred


def split_folds(X_train, y_train, num_folds):
    """Split up the training data into `num_folds` folds.

    The goal of the functions is to return training sets (features and labels) along with
    corresponding validation sets. In each fold, the validation set will represent (1/num_folds)
    of the data while the training set represent (num_folds-1)/num_folds.
    If num_folds=5, this corresponds to a 80% / 20% split.

    For instance, if X_train = [0, 1, 2, 3, 4, 5], and we want three folds, the output will be:
        X_trains = [[2, 3, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 2, 3]]
        X_vals = [[0, 1],
                  [2, 3],
                  [4, 5]]

    Args:
        X_train: numpy array of shape (N, D) containing N examples with D features each
        y_train: numpy array of shape (N,) containing the label of each example
        num_folds: number of folds to split the data into

    jeturns:
        X_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds, D)
        y_trains: numpy array of shape (num_folds, train_size * (num_folds-1) / num_folds)
        X_vals: numpy array of shape (num_folds, train_size / num_folds, D)
        y_vals: numpy array of shape (num_folds, train_size / num_folds)

    """
    assert X_train.shape[0] == y_train.shape[0]

    validation_size = X_train.shape[0] // num_folds
    training_size = X_train.shape[0] - validation_size

    X_trains = np.zeros((num_folds, training_size, X_train.shape[1]))
    y_trains = np.zeros((num_folds, training_size), dtype=np.int)
    X_vals = np.zeros((num_folds, validation_size, X_train.shape[1]))
    y_vals = np.zeros((num_folds, validation_size), dtype=np.int)

    # YOUR CODE HERE
    # Hint: You can use the numpy array_split function.

    # split the samples in 5
    X_folds=np.array_split(X_train,5)
    for i in range(num_folds):
        # default concatenate the matrix in axis=0 direction
        X_trains[i,:]=np.concatenate(X_folds[:i]+X_folds[i+1:])
        X_vals[i,:]=X_folds[i]
    y_folds=np.array_split(y_train,5)
    for i in range(num_folds):
        y_trains[i,:]=np.concatenate(y_folds[:i]+y_folds[i+1:])
        y_vals[i,:]=y_folds[i]
    pass
    # END YOUR CODE

    return X_trains, y_trains, X_vals, y_vals
