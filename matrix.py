import numpy as np
from scipy.linalg import sqrtm



def shape_errors(A, B):
    if A.shape != B.shape:
        raise ValueError(f'Matrices must have the same shape. Received {A.shape} and {B.shape}.')
    elif A.ndim < 2:
        raise ValueError(f'Matrices must have at least two dimensions. Received {A.ndim}.')
    elif A.ndim > 3:
        raise ValueError(f'Matrices must have at most three dimensions. Received {A.ndim}.')
    else:
        return True


def frobenius_norm(A, B):
    if not shape_errors(A, B):
        return -1
    diff = A - B
    if A.ndim == 2:
        return np.linalg.norm(diff)
    elif A.ndim == 3:
        return np.linalg.norm(diff, axis=(-2, -1))
    

def airm(A, B):
    if not shape_errors(A, B):
        return -1
    if A.ndim == 2:
        root =  sqrtm(A)
        return np.linalg.norm(np.log(root @ B @ root), axis=0)
    elif A.ndim == 3:
        result = np.zeros(A.shape[0])
        for i, (a, b) in enumerate(zip(A, B)):
            root = sqrtm(a)
            result[i] = np.linalg.norm(np.log(root @ b @ root))
        return result
    

def riemannian_metric_spd(A, B):
    if not shape_errors(A, B):
        return -1
    if A.ndim == 2:
        root = sqrtm(A)
        return np.sqrt(np.trace(A) + np.trace(B) - 2*np.trace(sqrtm(root @ B @ root)))
    elif A.ndim == 3:
        result = np.zeros(A.shape[0])
        for i, (a, b) in enumerate(zip(A, B)):
            root = sqrtm(a)
            result[i] = np.sqrt(np.trace(a) + np.trace(b) - 2*np.trace(sqrtm(root @ b @ root)))
        return result
    

def kappa(X):
    if X.ndim < 2:
        raise ValueError(f'Matrix must have at least two dimensions. Received {X.ndim}.')
    if X.ndim > 3:
        raise ValueError(f'Matrices must have at most three dimensions. Received {X.ndim}.')
    if X.ndim == 2:
        term1 = np.outer(np.diag(X), np.ones(X.shape[0]).T)
        term2 = np.outer(np.ones(X.shape[0]), np.diag(X).T)
        return term1 + term2 - 2*X
    if X.ndim == 3:
        result = np.zeros(X.shape)
        for i, x in enumerate(X):
            term1 = np.outer(np.diag(x), np.ones(x.shape[0]).T)
            term2 = np.outer(np.ones(x.shape[0]), np.diag(x).T)
            result[i] = term1 + term2 - 2*x
        return result


def moving_average(data, window_size):
    n_samples = len(data)
    n_windows = n_samples - window_size + 1
    matrix_shape = data[0].shape
    
    results = np.zeros((n_windows, *matrix_shape))
    for i in range(n_windows):
        window = data[i:i+window_size]
        results[i] = np.mean(window, axis=0)
    
    return results


def ponderated_difference(data, distance, k, alpha):
    n_samples = len(data)

    # Set increasing weigths adding up to 1
    ks = np.arange(1, k+1, 1) / (k *(k+1))
    weigths = (1/alpha)**(ks-1) / np.sum((1/alpha)**(ks-1))
    # print(weigths)
    # print(np.sum(weigths))
    
    results = np.zeros(n_samples)
    for i in range(n_samples):
        if i < k:
            results[i] = float('NaN')
        else:
            for j in range(k):
                results[i] += weigths[j] * distance(data[i], data[i-j])
    
    return results


if __name__=='__main__':

    # Create a simple test dataset: a list of 1D arrays with increasing integers
    data = [np.array([[1, 2], [3, 4]])*i for i in range(10)]
    k = 50

    result = ponderated_difference(data, riemannian_metric_spd, k, 10e-32)

    print("Input data:")
    print(np.array(data))
    print("\nResult of ponderated_difference:")
    print(result)
