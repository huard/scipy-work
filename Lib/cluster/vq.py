"""*
Vector Quantization Module
    Provides several routines used in creating a code book from a set of
    observations and comparing a set of observations to a code book.
    
    All routines expect an "observation vector" to be stored in each
    row of the obs matrix.  Similarly the codes are stored row wise
    in the code book matrix.

    whiten(obs) -- 
        Normalize a group of observations on a per feature basis
    vq(obs,code_book,return_dist=0) -- 
        Calculate code book membership of obs
    kmeans(obs,k_or_guess,iter=20,thresh=1e-5) -- 
        Train a codebook for mimimum distortion using the kmeans algorithm
    
*"""
from Numeric import *
from RandomArray import randint
import scipy


def whiten(obs):
    """* Normalize a group of observations on a per feature basis

    Description:
        Before running kmeans algorithms, it is beneficial to "whiten", or scale,
        the observation data on a per feature basis.  This is done by dividing
        each feature by its standard deviation across all observations.
    Arguments:
        obs -- 2D array.
                Each row of the array is an observation.  The
                columns are the "features" seen during each observation

                          #   f0    f1    f2
                    obs = [[  1.,   1.,   1.],  #o0
                           [  2.,   2.,   2.],  #o1
                           [  3.,   3.,   3.],  #o2
                           [  4.,   4.,   4.]]) #o3

        XXX perhaps should have an axis variable here.
    Outputs:
        result -- 2D array.
                Contains the values in obs scaled by the standard devation of
                each column.

    Test:
        >>> features  = array([[  1.9,2.3,1.7],
        ...                    [  1.5,2.5,2.2],
        ...                    [  0.8,0.6,1.7,]])
        >>> whiten(features)
        array([[ 3.41250074,  2.20300046,  5.88897275],
               [ 2.69407953,  2.39456571,  7.62102355],
               [ 1.43684242,  0.57469577,  5.88897275]])

    *"""
    std_dev = scipy.stdev(obs,axis=0)
    return obs / std_dev

def vq(obs,code_book,return_dist=0):
    """* Vector Quantization: assign features sets to codes in a code book.

    Description:
        Vector quantization determines which code in the code book best
        represents an observation of a target.  The features of each
        observation are compared to each code in the book, and assigned the
        one closest to it.  The observations are contained in the obs array.
        These features should be "whitened," or nomalized by the standard
        deviation of all the features before being quantized.  The code book
        can be created using the kmeans algorithm or something similar.

    Arguments:
        obs -- 2D array.
                Each row of the array is an observation.  The
                columns are the "features" seen during each observation
                The features must be whitened first using the
                whiten function or something equivalent.
        code_book -- 2D array.
                The code book is usually generated using the kmeans
                algorithm.  Each row of the array holds a different
                code, and the columns are the features of the code.
                                #   c0    c1    c2   c3
                    code_book = [[  1.,   2.,   3.,   4.],  #f0
                                 [  1.,   2.,   3.,   4.],  #f1
                                 [  1.,   2.,   3.,   4.]]) #f2
    Outputs:
        result -- 1D array.
                If obs is a NxM array, then a length M array
                is returned that holds the selected code book index for
                each observation.
    Reference:
    Test:
        >>> code_book = array([[1.,1.,1.],
        ...                    [2.,2.,2.]])
        >>> features  = array([[  1.9,2.3,1.7],
        ...                    [  1.5,2.5,2.2],
        ...                    [  0.8,0.6,1.7]])
        >>> vq(features,code_book,return_dist=1)
        (array([1, 1, 0]), array([ 0.43588989,  0.73484692,  0.83066239]))

    *"""
    No,Nf = shape(obs) #No = observation count, Nf = feature count
    # code books and observations should have same number of features
    assert(Nf == code_book.shape[1])
    diff = obs[NewAxis,:,:]-code_book[:,NewAxis,:]
    dist = sqrt(sum(diff*diff,-1))
    code = argmin(dist,0)
    if return_dist:
        min_dist = minimum.reduce(dist,0) #the next line I think is equivalent - and should be faster
        #min_dist = choose(code,dist) # but in practice, didn't seem to make much difference.
        return code, min_dist
    else:
        return code

def kmeans_(obs,guess,thresh=1e-5):
    """* See kmeans
    Outputs:
        code_book -- the lowest distortion codebook found.
        avg_dist -- the average distance a observation is
                    from a code in the book.  Lower means
                    the code_book matches the data better.
    Test:
        Note: not whitened in this example.
        
        >>> features  = array([[ 1.9,2.3],
        ...                    [ 1.5,2.5],
        ...                    [ 0.8,0.6],
        ...                    [ 0.4,1.8],
        ...                    [ 1.0,1.0]])        
        >>> book = array((features[0],features[2]))
        >>> kmeans_(features,book)        
        (array([[ 1.7       ,  2.4       ],
               [ 0.73333333,  1.13333333]]), 0.40563916697728591)

    *"""
    code_book = array(guess,copy=1)
    Nc = code_book.shape[0]
    avg_dist=[]
    diff = thresh+1.
    while diff>thresh:
        #print diff
        #compute membership and distances between obs and code_book
        obs_code, distort = vq(obs,code_book,return_dist=1)
        avg_dist.append(scipy.mean(distort))
        #recalc code_book as centroids of associated obs
        if(diff > thresh):
            has_members = []
            for i in arange(Nc):
                cell_members = compress(equal(obs_code,i),obs,0)
                if cell_members.shape[0] > 0:
                    code_book[i] = scipy.mean(cell_members,0)
                    has_members.append(i)
            #remove code_books that didn't have any members
            #print has_members
            code_book = take(code_book,has_members,0)
        if len(avg_dist) > 1:
            diff = avg_dist[-2] - avg_dist[-1]
    #print avg_dist
    return code_book, avg_dist[-1]

def kmeans(obs,k_or_guess,iter=20,thresh=1e-5): 
    """* Generate a code book with minimum distortion
        
    Description:
            
    Arguments:
        obs -- 2D array
                Each row of the array is an observation.  The
                columns are the "features" seen during each observation
                The features must be whitened first using the 
                whiten function or something equivalent.                                             
    Outputs:
        result -- 1D array.
                If obs is a NxM array, then a length N array
                is returned that holds the selected code book index for 
                each observation.
        k_or_guess -- integer or 2D array.
            If integer, it is the number of code book elements.
            If a 2D array, the array is used as the intial guess for
            the code book.  The array should have k rows, and the
            same number of columns (features) as the obs array.
        iter -- integer.
            The number of times to restart the kmeans algorithm with
            a new initial guess.  If k_or_guess is a 2D array (codebook),
            this argument is ignored and only 1 iteration is run.           
        thresh -- float
            Terminate each kmeans run when the distortion change from 
            one iteration to the next is less than this value.  
    Reference:
    Test:
        ("Not checked carefully for accuracy..." he said sheepishly)
        
        >>> features  = array([[ 1.9,2.3],
        ...                    [ 1.5,2.5],
        ...                    [ 0.8,0.6],
        ...                    [ 0.4,1.8],
        ...                    [ 0.1,0.1],
        ...                    [ 0.2,1.8],
        ...                    [ 2.0,0.5],
        ...                    [ 0.3,1.5],
        ...                    [ 1.0,1.0]])        
        >>> whitened = whiten(features)
        >>> book = array((whitened[0],whitened[2]))
        >>> kmeans(whitened,book)        
        (array([[ 2.3110306 ,  2.86287398],
               [ 0.93218041,  1.24398691]]), 0.85684700941625547)
               
        >>> import RandomArray       
        >>> RandomArray.seed(1000,2000)
        >>> codes = 3
        >>> kmeans(whitened,codes)
        (array([[ 2.3110306 ,  2.86287398],
               [ 1.32544402,  0.65607529],
               [ 0.40782893,  2.02786907]]), 0.5196582527686241)
               
    *"""
    if int(iter) < 1:
        raise ValueError, 'iter must be >= to 1.'
    if type(k_or_guess) == type(array([])): 
        guess = k_or_guess
        result = kmeans_(obs,guess,thresh=thresh)
    else:
        best_dist = 100000 #initialize best distance value to a large value
        No = obs.shape[0]
        k = k_or_guess
        #print 'kmeans iter: ',
        for i in range(iter):   
            #print i,
            #the intial code book is randomly selected from observations
            guess = take(obs,randint(0,No,k),0) 
            book,dist = kmeans_(obs,guess,thresh=thresh)
            if dist < best_dist: 
                best_book = book
                best_dist = dist
        #print      
        result = best_book,best_dist
    return result               