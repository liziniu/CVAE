
import numpy as np

def create_lag_matrix(F, lags):
    """
    Create a matrix of sequences lagged 1 step.

    Parameters
    ----------
    F : NxD np array
        A sequence (or time series) of data length N with D dimensions.
    lags : int
        Number of lags to create
    """
    if isinstance(F, list) or F.ndim==1:
        F = np.c_[F] # Convert to 2d array
    print F
    print range(F.shape[1])
    return create_lag_matrix_complex(F, range(F.shape[1]), [], lags)

def create_lag_matrix_complex(F, S_idx, U_idx, lags, seq_id=None, return_row_ids=False,
                            diff_output=False):
    """
    Create a matrix of sequences lagged 1 step.

    The difference between the S_idx and U_idx is that the samples given by the
    U_idx are not incluede in the last step.

    Parameters
    ----------
    F : NxD numpy array
        A matrix with N samples and D dimensions/features/variables. The sample
        are assumed to be sequence in time (it not split up by using seq_id).
    S_idx : list
        List of indexes of colums to use in the matrix F.
    U_idx : list
        List of indexes of colums to use in the matrix F.
    lags : int
        Number of lags to create
    seq_id : 1d numpy array, optional
        A list of numbers that identify sequences. If the ids for the rows
        are the same, then the are consequtive. E.g. [1, 1, 1, 2, 2] says that
        there are two sequences F.

    Returns
    -------
    res : np array
        [ S(i) U(i) S(i+1) U(i+1) ... S(i+d-1) U(i+d-1)     S(i+d) ]    
    """
    lx = len(S_idx)
    lu = len(U_idx)
    A_idx = S_idx + U_idx
    la = len(A_idx)
    res = np.zeros((F.shape[0], (lx+lu)*lags+lx))
    row = 0
    row_id = []
    if seq_id==None:
        seq_id = np.ones(F.shape[0])
    # Create lag space matrix
    for seq in np.unique(seq_id):
        blk = F[seq==seq_id, :]
        for i in range(blk.shape[0]-lags):
            for j in range(lags):
                res[row, j*la:(j+1)*la] = blk[i+j, A_idx]
            if diff_output:
                res[row, lags*la:lags*la+lx] = blk[i+lags, S_idx] - blk[i+lags-1, S_idx]
            else:
                res[row, lags*la:lags*la+lx] = blk[i+lags, S_idx]
            row_id.append(seq)
            row += 1
    if return_row_ids==True:
        return (res[:row, :], row_id)
    else:
        return res[:row, :]

def test1():
    F = np.linspace(1,5,5)
    F = np.reshape(np.repeat(F,5), (5,5))+F/10
    R = create_lag_space_matrix(F,[0,1],[2,3],2,ones(5))
    CR = np.array([[ 1.1,  1.2,  1.3,  1.4,  2.1,  2.2,  2.3,  2.4,  3.1,  3.2],
                   [ 2.1,  2.2,  2.3,  2.4,  3.1,  3.2,  3.3,  3.4,  4.1,  4.2],
                   [ 3.1,  3.2,  3.3,  3.4,  4.1,  4.2,  4.3,  4.4,  5.1,  5.2]])
    if np.sum(abs(R-CR))>10e-7:
        print "Error"

def test2():
    F = np.linspace(1,5,5)
    F = np.reshape(np.repeat(F,5), (5,5))+F/10
    R = create_lag_space_matrix(F,[0,1],[2,3],1,[1,1,2,2,2]) # Sequence indices
    CR = array([[ 1.1,  1.2,  1.3,  1.4,  2.1,  2.2],
                [ 3.1,  3.2,  3.3,  3.4,  4.1,  4.2],
                [ 4.1,  4.2,  4.3,  4.4,  5.1,  5.2]])
    if np.sum(abs(R-CR))>10e-7:
        print "Error"

