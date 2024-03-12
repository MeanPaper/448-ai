'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    m = 0
    n = 0
    Pjoint = []
    total_N = len(texts)

    for sub in texts:
      m = max(m, sub.count(word0)) # rows
      n = max(n, sub.count(word1)) # cols
    
    Pjoint = np.zeros((m+1,n+1))
    for sub in texts:
      m = sub.count(word0) 
      n = sub.count(word1)
      Pjoint[m,n] += (1/total_N)
      
    # print(m, n)
    # raise RuntimeError('You need to write this part!')
    return Pjoint

def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
    '''
    if index == 0:
      Pmarginal = np.sum(Pjoint, axis = 1)
    if index == 1:
      Pmarginal = np.sum(Pjoint, axis = 0)

    # raise RuntimeError('You need to write this part!')
    return Pmarginal
    
def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)
    
    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    # raise RuntimeError('You need to write this part!')
    Pcond = Pjoint
    for i in range(len(Pmarginal)):
      Pcond[i] = np.nan if Pmarginal[i] == 0 else Pcond[i] / Pmarginal[i] 

    return Pcond

def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    mu (float) - the mean of X
    '''
    
    mu = 0
    for i in range(len(P)):
      mu += i * P[i]
    # raise RuntimeError('You need to write this part!')
    return mu

def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)
    
    Outputs:
    var (float) - the variance of X
    '''
    var = 0
    mu = mean_from_distribution(P)
    for i in range(len(P)):
      var += P[i] * ((i - mu)**2)
    # raise RuntimeError('You need to write this part!')
    return var

def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)
    
    Outputs:
    covar (float) - the covariance of X0 and X1
    '''
    # raise RuntimeError('You need to write this part!')
    
    marginal_x0 = marginal_distribution_of_word_counts(P, 0) 
    marginal_x1 = marginal_distribution_of_word_counts(P, 1)
    mean_X0 = mean_from_distribution(marginal_x0)
    mean_X1 = mean_from_distribution(marginal_x1)
    covar = np.float32(0.0)
    for i in range(len(P)):
      for j in range(len(P[0])):
        covar += np.float32(P[i][j] * (i - mean_X0) * (j - mean_X1))
    return covar

def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0.0
    for i in range(len(P)):
      for j in range(len(P[0])):
        expected += P[i][j] * f(i,j)
    # raise RuntimeError('You need to write this part!')
    return expected
    
