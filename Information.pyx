# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:22:08 2019

@author: sparra
"""

# from cpython.array cimport array

# def summer(double[:] mv):
    # """Sums its argument's contents."""
    # cdef double d, ss = 0.0
    # for d in mv:
        # ss += d
    # return ss
    
    
    
#-------------------------------------------------------------------------------
from cpython.array cimport array
def Mutual_information(double[:, :] PrIs, int nrrn, int ncol, double[:] Ps):
    """
    Program written by Sergio Parra S\'anchez. 
    This function calculates the mutual information of the 2 distributions:
    *2 PrIs  Probability of r given s have ocurred
    *3 Ps    Probability of the stimulus
    
    
    This function calculates the mutual information according to the following
    expression:
        I=\sum_{r} \sum_{s} {  P(s)*P(r|s)*log2(   P(r|s)/(P(r) )   }
    
    Pr must be an array with two columns and multiple rows
    PrIs  must be a matrix where the columns are sorted in the same
    order than Ps
    
    """
    from numpy import log2
    cdef double I=0.0, Pr #Initialize the value of mutual information
    cdef int stim, r, stim2
    for stim in range(ncol): # sum over stimuli
        for r in range(nrrn): #Sum over any neural-response-metric outcome
                 Pr=0
                 for stim2 in range(ncol):
                     Pr+=PrIs[r, stim2]*Ps[stim2]  #Probabilidad marginal.
                 if Pr>0 and PrIs[r, stim]>0:# By convention zero develops zero
                     I+=Ps[stim]*PrIs[r, stim]*log2(PrIs[r, stim]/Pr)
    return I
    
    
    
def entropy(double[:] Pr, int nr):
    """
    Program written by Sergio Parra S\'anchez. 
    This function calculates the entropy of a distribution:
    *1 Pr    Probability of r observed
    *2 nr    Number of elements in vector
    
    
    This function calculates the mutual information according to the following
    expression:
        I=\sum_r {  P(r)*log2(   1/P(r))    }
    
    Pr must be an array
    
    """
    from numpy import log2, dot
    cdef double H=0.0 #Initialize the value of mutual information
    cdef int r
    #mask=Pr>0 #Elaborates a mask to select Pr>0
    #H=dot(Pr[mask], log2(Pr[mask]))
    for r in range(nr):
        if Pr[r]>0:
            H+=Pr[r]*log2(Pr[r])
    H*=-1
    return H             



def conditional_entropy(double[:] Ps, double[:, :] PrIs, int nstim, int nr):
    """
        # Program written by Sergio Parra S\'anchez
        This function calculates the conditional entropy between 
        two variables R, S, the function calculates conditional entropy 
        by using the following equation:
        H(R|S)=-\sum_{j} {  p(s_j)\sum_{i}{ p(r_i|s_j)*log_2(p(r_i|s_j))    }    }
        
        The inputs are:
        * Ps a vector with the stimulus probabilities
        * PrIs a matrix where the rows represent the probability of any response given the
            stimulus in the column. Then PrIs is a matrix with the same columns as 
            elements in Ps and must be in the same order too. This is, the information in the ith column must 
            correspond to the same stimuli in the ith element of the array Ps
        * nr is the number of elements in nr
        * nstim is the number of columns in PrIs matrix
    """ 
    from numpy import log2 
    cdef double HRIS=0.0, temp=0.0
    cdef int rows, columns
    for columns in range(nstim):
        for rows in range(nr):
            if PrIs[rows][columns]>0:
                temp+=PrIs[rows][columns]*log2(PrIs[rows][columns])
        HRIS+=temp*Ps[columns]
        temp=0
    return -1*HRIS
    
