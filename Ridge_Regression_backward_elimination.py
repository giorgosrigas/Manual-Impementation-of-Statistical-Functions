# Q2
# Student number:
import numpy as np
import statsmodels.api as sm
from scipy import stats
import pandas as pd


def ridge(y,X,a):
    """
    function which receives two arrays and a regularization value
    as inputs and implements ridge regression.
    
    :param y: a response vector with shape (n,1)
    :param X: an array of predictors with shape (n,p)
    :param a: a single number or a list of numbers which represent the regularization strength

    
    :return: returns a number or a list of numbers if a list has been provided for a
    which are the beta coefficient vectors.
    """
    n = X.shape[0]
    y.shape = n,1 # define shape of y
    x_ones = np.hstack((X, np.ones((X.shape[0], 1), dtype = X.dtype))) # append collumn with intercepts
    transp = x_ones.T # matrix transpose
    try:
        # ridge regression implementation if the a argument is a list
        inverse = (np.linalg.inv(transp @ x_ones + [np.identity(x_ones.shape[1]) * scalar for scalar in a])) # inverse matrix 
        B_list = inverse @ transp @ y
        return B_list
    except:
        # implementation of ridge regression if the argument a is a single number
        inverse1 = (np.linalg.inv(transp @ x_ones + np.identity(x_ones.shape[1]) * a )) # inverse matrix
        B = inverse1 @ transp @ y
    return B


def remove_variables(y, X,  threshold , variable_names):
    """
    function which receives a response and outcome array, a  threshold  and a list of variable names
    ,performs regression tests and eliminates a variable with p value higher than the p argument
    after each test untill no variable p value is higher than the threshold p.
    
    :param y:  1-D array
    :param X: an array of predictors with shape (n,p)
    :param p: a threshold p value
    :param variable_names: a list of column names or None
    
    :return: returns a the matrix of columns after the non-signifficant ones
    have been eliminated and their names as a tuple.
    """
    # convert input numpy arrays to pandas dataframe for manipulation
    df=pd.DataFrame(data=X[0:,0:],
        index=[i for i in range(X.shape[0])],
        columns=['x' + str(i+1) if variable_names == None else variable_names[i] for i in range(X.shape[1])])
    df2=pd.DataFrame(data=y[0:,0:])
    # add of intercept
    df["intercept"] = 1
    # list with column names for later iteration over those columns
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]  
    # iteration over the pandas dataframes  to implement the linear regression 
    for i in range(df.shape[1]):
        model1 = sm.OLS(df2, df).fit()
    
    # determine the highest p value excluding the one of intercept
        maxPval = max(model1.pvalues[1:], default=0)
        cols = df.columns.tolist()

        # delete the column-predictor with the highest p value and 
        # repeating this process until no further predictor's p value is higher than the threshold p.  
        if maxPval >= threshold:
            for j in cols:
                if j != 'intercept':
                    if (model1.pvalues[j] == maxPval):
                        #if j != 'intercept':
                        del df[j]
        else:
            break # end of loop
        
    if len(cols[1:]) == 0:
        return (None, None)
    else:
        # turn pandas dataframe into numpy array and drop the intercept column
        new_x = df.to_numpy()
        new_X = new_x[:,1:] #exclude intercept
        new_variable_names = cols[1:]      
        return (new_X, new_variable_names)

