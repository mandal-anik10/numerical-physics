'''
This script contains some usefull function used in the simulation scripts.

Author : Anik Mandal
'''

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

# Dimension of the Vector:
def V_Dim(v1):
    '''
    Returns the dimension of the vector

    -----Inputs----------
    v1 : array-like : input vector
    -----Output----------
    D : int : dimension of the input vector
    '''
    return len(v1)

# Vector Sum:
def V_Sum(v1, v2):
    '''
    Returns sum of two vectors

    -----Inputs----------
    v1 : array-like : first vector
    v2 : array-like : second vector
    -----Output----------
    vs : array-like : sum of the two vectors
    '''
    if V_Dim(v1) == V_Dim(v2):
        vs = np.zeros((V_Dim(v1), 1))
        for i in range(V_Dim(v1)):
            vs[i][0] = v1[i][0] + v2[i][0]
        return vs
    else:
        raise ValueError('Dimension of the vectors doesn\'t match')


# Vector Negative:
def V_Neg(v1):
    '''
    Returns neg vector of a vector

    -----Inputs----------
    v1 : array-like : input vector
    -----Output----------
    vn : array-like : neg vector of the input vector
    '''

    vn = np.zeros((V_Dim(v1), 1))
    for i in range(V_Dim(v1)):
        vn[i][0] = -v1[i][0]
    return vn


# Vector Substraction:
def V_Subtract(v1, v2):
    '''
    Returns subtraction of two vectors

    -----Inputs----------
    v1 : array-like : first vector
    v2 : array-like : second vector
    -----Output----------
    vs : array-like : subtraction of the two vectors
    '''
    vn = V_Sum(v1, V_Neg(v2))
    return vn


# Vector Dot Product:
def V_Dot(v1, v2):
    '''
    Returns scaler product of two vectors

    -----Inputs----------
    v1 : array-like : first vector
    v2 : array-like : second vector
    -----Output----------
    vs : array-like : scaler product of the two vectors
    '''
    if V_Dim(v1) == V_Dim(v2):
        vs = 0
        for i in range(V_Dim(v1)):
            vs = vs + v1[i][0] * v2[i][0]
        return vs
    else:
        raise ValueError('Dimension of the vectors doesn\'t match')


# Vector Scaler Product:
def V_Scale(v1, f):
    '''
    Returns scaled vector of a vector

    -----Inputs----------
    v1 : array-like : input vector
    f: floar : scaling factor
    -----Output----------
    vs : array-like : scaled vector of the input vector
    '''
    vs = np.zeros((V_Dim(v1), 1))
    for i in range(V_Dim(v1)):
        vs[i][0] = f * v1[i][0]
    return vs


# Vector Mod:
def V_Mod(v1):
    '''
    Returns length/modulus of a vector

    -----Inputs----------
    v1 : array-like : input vector
    -----Output----------
    vm : float : length of the vector
    '''
    vm = (V_Dot(v1, v1))**0.5
    return vm


# Unit Vector:
def V_Unit(v1):
    '''
    Returns length/modulus of a vector

    -----Inputs----------
    v1 : array-like : input vector
    -----Output----------
    vu : float : length of the vector
    '''
    vu = np.zeros((V_Dim(v1), 1))
    for i in range(V_Dim(v1)):
        vu[i][0] = v1[i][0]/V_Mod(v1)
    return vu


# Nolinear Fit Module:
def LassoRegression(x_data, y_data, max_degrees=10, alpha_value=0.2, num_points=int(1e5)):
    '''
    Returns data set of best fit curve using sklearn Lasso Regression

    -----Inputs----------
    x_data : list : x coordinates of the input values
    y_data : list : y coordinates of the input values
    max_degree : int : maximum degree for interpolation
        - default : 10
    alpha_value : float : defines alpha parameter for Lasso regression
        - default : 0.2
    num_points : int : number of point for interpolation
        - default : 1e5 
    -----Outputs----------
    x_new : list : x coordinates of the fitted curve
    y_new : list : y coordinates of the fitted curve
    '''  
    d = max_degrees
    poly = PolynomialFeatures(degree=d, include_bias=False)
    x_new = poly.fit_transform(x_data)

    alp = alpha_value
    sl = Lasso(alpha=alp).fit(x_new, y_data)

    m = sl.coef_
    c = sl.intercept_
    s = sl.score(x_new, y_data)

    x_new = np.linspace(min(x_data), max(x_data), num_points)
    y_new = []
    for i in range(len(x_new)):
        a = 0
        for j in range(len(m)):
            a = a + m[j] * x_new[i] ** (j + 1)
        y_new.append(a + c[0])
    return x_new, y_new
