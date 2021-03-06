#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:15:17 2017

@author: sss
"""

import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print('x:\n{}'.format(x))



from scipy import sparse

# Create a 2D Numpy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print('Numpy array:\n{}'.format(eye))



# Convert the Numpy array to Scipy sparse matrix in CSR format
# Only the nonzero entries are stored
sparse_matrix = sparse.csr_matrix(eye)
print('\nScipy sparse CSR matrix:\n{}'.format(sparse_matrix))



data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO representation:\n{}'.format(eye_coo))



#%matplotlib inline

import matplotlib.pyplot as plt

# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
# Create a second array using sine
y = np.sin(x)
# The plot function makes a line chart of one array againse another
plt.plot(x, y, marker='x')



import pandas as pd

# Create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [24, 13, 53, 33]
       }

data_pandas = pd.DataFrame(data)
# IPython.display allows "pretty printing" of dataframes
# in the Jupyter notebook
#display(data_pandas)



import sys
print('Python version: {}'.format(sys.version))

print('pandas version: {}'.format(pd.__version__))

import matplotlib
print('matplotlib version: {}'.format(matplotlib.__version__))

print('Numpy version: {}'.format(np.__version__))

import scipy
print('Scipy version: {}'.format(scipy.__version__))

import IPython
print('IPython version: {}'.format(IPython.__version__))

import sklearn
print('scikit-learn version: {}'.format(sklearn.__version__))
