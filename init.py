import numpy as np
import glob,os
from multiprocessing import Process, Pool, active_children
import scipy
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from itertools import repeat
import matplotlib.pyplot as plt

c = 2.998e10
G = 6.672e-8
h = 6.626e-27
kB = 1.381e-16
Msun = 1.989e33
pc2cm = 3.086e18