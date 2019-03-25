#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import optim
import pyro
import pyro.distributions as dist

from tensorboardX import SummaryWriter

from tqdm import tqdm
import time


import numpy as np
import random
from numpy.random import *
import matplotlib.pyplot as plt 
#np.random.seed(100)
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

import csv

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 1234
rng = np.random.RandomState(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    device='cuda'
else:
    device='cpu'


# In[2]:


size = 70000
treat = np.random.binomial(1,0.5,size=size)
effect = 1.5
beta2 = 0.8
beta3 = 0.3
beta4 = 0.5

beta5 = 0.4
beta6 = 0.3
beta7 = 0.5
beta8 = 0.3
beta9 = 0.2
beta10 = 0.2

eps = 1e-10
#b = np.random.normal(3,0.2)
b = 0
def mkdata(x1=treat,effect=effect,beta2=beta2,beta3=beta3, beta4=beta4,beta5=beta5,beta6=beta6,beta7=beta7,beta8=beta8,          beta9=beta9,beta10=beta10, b=b,size=size):
    #x1 = np.random.binomial(1,0.5,size=size)
    x2 = np.random.normal(1.4*x1+0.4,2.0,size=size)
    x3 = np.random.normal(-0.8*x1+1.3*x2,0.5,size=size)
    x4 = np.random.normal(0.2*x1+-0.9*x2+2.3*x3,0.4,size=size)
    
    x5 = np.random.normal(0.4*x1+0.4,2.0,size=size)
    x6 = np.random.normal(-0.3*x1+1.5*x5+0.2,2.5,size=size)
    x7 = np.random.normal(0.2*x1+1.5*x5-1.4*x6,1.4,size=size)
    x8 = np.random.normal(1.0*x1+2.3*x7+1.2*x5,2.0,size=size)
    x9 = np.random.normal(-0.8*x1+0.4*x6+0.6*x2,1.5,size=size)
    x10 = np.random.normal(0.2*x1+0.5,1.4,size=size)
    
    yogo= effect*x1+beta2*x2+beta3*x3+beta4*x4+beta5*x5+beta6*x6+beta7*x7+beta8*x8+beta9*x9+beta10*x10+b
    return(np.array((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,yogo)).T)

data = mkdata(x1=treat,size=size)
train_data = np.array(data[:60000])#train[:,0]:conditional, train[:,1]:feature, train[:,2]:outcome
test_data  = np.array(data[60000:])
