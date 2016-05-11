
# coding: utf-8

# In[15]:

import deep_residual_learning_CIFAR10
reload(deep_residual_learning_CIFAR10)

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np


# In[6]:

dat = deep_residual_learning_CIFAR10.load_data()


# In[30]:

X_train = dat["X_train"]


# In[35]:

def re_view(img):
    return np.transpose(img, [1,2,0])


# In[49]:

plt.imshow(re_view(X_train[12]))


# In[25]:

dat["X_train"][0].shape


# In[28]:

np.transpose(dat["X_train"][0], [2,1,0])


# In[54]:

dat["Y_train"]


# In[57]:

28*28*50000


# In[58]:

32*32*3*50000


# In[ ]:



