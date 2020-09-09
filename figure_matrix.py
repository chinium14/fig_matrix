#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *

empty_index = [1]
width=5
height=5
rows = 3
cols = 2
axes=[]
fig=plt.figure()
param_1 = ['0.001', '0.004', '0.02']
param_2 = ['nospaPval', 'spaPval']


cols_name = ['Column {}'.format(col) for col in range(1, 3)]
rows_name = ['Row {}'.format(row) for row in ['A', 'B', 'C']]
    
empty_index = set(empty_index)

for a in range(rows*cols):
    index_1 = a//2
    index_2 = a%2
    im = array(Image.open('sample_random_10000_tau_0.25_withCov.survTime.lambda.0.001.dist.Exponential_censoring_lambdac_'+param_1[index_1]+'.txt_sub_qqplot_sub_'+param_2[index_2]+'_QQ.png'))
    #b = np.random.randint(7, size=(height,width))
    if a not in empty_list:
        axes.append( fig.add_subplot(rows, cols, a+1) )
    else:
        im = array(Image.open('sample_random_10000_tau_0.25_withCov.survTime.lambda.0.001.dist.Exponential_censoring_lambdac_'+param_1[index_1]+'.txt_sub_qqplot_sub_'+param_2[index_2]+'_QQ.png'))
        axes.append( fig.add_subplot(rows, cols, a+1) )
        
    #subplot_title=("Subplot"+str(a))
    #axes[-1].set_title(subplot_title)  
    plt.imshow(im)
    if index_1 == 0:
        axes[-1].set_title(cols_name[index_2])
    if index_2 == 0:
        axes[-1].set_ylabel(rows_name[index_1], rotation=0, size='large')
#for ax, col_name in zip(axes[0], cols_name):
#    ax.set_title(col_name)

#for ax, row_name in zip(axes[:,0], rows_name):
#    ax.set_ylabel(row_name, rotation=0, size='large')
    
fig.tight_layout()    
plt.show()

fig.savefig('./temp.pdf')


# In[ ]:




