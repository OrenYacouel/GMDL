"""
Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import time
import numpy as np
import matplotlib.pylab as pylab
from pylab import plt
from GMMInference import GMMInference

img = pylab.imread('landscape.jpg')
img=img[::10,::10]
img=img.astype(float)/255
N = img.shape[0]*img.shape[1]
print ('N:',N)

# data
x = img.reshape(N,3).T.copy()
data_dim=x.shape[0]


K= 4
hyperparams=dict()
hyperparams['Psi']=np.eye(data_dim)* (0.2)**2
hyperparams['m'] = np.ones(data_dim)* 0.5 # the image is in [0,1], so take the mid value.
hyperparams['kappa'] = 1
hyperparams['nu'] = 8000
hyperparams['alphas']= np.ones(K) * 1000
 
colors = np.random.rand(K,3)    

gmm_inference = GMMInference(data_dim=data_dim,K=K,hyperparams=hyperparams,colors=colors)

tic = time.time()
gmm_inference.gibbs_sampling(x=x,nIters=40,N=N,tf_savefig=False) 
toc=time.time()
print ('time:',toc-tic)


   

labels=gmm_inference.z_est

seg = labels.reshape(img.shape[0],img.shape[1])

plt.figure(1)
plt.subplot(131)
plt.imshow(img,interpolation="Nearest")
plt.title('Image')
plt.subplot(132)
plt.imshow(seg,interpolation="Nearest")
plt.title("Segmentation")

plt.subplot(133)
palette = np.array([comp.m_posterior for comp in gmm_inference.comps])
plt.imshow(palette[seg],interpolation="Nearest")
plt.title("Means")
plt.show()