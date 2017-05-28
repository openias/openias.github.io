import numpy as np

import theano as th

import sklearn.datasets
import sklearn.preprocessing

#from sklearn import datasets
#from sklearn.datasets import make_moons
#from sklearn.preprocessing import scale

def rigid_transform(x, theta, dx=np.zeros(2)):
    R = [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta),  np.cos(theta)]]

    return np.dot(R, x.T).T + dx

def make_moons(N):

    x, y = sklearn.datasets.make_moons(noise=abs(np.random.normal(0.05, 0.05)), n_samples=N)
    
    
    x[y == 0] = rigid_transform(x[y == 0], theta=np.random.uniform(low=-0.01*2*np.pi, high=0.05*2*np.pi))
    x[y == 1] = rigid_transform(x[y == 1], theta=np.random.uniform(low=-0.05*2*np.pi, high=0.01*2*np.pi))
    
    x = rigid_transform(x, theta=np.random.normal(0, 2*np.pi), dx=np.random.normal(0, 0.25, size=(2)))
    
    x = sklearn.preprocessing.scale(x).astype(th.config.floatX)
    y = y.reshape(y.shape[0], 1).astype(th.config.floatX)
    
    return x, y
