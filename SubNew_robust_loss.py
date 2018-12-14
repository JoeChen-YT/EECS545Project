import numpy as np
import matplotlib.pyplot as plt
import random
import pylab as pl
import time
import csv
import pandas as pd


#randomly generate data
#For convience of comparison, use the same way to generate as one of the homework 
n = 20000
np.random.seed(0)
x = np.random.rand(n,1)
z = np.zeros([n,1])
k = int(n*0.4)
rp = np.random.permutation(n)
outlier_subset = rp[1:k]
z[outlier_subset] = 1 
y = (1-z)*(10*x + 5 + np.random.randn(n,1)) + z*(20 - 20*x + 10*np.random.randn(n,1))

ones_col = np.ones((n,1))
x_new = np.concatenate((ones_col,x),axis=1)

#calculate and store Hessian matrix components
st_m = np.zeros((2,2,n))
i=0
for i in range(n):
    st_m[:,:,i] = np.outer(x_new[i,:],x_new[i,:])

J_star = 6.89445248737445
theta_star = np.array([[6.465120311622218],[3.8306165636344693]])

#project
#GD,FullNew,SubNew1,SubNew2
start = time.time()
theta = np.zeros((2,1))
m=300     #m=n for Newton's method, m=300 for Sub
g=n
lamda = 0.025
theta_list = []
J_list = []

#main loop
for count in range(25):
    m=m+300       ##for SubNew 2 only
    J=0
    H = np.zeros((2,2))
    G = np.zeros((2,1))
    
    i=0
    for i in range(n):
        r = y[i].reshape(1,1) - theta.T@x_new[i,:].reshape(2,1)
        J = J + (1+r**2)**0.5
    J = J/n + lamda*( np.linalg.norm(theta) )**2        ##Calculate Objective function J
    
    
    i=0
    for i in range(g):
        r = y[i].reshape(1,1) - theta.T@x_new[i,:].reshape(2,1)
        G = G + x_new[i,:].reshape(2,1)*r*(1+r**2)**-0.5
    G = -G/g + lamda*2*theta          ##Calculate gradient
        
        
    i=0
    for i in range(m):
        r = y[i].reshape(1,1) - theta.T@x_new[i,:].reshape(2,1)
        H = H +  (1+r**2)**(-1.5) *st_m[:,:,i]
    H = H/m + lamda*2*np.identity(2)           ##Calculate Hessian
    
    theta_list.append(theta)
    J_list.append(J[0,0])
    
    #Gradiant descent
    #theta = theta - 1*G
    
    #Newton method
    theta = theta - np.linalg.inv(H)@G
    
end = time.time()
print("Total time spent = ",end - start)
print(J[0,0])
print(theta[0,0],theta[1,0])

#plot J
fig, ax = plt.subplots()
plt.plot(range(0,len(J_list)), J_list-np.ones(len(J_list))*J_star)

ax.set(xlabel='Iterations', ylabel='J',
      title='Value of Objective Function')

my_df = pd.DataFrame(J_list-np.ones(len(J_list))*J_star)
my_df.to_csv('out.csv', index=False, header=False)

res_ratio = []

for i in range (1,len(theta_list)-1):
    ratio = np.linalg.norm(theta_list[i+1] - theta_star)  /  np.linalg.norm(theta_list[i] - theta_star)
    res_ratio.append(ratio)

fig, ax = plt.subplots()
plt.plot(range(0,len(res_ratio)), res_ratio)

ax.set(xlabel='Iterations', ylabel='Rate',
      title='Convergence Rate')