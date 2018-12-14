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

#Ordinary least squares
x_tilt = x - x.mean(0)
y_tilt = y - y.mean(0)
w_ols = np.linalg.inv(x_tilt.T@x_tilt)*x_tilt.T@y_tilt
b_ols = y.mean(0) - w_ols.T@x.mean(0)
print(' w_ols=',w_ols[0,0],'\n','b_ols=',b_ols[0])
theta_star = np.array([[b_ols[0]],[w_ols[0,0]]])
i=0;J_star=0
for i in range(n):
    r = y[i].reshape(1,1) - theta_star.T@x_new[i,:].reshape(2,1)
    J_star = J_star + r**2
J_star = J_star/n + lamda*( np.linalg.norm(theta) )**2   ##Calculation of objective function J
print('J_star = ',J_star[0,0])

#project
#GD,FullNew,SubNew1,SubNew2
start = time.time()
g=n
m=16         ##m=16 for SubNew1,SubNew2, m=n for FullNew
theta = np.zeros((2,1))
theta_list = []
J_list = []
lamda = 0.00
itera = 45
itera_GD = 200         #gradient descent

for count in range(itera):
    #m=m+16               ##For SubNew2
    J=0
    H = np.zeros((2,2))
    G = np.zeros((2,1))
    
    i=0
    for i in range(n):
        r = y[i].reshape(1,1) - theta.T@x_new[i,:].reshape(2,1)
        J = J + r**2
    J = J/n + lamda*( np.linalg.norm(theta) )**2   ##Calculation of objective function J
    
    i=0
    for i in range(g):
        r = y[i].reshape(1,1) - theta.T@x_new[i,:].reshape(2,1)
        G = G + 2*x_new[i,:].reshape(2,1)*r
    G = -G/g + lamda*2*theta    ##Calculation of gradient
    
    #i=0
    #for i in range(m):
        #r = y[i].reshape(1,1) - theta.T@x_new[i,:].reshape(2,1)
        #H = H + 2*np.outer(x_new[i,:],x_new[i,:])
    #H = H/m + lamda*2*np.identity(2)    ##Calculation of Hessian 
    
    theta_list.append(theta)
    J_list.append(J)
    
    #Gradiant descent
    theta = theta - 0.1*G
    
    #Newton method
    #theta = theta - np.linalg.inv(H)@G
end = time.time()
print("Total time spent = ",end - start)
print(J)
print(theta)


# plot J
plot_J=[]
for i in range (1,len(J_list)-1):
    plotJ = np.linalg.norm(J_list[i+1])
    plot_J.append(plotJ)
    
fig, ax = plt.subplots()
plt.plot(range(0,len(plot_J)), plot_J)

ax.set(xlabel='Iterations', ylabel='J',
      title='Value of Objective Function')

my_df = pd.DataFrame(plot_J)
my_df.to_csv('out.csv', index=False, header=False)


res_ratio = []
theta_star = np.array([[b_ols[0]],[w_ols[0,0]]])
for i in range (1,len(theta_list)-1):
    ratio = np.linalg.norm(theta_list[i+1] - theta_star)  /  np.linalg.norm(theta_list[i] - theta_star)
    res_ratio.append(ratio)

fig, ax = plt.subplots()
plt.plot(range(0,len(res_ratio)), res_ratio)

ax.set(xlabel='Iterations', ylabel='Rate',
      title='Convergence Rate')

