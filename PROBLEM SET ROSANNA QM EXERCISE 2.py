# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:53:15 2020

@author: ROSANNA_PC
"""
"Import all the libraries that can be useful doing this exercise"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import scipy.stats as st
import math
"EXERCISE 2"

#PART A
#Start setting the parameters of the model 
A_f=1 #aggregate hours at the workplace
A_nf=1 #aggregate hours teleworking
rho=1.1
kappa_f=0.2 #preferences
kappa_nf=0.2
omega=20 #how much planner cares about death
gamma=0.9 #prob of survive, this means that 1-gamma is the rate of infected people that die
i_o=0.2 #initial share of infections at work
N=1 #individuals, normalization
n=10 #subdivisions for variable

#The objetive function of the model and the constraint can be writen as:

function = lambda H:-1* ((A_f*H[0]**((rho-1)/rho)+grid[j]*A_nf*H[1]**((rho-1)/rho))**(rho/(rho-1)) - kappa_f*H[0]-kappa_nf*H[1] - omega*((1-gamma)*grid[i]*(i_o*H[0]**2/N))) 
constraint=({'type':'ineq','fun': lambda H: N-H[0]-H[1]})

# a lambda function is a single-line function, which can have any number of arguments, but it can only have one expression.
# the constraint type: 'eq' for equality and 'ineq' for inequality. 
#fun is the function defining the constraint.

grid=np.linspace(0,1, n) #since beta(HC) and c(TW) are between 0 and 1, both included.

#we are creating a matrix of zeros n by n dimensions of the values for each H, H_f and H_nf respectively.
result_hf=np.zeros (shape=(n,n))
result_hnf=np.zeros(shape=(n,n))
#Let's completing it doing a loop

for i in range(n):
    for j in range(n):
        H_0=[0.4,0.6] #initial guess for H_f and H_nf
        lim=[(0,1),(0,1)] #as H_f+H_nf=1, if one takes value 1, the other takes value 0
        optimum=minimize(function,H_0, constraints=constraint, bounds=lim)
        result_hf[i][j]=optimum.x[0]
        result_hnf[i][j]=optimum.x[1]

#We have compute the optimum value for h_f and h_nf

h=result_hf+result_hnf #the total h is the sum of the others two

H=np.around(h, decimals = 0, out = None) 
"""
I have seen that the matrix of h is not equal to 1, there are some cases that is 0.99, to avoid this problem of 
decimals, I have use the comand around to approximate all values to 1.
"""
Hf_divided_H=result_hf/H #we are asked for computing this

#Let's calculate the infection of the economy:
Infection= np.zeros(shape=(n,n))
for i in range(n):
    for j in range(n):
        Infection[i][j]=result_hf[i][j]**2*i_o*grid[i]
        
#Once we have calculate the infections, we can obtain the deaths of this economy:
Deaths=(1-gamma)*Infection

#Now, the optimal value of Y:
Y=np.zeros(shape=(n,n))
for i in range(n):
    for j in range (n):
        Y[i][j]=(A_f*result_hf[i][j]**(rho-1/rho)+grid[j]*A_nf*result_hnf[i][j]**(rho-1/rho))**(rho/(rho-1))
    
#Recall that our n is equal to 10, then we should divide the space into 10 elements as well:
    
space= [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]  
Welfare=np.zeros(shape=(n,n))
for i in range(n):
    for j in range (n):
        Welfare[i][j]= ((A_f*result_hf[i][j]**((rho-1)/rho)+grid[j]*A_nf*result_hnf[i][j]**((rho-1)/rho))**(rho/(rho-1)) - kappa_f*result_hf[i][j]-kappa_nf*result_hnf[i][j] - omega*((1-gamma)*grid[i]*(i_o*result_hf[i][j]**2/N))) 

#GRAPHS:
fig,ax=plt.subplots()
sns.heatmap(result_hf,cbar_kws={'label':'$H_f$'},xticklabels=space, yticklabels=space)
plt.title('$H_f$',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(H,cbar_kws={'label':'$H$'},xticklabels=space, yticklabels=space)
plt.title('$H$',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(result_hnf,cbar_kws={'label':'$H_{nf}$'},xticklabels=space, yticklabels=space)
plt.title('$H_{nf}$',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(Hf_divided_H,cbar_kws={'label':'$H_f / H$'},xticklabels=space, yticklabels=space)
plt.title('$H_f/H$',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(Infection,cbar_kws={'label':'$Infection$'},xticklabels=space, yticklabels=space)
plt.title('Infection',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(Deaths,cbar_kws={'label':'$Deaths$'},xticklabels=space, yticklabels=space)
plt.title('Deaths',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(Welfare,cbar_kws={'label':'$Welfare$'},xticklabels=space, yticklabels=space)
plt.title('Welfare',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')

fig,ax=plt.subplots()
sns.heatmap(Y,cbar_kws={'label':'$Output$'},xticklabels=space, yticklabels=space)
plt.title('Output',fontsize=16)
plt.xlabel('c(TW)')
plt.ylabel('$\\beta(HC)$')
#PART B
#The first modifications consist of setting omega to 150 and the second one, rho equals to 10. I have change the previous parameters and run the code.