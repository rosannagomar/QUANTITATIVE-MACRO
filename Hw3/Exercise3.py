# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:33:25 2020

@author: ROSANNA_PC
"""

#============================
#IMPORT LIBRARIES
#============================
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.optimize import minimize_scalar, bisect

#As we have done in exercise 1 and in exercise 2, we start setting the parameters of the model:
Beta=0.988 #discounting factor
Theta=0.679 #labor share
Delta=0.013 #depreciation rate

Kappa=5.24 
Nu=2



#The following step consists of  define a grid for the values of capital, this is common for all the sections:

kss= ((1-Beta*(1-Delta))/(Beta*(1-Theta)))**(-1/Theta) #This is the capital of the steady state in this economy, look at HW 2 
nk=100 #nk evenly spaced points, set density of the grid
k_grid=np.linspace(0.5,kss,nk) #we have established the set lower bound and upper bound for the capital grid, close to steady state,

#%%#====================================================
#EXERCISE 3
#====================================================

#FOLLOWING MAKOTO NAKAJIMA'S NOTES

    

#The FIRST step consists of setting the order of polynomials used for approximation:
n=5
#The SECOND step consists of setting the number of collocation points used m, we observe that m>=n
m=nk
#The THIRD step consist in setting a tolerance parameter, we pick the same value than in the other exercises:
tolerance_error=0.005 #we set the parameter of tolerance error equal as in previous exercises.
#The FOUR step consists of setting an upper and lower bound of the state space
#We observe that the lower bound is 0.5 and the upper bound 1.5*kss, as in the other exercises

#The FIVE step consist in computing the collocation points, as we have seen in class the Chebyshev polynomials 
#are defined in the domain [-1,1], but we can move to another interval with the same distance, using the formula below:

roots = np.zeros(m)
for i in range(m):
    roots[i] = np.cos((((2*i)-1)/(2*m))*np.pi)
    

points = np.zeros(m)
for i in range(m):
    points[i] = ((roots[i] + 1)*(kss - tolerance_error)/2)+ tolerance_error
    

#The following step consist of an initial guess of the value function
y0 = np.ones((m,1))

#Then, we obtain the corresponding guess of the coefficients
coef_0 = np.polynomial.chebyshev.chebfit(points,y0,n)

#So, the value function is:
Vi = np.polynomial.chebyshev.chebval(k_grid,coef_0)
Vi = np.reshape(Vi, (m,1))

start_criteria = time.time() 

#Define the return matrix:
M=np.zeros((nk,nk)) #the return matri is zero at the beginning 
for i in range(0,nk):
    for j in range(0,nk):
        if k_grid[j]<((k_grid[i])**(1-Theta)+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- k_grid[j])-(Kappa/(1+(1/Nu))) 
        else:
            M[i,j]=-100000
           
def CHEB(M,Vi,coef_0, y0):

    
    Chi=np.empty((nk,nk))
    g = np.ones(nk) 
    y1 = np.empty((nk,1)) 
    
    #Generate the Chi Matrix
    
    for i in range(nk):
        for j in range(nk):
            Chi[i,j] = M[i,j] + Beta*Vi[j]
            
    #Next step, updated the value function 
    for i in range(nk):
        g[i] = np.argmax(Chi[i,:])
        
    for i in range(nk):     
        y1[i] = np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- g[i])-(Kappa/(1+(1/Nu))) + Beta*Vi[np.int_(g[i])]
    
    #Updated guess for the coefficients
    
    y1 = np.reshape(y1, (m,))
    coef_1 = np.polynomial.chebyshev.chebfit(points,y1,n)
    Vj = np.polynomial.chebyshev.chebval(k_grid,coef_1)
    Vj = np.reshape(Vj, (m,1))
        
    #Compare the two consecutive set of parameters
    
    intera = 0 
    while np.any(abs(coef_1 - coef_0) > tolerance_error): 
        intera+=1
        coef_0 = coef_1.copy()                             
        for i in range(nk):
            for j in range(nk):         
                Chi[i,j] = M[i,j] + Beta*Vj[j]
    
        for i in range(nk):
            g[i] = np.argmax(Chi[i,:])
            
                
        for i in range(nk):     
            y1[i] = np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- g[i])-(Kappa/(1+(1/Nu)))+ Beta*Vj[np.int_(g[i])]
            
            
        y1 = np.reshape(y1, (m,))
        coef_1 = np.polynomial.chebyshev.chebfit(points,y1,n)
        Vj = np.polynomial.chebyshev.chebval(k_grid,coef_1)
        Vj = np.reshape(Vj, (m,1))
        
        
          
        return y1,intera,g
 
ValueFunction_Cheb, intera,g = CHEB(M,Vi,coef_0, y0)

#Policy functions:
gk=np.empty(nk)
gc=np.empty(nk)
for i in range(nk):
    gk[i]=k_grid[int(g[i])]
    gc[i]=(k_grid[i]**(1-Theta))+(1-Delta)*k_grid[i]-gk[i]

   
stop_criteria=time.time()  
time_needed = stop_criteria - start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it has started. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals



print('We need', intera, 'iterations of the value function to reach convergence and the time needed is', time_needed, 'seconds' )


#PLOTTING
plt.plot(k_grid, ValueFunction_Cheb,color='tab:red',linewidth=3) 
plt.title('Value Function doing Chebyshev',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()

