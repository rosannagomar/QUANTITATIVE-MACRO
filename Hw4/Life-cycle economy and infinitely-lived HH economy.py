# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:53:24 2020

@author: ROSANNA_PC
"""


#We start importing the libraries needed:
import numpy as np
from numpy import vectorize
from itertools import product
import matplotlib.pyplot as plt
import scipy as sp
import time

#First we set the parameters of the model:
    
r=0.04             #interest rate
rho=0.06   
w=1                #following the pdf, we normalize the wage to 1
beta=1/(1+rho)     #discount factor
sigma=2            #coefficient of relative risk aversion
periods=100        #number of periods
sigma_y=0.2        #parameter for income shock
gamma=0.5          #correlation between y' and y

#The FIRST step consists of discretizing the state space:
#Define the 2-state process for the income:
Y=[1-sigma_y,1+sigma_y]
c_bar=100*Y[1]  #max level of consumption

#Define the natural borrowing limit:
a_min=(-Y[0]*(1+r)/r)

#Define the transition matrix:

pi = np.array([[(1+gamma)/2, (1-gamma)/2],[(1-gamma)/2, (1+gamma)/2]]) 

#%%

#*****************************
#QUADRATIC UTILITY FUNCTION
#*****************************

#===================================================================
#II.2 INFINITELY-LIVED HOUSEHOLDS ECONOMY (DISCRETE METHOD)
#===================================================================
#Define the evenly spaced grid for assets:
assets=np.array(np.linspace(a_min, 30, periods))
assets_y_grid=list(product(Y,assets,assets))
assets_y_grid=np.array(assets_y_grid)

Start_criteria = time.time()
#The SECOND step is about our initial guess for the value function:
Vs=np.zeros(200)

#The THIRD step is the feasible return matrix, known as M.
# I construct a grid of assets today, assets tomorrow and income shocks.
y=assets_y_grid[:,0]    
assets_today=assets_y_grid[:,1]
assets_tomorrow=assets_y_grid[:,2]

c=y+(1+r)*assets_today-assets_tomorrow

#I define the return matrix that accounts for the utility:
@vectorize
def M(c):
    if c>0:
        return -(1/2)*(c-c_bar)**2
    else:
        return -100000
    
'''As we did in previous problem sets, this return the utility level associated to
all feasible combinations of assets and income shocks. If the combinations are not feasible,
we want a very negative number. '''
    
#Return vector:
M=M(c)

#Return matrix:
M = np.reshape(M,(1,20000))
M=np.reshape(M,(200,100))
 
#In the step FOUR, we define the matrix W for the value function with an initial guess for the value function:
def W1(assets):   
    
    return pi[0, 0]*(-0.5*(Y[0] + (1+r)*assets - assets - c_bar)**2)/(1-beta) + pi[0, 1]*(-0.5*(Y[1] + (1+r)*assets - assets - c_bar)**2)/(1-beta)

def W2(assets):
    
    return pi[1, 0]*(-0.5*(Y[0] + (1+r)*assets - assets - c_bar)**2)/(1-beta) + pi[1, 1]*(-0.5*(Y[1] + (1+r)*assets - assets - c_bar)**2)/(1-beta) 

#We set up the W matrix for the bad shocks:
W1=W1(assets)
W1=np.reshape(W1, (100,1))
W1 = np.tile(W1, 100)
W1 = np.transpose(W1)
#We transpose the matrix to have the transition only from Y[0]

#We set up the W matrix for the good shocks:
W2=W2(assets)
W2=np.reshape(W2, (100,1))
W2 = np.tile(W2, 100)
W2 = np.transpose(W2)
#We transpose the matrix to have the transition only from Y[1]


#Now, we create the W matrix for the good and bad shocks:
W = [W1, W2]
W= np.reshape(W, (200,100))  

#In the step 5, we create the chi matrix for the value function to obtain the new guess:
Chi=M+beta*W

#We need to update the value function:
V1=np.amax(Chi,axis=1)


#Compute the difference between the update the value fumction and the previous one:
difference_valuefunction=abs(V1-Vs)
iterations=0

#Run the whole WHILE LOOP:
epsilon=0.01   #tolerance error
while difference_valuefunction.all()>epsilon:
    Vs_guess=V1
    V_guess=[Vs_guess[0:periods], Vs_guess[periods:]]

#The vector has two columns since we have two shocks, one for each shock.

    V_guess=np.array(V_guess)
    
    def W1(V_guess):
        
        return pi[0, 0]*V_guess[0, :] + pi[0, 1]*V_guess[1, :]
    
    def W2(V_guess):
        
        return pi[1, 0]*V_guess[0, :] + pi[1, 1]*V_guess[1, :]

#We set up the W matrix for the bad shocks:
    W1=W1(V_guess)
    W1=np.reshape(W1, (1,100))
    W1 = np.tile(W1, 100)
    W1 = np.transpose(W1)
#We transpose the matrix to have the transition only from Y[0]


#We set up the W matrix for the good shocks:
    W2=W2(V_guess)
    W2=np.reshape(W2, (1,100))
    W2 = np.tile(W2, 100)
    W2 = np.transpose(W2)
#We transpose the matrix to have the transition only from Y[1]
# W matrix for the combination of good and bad shocks:
    W = [W1, W2]
    W= np.reshape(W, (200,100))
    
    Chi=M+beta*W
    V1=np.amax(Chi,axis=1)
    difference_valuefunction=abs(Vs_guess-V1)
    
    iterations+=1
    
    Stop_criteria = time.time() #when the process finishes
    time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
    time_needed=round(time_needed,3)
    
print('We need', iterations, 'iterations of the value function to reach convergence and the time needed is', time_needed, 'seconds' )



#The last step consists in getting the policy functions, for doing this, we need the Chi matrix:
    
V_bad = V1[0:100]
V_good = V1[100:]

Chi = M + beta*W

#Assets policies:
    
g = np.argmax(Chi, axis = 1) #it give us the column number that maximizes row i of the chi matrix.
assets_bad = assets[g[0:100]]     
assets_good = assets[g[100:]]

#Once we have calculated the assets policies, we can compute the consumption policies:
    
consumption_bad = Y[0]*np.ones(100) + (1+r)*assets - assets_bad
consumption_good = Y[1]*np.ones(100) + (1+r)*assets - assets_good

# Plot the value function and the optimal policy:
    
plt.figure()
plt.plot(assets, V_bad, color='tab:red',label = 'Value function for bad shock')
plt.plot(assets, V_good, color='tab:green',label = 'Value function for good shock')
plt.title('Value Function Iterations for infinite horizon Quadratic')
plt.legend()
plt.ylabel('Value Function')
plt.xlabel('Assets')
plt.show()
    
plt.figure()
plt.plot(assets, assets_bad,color='tab:red', label = 'Optimal assets for bad shock')
plt.plot(assets, assets_good,color='tab:green', label = 'Optimal assets for good shock')
plt.title('Policy function for assets for infinite horizon Quadratic')
plt.legend()
plt.ylabel('Assets tomorrow')
plt.xlabel('Assets today')
plt.show()

plt.figure()
plt.plot(assets, consumption_bad,color='tab:red', label = 'Optimal consumption for bad shock')
plt.plot(assets, consumption_good,color='tab:green', label = 'Optimal consumption for good shock')
plt.title('Policy function for consumption for infinite horizon Quadratic')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()

#===================================================================
#II.3 LIFE-CYCLE ECONOMY (BY BACKWARD INDUCTION)
#===================================================================


W = np.zeros((200, 100))

iterations = 0

while iterations < 45:
    
    W = np.amax((M + beta*W), axis = 1)
    W = np.reshape(W,(200, 1))
    W = W*np.ones((200, 100))
    
    iterations += 1

plt.plot(assets, W[0:100, 0], color='tab:red',label = 'Value function for bad shock')
plt.plot(assets, W[100:, 0], color='tab:green', label = 'Value function for good shock')
plt.legend()
plt.title('Value function for finite horizon Quadratic')
plt.ylabel('Value function')
plt.xlabel('Assets')
plt.show()

Chi = M + beta*W
g = np.argmax(Chi, axis = 1)

assets_bad = assets[g[0:100]]    
assets_good = assets[g[100:]]      

consumption_bad = Y[0]*np.ones(100) + (1+r)*assets - assets_bad

consumption_good= Y[1]*np.ones(100) + (1+r)*assets - assets_good


        
plt.figure()
plt.plot(assets, assets_bad, color='tab:red', label = 'Optimal assets for bad shock')
plt.plot(assets, assets_good, color='tab:green',label = 'Optimal assets for good shock')
plt.legend()
plt.title('Policy rule for assets for finite horizon Quadratic')
plt.ylabel('Assets tomorrow')
plt.xlabel('Assets today')
plt.show()

plt.figure()
plt.plot(assets, consumption_bad, color='tab:red', label = 'Optimal consumption for bad shock')
plt.plot(assets, consumption_good, color='tab:green',label = 'Optimal consumption for good shock')
plt.title('Policy rule for consumption for finite horizon Quadratic')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()


#%%

#************************
#CRRA UTILITY FUNCTION
#************************

#===================================================================
#II.2 INFINITELY-LIVED HOUSEHOLDS ECONOMY (DISCRETE METHOD)
#===================================================================

#Define the evenly spaced grid for assets:
assets=np.array(np.linspace(a_min, 30, periods))
assets_y_grid=list(product(Y,assets,assets))
assets_y_grid=np.array(assets_y_grid)

Start_criteria = time.time()
#The SECOND step is about our initial guess for the value function:
Vs=np.zeros(200)

#The THIRD step is the feasible return matrix, known as M.
# I construct a grid of assets today, assets tomorrow and income shocks.
y=assets_y_grid[:,0]    
assets_today=assets_y_grid[:,1]
assets_tomorrow=assets_y_grid[:,2]

c=y+(1+r)*assets_today-assets_tomorrow

#I define the return matrix that accounts for the utility:
M = np.zeros(20000)

for i in range(0, 20000):
    
    if c[i] >= 0:
        
        M[i] = ((c[i]**(1-sigma))-1)/(1-sigma)
        
    if c[i] < 0:
        
        M[i] = -100000
'''As we did in previous problem sets, this return the utility level associated to
all feasible combinations of assets and income shocks. If the combinations are not feasible,
we want a very negative number. '''
    

#Return matrix:
M = np.reshape(M,(1,20000))
M=np.reshape(M,(200,100))
 
#In the step FOUR, we define the matrix W for the value function with an initial guess for the value function:

def W1(assets):   
    
    return pi[0, 0]*(((Y[0] + (1+r)*assets - assets)**(1-sigma))-1)/((1-sigma)*(1-beta)) + pi[0, 1]*(((Y[1] + (1+r)*assets - assets)**(1-sigma))-1)/((1-sigma)*(1-beta))

def W2(assets):
    
    return pi[1, 0]*(((Y[0] + (1+r)*assets - assets)**(1-sigma))-1)/((1-sigma)*(1-beta)) + pi[1, 1]*(((Y[1] + (1+r)*assets - assets)**(1-sigma))-1)/((1-sigma)*(1-beta))
#We set up the W matrix for the bad shocks:
W1=W1(assets)
W1=np.reshape(W1, (100,1))
W1 = np.tile(W1, 100)
W1 = np.transpose(W1)
#We transpose the matrix to have the transition only from Y[0]

#We set up the W matrix for the good shocks:
W2=W2(assets)
W2=np.reshape(W2, (100,1))
W2 = np.tile(W2, 100)
W2 = np.transpose(W2)
#We transpose the matrix to have the transition only from Y[1]


#Now, we create the W matrix for the good and bad shocks:
W = [W1, W2]
W= np.reshape(W, (200,100))  

#In the step 5, we create the chi matrix for the value function to obtain the new guess:
Chi=M+beta*W

#We need to update the value function:
V1=np.amax(Chi,axis=1)


#Compute the difference between the update the value fumction and the previous one:
difference_valuefunction=abs(V1-Vs)
iterations=0

#Run the whole WHILE LOOP:
epsilon=0.01   #tolerance error
while difference_valuefunction.all()>epsilon:
    Vs_guess=V1
    V_guess=[Vs_guess[0:periods], Vs_guess[periods:]]

#The vector has two columns since we have two shocks, one for each shock.

    V_guess=np.array(V_guess)
    
    def W1(V_guess):
        
        return pi[0, 0]*V_guess[0, :] + pi[0, 1]*V_guess[1, :]
    
    def W2(V_guess):
        
        return pi[1, 0]*V_guess[0, :] + pi[1, 1]*V_guess[1, :]

#We set up the W matrix for the bad shocks:
    W1=W1(V_guess)
    W1=np.reshape(W1, (1,100))
    W1 = np.tile(W1, 100)
    W1 = np.transpose(W1)
#We transpose the matrix to have the transition only from Y[0]


#We set up the W matrix for the good shocks:
    W2=W2(V_guess)
    W2=np.reshape(W2, (1,100))
    W2 = np.tile(W2, 100)
    W2 = np.transpose(W2)
#We transpose the matrix to have the transition only from Y[1]
# W matrix for the combination of good and bad shocks:
    W = [W1, W2]
    W= np.reshape(W, (200,100))
    
    Chi=M+beta*W
    V1=np.amax(Chi,axis=1)
    difference_valuefunction=abs(Vs_guess-V1)
    
    iterations+=1
    
    Stop_criteria = time.time() #when the process finishes
    time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
    time_needed=round(time_needed,3)
    
print('We need', iterations, 'iterations of the value function to reach convergence and the time needed is', time_needed, 'seconds' )



#The last step consists in getting the policy functions, for doing this, we need the Chi matrix:
    
V_bad = V1[0:100]
V_good = V1[100:]

Chi = M + beta*W

#Assets policies:
    
g = np.argmax(Chi, axis = 1) #it give us the column number that maximizes row i of the chi matrix.
assets_bad = assets[g[0:100]]     
assets_good = assets[g[100:]]

#Once we have calculated the assets policies, we can compute the consumption policies:
    
consumption_bad = Y[0]*np.ones(100) + (1+r)*assets - assets_bad
consumption_good = Y[1]*np.ones(100) + (1+r)*assets - assets_good

# Plot the value function and the optimal policy:
    
plt.figure()
plt.plot(assets, V_bad, color='tab:blue',label = 'Value function for bad shock')
plt.plot(assets, V_good, color='tab:orange',label = 'Value function for good shock')
plt.title('Value Function Iterations for infinite horizon CRRA')
plt.legend()
plt.ylabel('Value Function')
plt.xlabel('Assets')
plt.show()
    
plt.figure()
plt.plot(assets, assets_bad,color='tab:blue', label = 'Optimal assets for bad shock')
plt.plot(assets, assets_good,color='tab:orange', label = 'Optimal assets for good shock')
plt.title('Policy function for assets for infinite horizon CRRA')
plt.legend()
plt.ylabel('Assets tomorrow')
plt.xlabel('Assets today')
plt.show()

plt.figure()
plt.plot(assets, consumption_bad,color='tab:blue', label = 'Optimal consumption for bad shock')
plt.plot(assets, consumption_good,color='tab:orange', label = 'Optimal consumption for good shock')
plt.ylim([0, 4])
plt.title('Policy function for consumption for infinite horizon CRRA')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()

#===================================================================
#II.3 LIFE-CYCLE ECONOMY (BY BACKWARD INDUCTION)
#===================================================================

W = np.zeros((200, 100))

iterations = 0

while iterations < 45:
    
    W = np.amax((M + beta*W), axis = 1)
    W = np.reshape(W,(200, 1))
    W = W*np.ones((200, 100))
    
    iterations += 1

plt.plot(assets, W[0:100, 1], label = 'Value function for negative shock')
plt.plot(assets, W[100:, 1], label = 'Value function for positive shock')
plt.title('Value function for finite horizon CRRA')
plt.legend()
plt.ylabel('Value function')
plt.xlabel('Assets')
plt.show()

Chi = M + beta*W
g = np.argmax(Chi, axis = 1)

assets_bad = assets[g[0:100]]     
assets_good = assets[g[100:]]      


consumption_bad = Y[0]*np.ones(100) + (1+r)*assets - assets_bad

consumption_good = Y[1]*np.ones(100) + (1+r)*assets - assets_good


plt.figure()
plt.plot(assets, assets_bad, color='tab:blue',label = 'Optimal assets for bad shock')
plt.plot(assets, assets_good, color='tab:orange',label = 'Optimal assets for good shock')
plt.title('Policy rule for assets for finite horizon CRRA')
plt.legend()
plt.ylabel('Assets tomorrow')
plt.xlabel('Assets today')
plt.show()

plt.figure()
plt.plot(assets, consumption_bad, color='tab:blue',label = 'Optimal consumption for bad shock')
plt.plot(assets, consumption_good, color='tab:orange',label = 'Optimal consumption for good shock')
plt.title('Policy rule for consumption for finite horizon CRRA')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()
