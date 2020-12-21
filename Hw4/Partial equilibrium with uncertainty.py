# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:38:26 2020

@author: ROSANNA_PC
"""
#===================================================================
#II.4 PARTIAL EQUILIBRIUM- UNCERTAINTY
#===================================================================

#We start importing the libraries needed:
    
import numpy as np 
from numpy import vectorize
import matplotlib.pyplot as plt
from itertools import product
import time

# #First we set the parameters of the model:
r = 0.04              #interest rate
w = 1                 #following the pdf, we normalize the wage to 1
rho = 0.06
beta = 1/(1+rho)      #discount factor
gamma=0.95              #correlation between y' and y
sigma_y = 0.5           #parameter for income shock
c_bar = 100           #max level of consumption
periods=100           #number of periods
sigma=2               #coefficient of relative risk aversion

#The FIRST step consists of discretizing the state space:
#Define the 2-state process for the income:
Y=[1-sigma_y,1+sigma_y]

#%%
#*****************************
#QUADRATIC UTILITY FUNCTION
#*****************************
#===================================================================
#INFINITELY-LIVED HOUSEHOLDS ECONOMY (DISCRETE METHOD)
#===================================================================
#Define the natural borrowing limit:
a_min=(-Y[0]*(1+r)/r)

# I construct a grid of assets today, assets tomorrow and income shocks.
# Define the evenly spaced grid for assets:
assets=np.array(np.linspace(a_min, 30, periods))
assets_y_grid=list(product(Y,assets,assets))
assets_y_grid=np.array(assets_y_grid)


#Define the transition matrix:

pi = np.array([[(1+gamma)/2, (1-gamma)/2],[(1-gamma)/2, (1+gamma)/2]]) 

y=assets_y_grid[:,0]    
assets_today=assets_y_grid[:,1]
assets_tomorrow=assets_y_grid[:,2]

# Consumption:

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
    
# Since we have the feasible constraints into account, now we can define the
# return matrix
     
#Return vector:
M=M(c)

#Return matrix:
M = np.reshape(M,(1,20000))
M=np.reshape(M,(200,100))
 

# Initial guess for the value function is a vector of zeros:

Vs = np.zeros(200)

# Compute the matrix W for the value function with an initial guess for the value function:
    
def W1(assets):
    
    return pi[0, 0]*(-0.5*(Y[0] + (1+r)*assets - assets - c_bar)**2)/(1-beta) + pi[0, 1]*(-0.5*(Y[1] + (1+r)*assets - assets - c_bar)**2)/(1-beta)

def W2(assets):
    
    return pi[1, 0]*(-0.5*(Y[0] + (1+r)*assets -assets - c_bar)**2)/(1-beta) + pi[1, 1]*(-0.5*(Y[1] + (1+r)*assets - assets - c_bar)**2)/(1-beta)


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

# Compute the matrix Chi for the value function to obtain the new guess:

Chi = M + beta*W

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

#The last step consists in getting the policy functions, for doing this, we need the Chi matrix:
    
V_bad = V1[0:100]
V_good = V1[100:]

Chi = M + beta*W

#Assets policies:
    
g = np.argmax(Chi, axis = 1) #it give us the column number that maximizes row i of the chi matrix.
assets_bad = assets[g[0:100]]     
assets_good = assets[g[100:]]

consumption_bad = Y[0]*np.ones(100) + (1+r)*assets - assets_bad
consumption_good = Y[1]*np.ones(100) + (1+r)*assets - assets_good

        
# Plot the value function and the optimal policy:

plt.figure()
plt.plot(assets, consumption_bad, '.', label = 'Optimal consumption for bad shock')
plt.plot(assets, consumption_good, color="yellow", label = 'Optimal consumption for positive shock')
plt.title('Policy rule for consumption uncertainty, infinite horizon (quadratic utility)')
plt.ylim([-2, 15])
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()

#===================================================================
#FINITELY-LIVED HOUSEHOLDS ECONOMY 
#===================================================================
# T = 45
# Normalize W of T+1 to zero


y=assets_y_grid[:,0]    
assets_today=assets_y_grid[:,1]
assets_tomorrow=assets_y_grid[:,2]


c = y+(1+r)*assets_today-assets_tomorrow

@vectorize
  
def M(c):
    
    return -(1/2)*(c-c_bar)**2
     
M = M(c)
M = np.reshape(M,(1, 20000))
M = np.reshape(M,(200, 100))
W = np.zeros(200*100)
W = np.reshape(W, (200,100))

iterations = 0
finiteV = []
finiteG = []

for iterations in range(1, 46):
    
    Chi = M + beta*W
    g = np.argmax(Chi, axis = 1)
    W = np.amax(Chi, axis = 1)
    
    finiteV.append(W)       # It stores each iteration for obtaining the value function at each period (or age)
    finiteG.append(g)
    
    W = np.reshape(W, [200,1])
    W = np.tile(W, 100)
    W = np.transpose(W)
    W1 = W[:periods, :periods]
    W2 = W[:periods, periods:]
    W = np.concatenate((W1, W2))
    iterations = iterations+1
    
   
finiteV = np.array(finiteV)
finiteV = np.transpose(finiteV)
finiteG = np.array(finiteG)
finiteG = np.transpose(finiteG)

# Individual at periods 5 and 40:

A5 = assets[finiteG[0:100, 5]]
A40 = assets[finiteG[0:100, 40]]

C5 = Y[0]*np.ones(100) + (1+r)*assets - A5
C40 = Y[0]*np.ones(100) + (1+r)*assets - A40

plt.figure()
plt.plot(assets, C5,color="g",linestyle='--', label = 'Consumption for T=5')
plt.plot(assets, C40, color="purple", label = 'Consumption for T=40')
plt.title('Policy rule for consumption, uncertainty, T=45, quadratic preferences')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()

#SIMULATION PATHS

y = np.zeros([1, 45])

for i in range(0, 45):
    
    y[0, i] = np.random.choice((1-sigma_y, 1+sigma_y), p = ((1+gamma)/2, (1-gamma)/2))
 

 
g_assets = np.zeros([45,1])

for i in range(0, 45):
    if y[0, i] < 1:
    
        g_assets[i] = assets_bad[i]
        
for i in range(0, 44):
    
        g_assets[i] = assets_good[i]

          
t = np.linspace(0, 44, 44)

plt.figure()
plt.plot(t, g_assets[0:44], label = 'Assets ')
plt.title('Assets simulation for 45 periods, uncertainty (quadratic utility)')
plt.ylabel('Assets')
plt.xlabel('Time (periods)')
plt.show()

# Simulation and plot for consumption:
c=np.zeros(45)

for i in range(0, 44):
    
    c[i] = g_assets[i]*(1+r)+w*y[0, i]-g_assets[i+1]
    
    if c[i] <= 0:
        c[i] = 0

plt.figure()
plt.plot(t, c[0:44], label = 'Consumption')
plt.title('Consumption simulation for 45 periods, uncertainty (quadratic utility)')
plt.ylabel('Consumption')
plt.xlabel('Time (periods)')
plt.show() 

#%% 


#*****************************
#CRRA UTILITY FUNCTION
#*****************************

#===================================================================
#INFINITELY-LIVED HOUSEHOLDS ECONOMY (DISCRETE METHOD)
#===================================================================

#Define the natural borrowing limit:
a_min=(-Y[0]*(1+r)/r)

#Define the evenly spaced grid for assets:
assets=np.array(np.linspace(a_min, 30, periods))
assets_y_grid=list(product(Y,assets,assets))
assets_y_grid=np.array(assets_y_grid)

#Define the transition matrix:

pi = np.array([[(1+gamma)/2, (1-gamma)/2],[(1-gamma)/2, (1+gamma)/2]]) 

y=assets_y_grid[:,0]    
assets_today=assets_y_grid[:,1]
assets_tomorrow=assets_y_grid[:,2]


# Consumption:

c=y+(1+r)*assets_today-assets_tomorrow
M = np.zeros(20000)

for i in range(0, 20000):
    
    if c[i] >= 0:
        
        M[i] = ((c[i]**(1-sigma))-1)/(1-sigma)
        
    if c[i] < 0:
        
        M[i] = -100000

M = np.reshape(M, (1, 20000))        
M = np.reshape(M, (200, 100))
    
'''As we did in previous problem sets, this return the utility level associated to
all feasible combinations of assets and income shocks. If the combinations are not feasible,
we want a very negative number. '''
    
# Since we have the feasible constraints into account, now we can define the
# return matrix

# Initial guess for the value function is a vector of zeros:

Vs = np.zeros(200)

# Compute the matrix W:

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

# Compute the matrix Chi:

Chi = M + beta*W

#We need to update the value function:
Vs1=np.amax(Chi,axis=1)

#Compute the difference between the update the value fumction and the previous one:
difference_valuefunction=abs(Vs-Vs1)
iterations=0

#Run the whole WHILE LOOP:
epsilon=0.01   #tolerance error

for difference_valuefunction in range(1, 10000):
    
    Vss=Vs1
    Vs = [Vss[0:100], Vss[100:]]
    

#The vector has two columns since we have two shocks, one for each shock.

    Vs=np.array(Vs)
    
    def W1(V_guess):
        
        return pi[0, 0]*Vs[0, :] + pi[0, 1]*Vs[1, :]
    
    def W2(V_guess):
        
        return pi[1, 0]*Vs[0, :] + pi[1, 1]*Vs[1, :]

#We set up the W matrix for the bad shocks:
    W1=W1(Vs)
    W1=np.reshape(W1, (1,100))
    W1 = np.tile(W1, 100)
    W1 = np.transpose(W1)
#We transpose the matrix to have the transition only from Y[0]


#We set up the W matrix for the good shocks:
    W2=W2(Vs)
    W2=np.reshape(W2, (1,100))
    W2 = np.tile(W2, 100)
    W2 = np.transpose(W2)
#We transpose the matrix to have the transition only from Y[1]
# W matrix for the combination of good and bad shocks:
    W = [W1, W2]
    W= np.reshape(W, (200,100))
    
    
    Chi=M+beta*W
    V1=np.amax(Chi,axis=1)
    difference_valuefunction=abs(Vss-Vs1)
    
    iterations+=1
    
  

#The last step consists in getting the policy functions, for doing this, we need the Chi matrix:
    
V_bad = Vs1[0:100]
V_good = Vs1[100:]

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
plt.plot(assets, consumption_bad,".", label = 'Optimal consumption for bad shock')
plt.plot(assets, consumption_good, color="black", label = 'Optimal consumption for positive shock')
plt.title('Policy rule for consumption,infinite horizon (CRRA utility), uncertainty, precautionary savings')
plt.ylim([-2, 30])
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()


#===================================================================
#FINITELY-LIVED HOUSEHOLDS ECONOMY 
#===================================================================
# T = 45
# Normalize W of T+1 to zero


y=assets_y_grid[:,0]    
assets_today=assets_y_grid[:,1]
assets_tomorrow=assets_y_grid[:,2]



# Transition matrix:


c = y+(1+r)*assets_today-assets_tomorrow

M = np.zeros(20000)

for i in range(0, 20000):
    
    if c[i] >= 0:
        
        M[i] = ((c[i]**(1-sigma))-1)/(1-sigma)
        
    if c[i] < 0:
        
        M[i] = -100000

M = np.reshape(M, (1, 20000)) 
M = np.reshape(M,(200, 100))
W = np.zeros(200*100)
W = np.reshape(W, (200,100))

iterations = 0
finiteV = []
finiteG = []

for iterations in range(1, 46):
    
    Chi = M + beta*W
    g = np.argmax(Chi, axis = 1)
    W = np.amax(Chi, axis = 1)
    
    finiteV.append(W)       # It stores each iteration for obtaining the value function at each period (or age)
    finiteG.append(g)
    
    W = np.reshape(W, [200,1])
    W = np.tile(W, 100)
    W = np.transpose(W)
    W1 = W[:periods, :periods]
    W2 = W[:periods, periods:]
    W = np.concatenate((W1, W2))
    iterations = iterations+1
    
finiteV = np.array(finiteV)
finiteV = np.transpose(finiteV)
finiteG = np.array(finiteG)
finiteG = np.transpose(finiteG)

# Individual at periods 5 and 40:

A5 = assets[finiteG[0:100, 5]]
A40 = assets[finiteG[0:100, 40]]

C5 = Y[0]*np.ones(100) + (1+r)*assets - A5
C40 = Y[0]*np.ones(100) + (1+r)*assets - A40

for i in range(0, 100):
    
    if C5[i] < 0:
        
        C5[i] = 0
    
    if C40[i] < 0:
        
        C40[i] = 0

plt.figure()
plt.plot(assets, C5,color="black",linestyle='--', label = 'Consumption for T=5')
plt.plot(assets, C40,  color="red",label = 'Consumption for T=40')
plt.title('Policy rule for consumption, T=45, CRRA preferences, uncertainty, precautionary savings')
plt.legend()
plt.ylabel('Consumption')
plt.xlabel('Assets')
plt.show()

#***************45 periods*************************

y = np.zeros([1, 45])

for i in range(0, 45):
    
   y[0, i] = np.random.choice((1-sigma_y, 1+sigma_y), p = ((1+gamma)/2, (1-gamma)/2))
 
# Simulation and plot for assets:
        
assets = np.zeros([45,1])

for i in range(0, 45):
    if y[0, i] < 1:
    
        assets[i] = assets_bad[i]
    
    if y[0, i] > 1:    
          assets[i] = assets_good[i]

          
t = np.linspace(0, 44, 44)

plt.figure()
plt.plot(t, assets[1:],color="red", label = 'Assets ')
plt.title('Assets simulation for 45 periods,uncertainty (CRRA utility)')
plt.ylabel('Assets')
plt.xlabel('Time (periods)')
plt.show()

# Simulation and plot for consumption:

c = np.zeros(45)

for i in range(0, 44):
    
    c[i] = assets[i]*(1+r)+w*y[0, i]-assets[i+1]
    
    if c[i] <= 0:
        c[i] = 0

plt.figure()
plt.plot(t, c[0:44], color="red", label = 'Consumption')
plt.title('Consumption simulation for 45 periods, uncertainty (CRRA utility)')
plt.ylabel('Consumption')
plt.xlabel('Time (periods)')
plt.show()
