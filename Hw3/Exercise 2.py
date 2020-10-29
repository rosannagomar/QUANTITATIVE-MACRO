# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 18:32:38 2020

@author: ROSANNA_PC
"""
#========================================
#EXERCISE 2
#========================================

#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import time


#First, we need to set the parameters of the model

Beta=0.988 #discounting factor
Theta=0.679 #labor share
Delta=0.013 #depreciation rate
Kappa=5.24 #disutility from labor
Nu=2

#The FIRST STEP consists of  define a grid for the values of capital and labor, this is common for all the sections:

kss= ((1-Beta*(1-Delta))/(Beta*(1-Theta)))**(-1/Theta) #This is the capital of the steady state in this economy, look at pdf, the labor does not effect, we consider that both factors are perfect complementaries
n=200 #nk evenly spaced points, set density of the grid
k_grid=np.linspace(0.1,1.5*kss,n) #we have established the set lower bound and upper bound for the capital grid, close to steady state,
#if I write the same lower bound 1, and the same upper bound 1.5*kss as in exercise 1, we cannot observe the process of convergence.
#So, I have done several changes and its better if i change the lower bound to observe this convergence process

h_grid=np.linspace(0.1,1,n)
#The grid for labor is between 0.05 and 1, if i put 0, I obtain a brute force value function so negative.

#%%
#====================================================
#EXERCISE 2a:Bruce force Value Function Iteration
#====================================================


#THIRD step, we should obtain the feasible return matrix, called as M.

Start_criteria = time.time() #when the process starts

M=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if k_grid[j]<(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]-k_grid[j])-((Kappa*(h_grid[j])**(1+(1/Nu)))/(1+(1/Nu))) 
        else:
            M[i,j]=-10000
#We observe how the first if is refered to the feasibility constrainst k' has a lower bound equal to since it cannot be negative, but the RHS of the expression is the upper bound:((k)**(1-Theta)+(1-Delta)*k).
#If the feasibility constraint of the economy holds we want that the cells of the return matrix are equal to the utility function.
#Once we have constructed this matrix, we observe there exists some cells that are NaN, this happens due to the fact that not all values for c, and hence for k' are feasible, so 
#the STEP 4 consist of taking into account the nonnegativity restriction on consumption and replace the elements in the matrix M that are negative for a very negative number.     
#In the STEP 5, we carry out the value function matrix iteration.

#There is an introduction in the utility of the representative agent that is the disutility from labor and the changes in production function due to the fact that now the labor is not inellastically supplied.

tolerance_error=0.005 #we set the tolerance error
condition_nothold=True #this implies that if the difference between the updated value funcion and the value function in absolute values are higher than the tolerance error, we need to redo the process
Chi=np.empty((n,n)) 
intera=0 #we start with 0 interactions, and we are interested in analyzing how many interactions we need to reach convergence
g=np.ones((n,1)) #initial guess for policy function
Vj = np.empty((n,1))
Vi=np.zeros(n) #SECOND STEP, guess of the initial value function, it is a vector


while (condition_nothold==True):
    intera+=1
    for i in range(n):
        for j in range(n):
            Chi[i,j]=M[i,j]+Beta*Vi[j]
    for i in range(n):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((n,1))    
       
    else:
         condition_nothold=False
         
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step


#We define the policy functions/decisions rules, now we introduce the policy function for labor since it is a choice variable:
gk=np.empty(n)
gc=np.empty(n)
gh=np.empty(n)

for i in range(n):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))*(h_grid[i]**(Theta))+(1-Delta)*k_grid[i]-gk[i]
    gh[i]=h_grid[int(g[i])]

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals

print('We need', intera, 'iterations of the value function to reach convergence and the time needed is', time_needed, 'seconds' )

#PLOTTING
#Graph for the Value Function

plt.plot(k_grid,Vj,color='tab:red',linewidth=3) 
plt.title('Brute Force Value Function',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()
#Graph for the policy function for the capital
plt.plot(k_grid,gk, color='tab:blue',linewidth=3) 
plt.title('Policy function for capital',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_k$')
plt.show()
#Graph for the policy function for consumption
plt.plot(k_grid, gc,color='tab:purple',linewidth=3) 
plt.title('Policy function for consumption',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_c$')
plt.show()

#Graph for the policy function for labor
plt.plot(k_grid, gh,color='tab:green',linewidth=3) 
plt.title('Policy function for labor',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_h$')
plt.show()

#%%
#====================================================
#EXERCISE 2b: Monotonicity of the decision rule
#====================================================

"""
Here we pursue to reduce the number of points to be reached on the grid, to do so, we use the nknown property that the optimal decision rule increases in K, so

kj>ki, then g(kj)>g(ki)

Hence, if we want to find g(kj) when kj>ki, we can rule out searching for all grid values of capital that are smaller than g(ki), we need to modify the step 5
"""

M=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if k_grid[j]<(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]-k_grid[j])-((Kappa*(h_grid[j])**(1+(1/Nu)))/(1+(1/Nu))) 
        else:
            M[i,j]=-10000
#We observe how the first if is refered to the feasibility constrainst k' has a lower bound equal to since it cannot be negative, but the RHS of the expression is the upper bound:((k)**(1-Theta)+(1-Delta)*k).
#If the feasibility constraint of the economy holds we want that the cells of the return matrix are equal to the utility function.
#Once we have constructed this matrix, we observe there exists some cells that are NaN, this happens due to the fact that not all values for c, and hence for k' are feasible, so 
#the STEP 4 consist of taking into account the nonnegativity restriction on consumption and replace the elements in the matrix M that are negative for a very negative number.     
#In the STEP 5, we carry out the value function matrix iteration

tolerance_error=0.005
condition_nothold=True
Chi=np.empty((n,n))
intera=0
g=np.ones((n,1))
Vj = np.empty((n,1))
Vi=np.zeros(n)


while (condition_nothold==True):
    intera+=1
    for i in range(n):
        for j in range(n):
            if k_grid[j]>=k_grid[int(g[i])]:
                Chi[i,j]=M[i,j]+Beta*Vi[j]
            else:
                continue
    for i in range(n):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((n,1))    
       
    else:
         condition_nothold=False
         
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.empty(n)
gc=np.empty(n)
gh=np.empty(n)
#We observe that nk is equal to nh so we can use one of them indistinctivily
for i in range(n):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))*(h_grid[i]**(Theta))+(1-Delta)*k_grid[i]-gk[i]
    gh[i]=h_grid[int(g[i])]

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals
print('We need', intera, 'iterations of the value function with monotonicity to reach convergence and the time needed is', time_needed, 'seconds' )

#PLOTTING
#Graph for the Value Function with monotonicity
k=k_grid
plt.plot(k,Vj,color='tab:red',linewidth=3) 
plt.title(' Value Function with monotonicity',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()
#Graph for the policy function for the capital with monotonicity
plt.plot(k,gk, color='tab:blue',linewidth=3) 
plt.title('Policy function for capital with monotonicity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_k$')
plt.show()
#Graph for the policy function for consumption with monotonicity
plt.plot(k, gc,color='tab:purple',linewidth=3) 
plt.title('Policy function for consumption with monotonicity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_c$')
plt.show()         

#Graph for the policy function for labor with monotonicity
plt.plot(k_grid, gh,color='tab:green',linewidth=3) 
plt.title('Policy function for labor with monotonicity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_h$')
plt.show()

#%% 
#=======================================================
#EXERCISE 2c:  CONCAVITY
#=======================================================
"""
We use the known property that the maximand in the Bellmna equation, M_{i,j}+BetaVj, is strictly concave in k'
We have to modify again step 5.1

"""

#THIRD step, we should obtain the feasible return matrix, called as M.

Start_criteria = time.time() #when the process starts

M=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if k_grid[j]<(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]-k_grid[j])-((Kappa*(h_grid[j])**(1+(1/Nu)))/(1+(1/Nu))) 
        else:
            M[i,j]=-10000
#We observe how the first if is refered to the feasibility constrainst k' has a lower bound equal to since it cannot be negative, but the RHS of the expression is the upper bound:((k)**(1-Theta)+(1-Delta)*k).
#If the feasibility constraint of the economy holds we want that the cells of the return matrix are equal to the utility function.
#Once we have constructed this matrix, we observe there exists some cells that are NaN, this happens due to the fact that not all values for c, and hence for k' are feasible, so 
#the STEP 4 consist of taking into account the nonnegativity restriction on consumption and replace the elements in the matrix M that are negative for a very negative number.     
#In the STEP 5, we carry out the value function matrix iteration

tolerance_error=0.005
condition_nothold=True
Chi=np.empty((n,n))
intera=0
g=np.ones((n,1))
Vj = np.empty((n,1))
Vi=np.zeros(n)

while (condition_nothold==True):
    intera+=1
    for i in range(n):
        for j in range(n):
            Chi[i,j]=M[i,j]+Beta*Vi[j]
            if Chi[i,j-1]>Chi[i,j]:
                break
    for i in range(n):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((n,1))    
       
    else:
         condition_nothold=False
         
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.empty(n)
gc=np.empty(n)
gh=np.empty(n)
#We observe that nk is equal to nh so we can use one of them indistinctivily
for i in range(n):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))*(h_grid[i]**(Theta))+(1-Delta)*k_grid[i]-gk[i]
    gh[i]=h_grid[int(g[i])]

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals
print('We need', intera, 'iterations of the value function with concavity to reach convergence and the time needed is', time_needed, 'seconds' )
#PLOTTING
#Graph for the Value Function with concavity
k=k_grid
plt.plot(k,Vj,color='tab:red',linewidth=3) 
plt.title(' Value Function with concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()
#Graph for the policy function for the capital  with concavity
plt.plot(k,gk, color='tab:blue',linewidth=3) 
plt.title('Policy function for capital with concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_k$')
plt.show()
#Graph for the policy function for conumption  with concavity
plt.plot(k, gc,color='tab:purple',linewidth=3) 
plt.title('Policy function for consumption with concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_c$')
plt.show()

#Graph for the policy function for labor with concavity
plt.plot(k_grid, gh,color='tab:green',linewidth=3) 
plt.title('Policy function for labor with concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_h$')
plt.show()

#%% 
#=======================================================
#EXERCISE 2d:  LOCAL SEARCH
#=======================================================
"""
We exploit the property that the optimal decision rule is continuous, if kj=g(ki) and we have reasonable fine grids, then we hope that g(k_{i+1})is in a small neigborhood of kj. 
This restricts the elements of Chi that need to be computed.
"""


#THIRD step, we should obtain the feasible return matrix, called as M.

Start_criteria = time.time() #when the process starts

M=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if k_grid[j]<(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]-k_grid[j])-((Kappa*(h_grid[j])**(1+(1/Nu)))/(1+(1/Nu))) 
        else:
            M[i,j]=-10000
#We observe how the first if is refered to the feasibility constrainst k' has a lower bound equal to since it cannot be negative, but the RHS of the expression is the upper bound:((k)**(1-Theta)+(1-Delta)*k).
#If the feasibility constraint of the economy holds we want that the cells of the return matrix are equal to the utility function.
#Once we have constructed this matrix, we observe there exists some cells that are NaN, this happens due to the fact that not all values for c, and hence for k' are feasible, so 
#the STEP 4 consist of taking into account the nonnegativity restriction on consumption and replace the elements in the matrix M that are negative for a very negative number.     
#In the STEP 5, we carry out the value function matrix iteration

tolerance_error=0.005
condition_nothold=True
Chi=np.empty((n,n))
intera=0
g=np.ones((n,1))
Vj = np.empty((n,1))
Vi=np.zeros(n)

while (condition_nothold==True):
    intera+=1
    for i in range(n):
        for j in range(n):
            if (j >= g[i]) and (j <= g[i] + 5):
                Chi[i,j]=M[i,j]+Beta*Vi[j]
            
    for i in range(n):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((n,1))    
       
    else:
         condition_nothold=False
         
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.empty(n)
gc=np.empty(n)
gh=np.empty(n)
#We observe that nk is equal to nh so we can use one of them indistinctivily
for i in range(n):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))*(h_grid[i]**(Theta))+(1-Delta)*k_grid[i]-gk[i]
    gh[i]=h_grid[int(g[i])]

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals
print('We need', intera, 'iterations of the value function with local search to reach convergence and the time needed is', time_needed, 'seconds' )

#PLOTTING
#Graph for the Value Function with local search
k=k_grid
plt.plot(k,Vj,color='tab:red',linewidth=3) 
plt.title(' Value Function with local search',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()
#Graph for the policy function for the capital with local search
plt.plot(k,gk, color='tab:blue',linewidth=3) 
plt.title('Policy function for capital with local search',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_k$')
plt.show()
#Graph for the policy function for conumption with local search
plt.plot(k, gc,color='tab:purple',linewidth=3) 
plt.title('Policy function for consumption with local search',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_c$')
plt.show()
#Graph for the policy function for labor with local search
plt.plot(k_grid, gh,color='tab:green',linewidth=3) 
plt.title('Policy function for labor with local search',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_h$')
plt.show()
#%% 
#=======================================================
#EXERCISE 2e: CONCAVITY AND MONOTONICITY
#=======================================================
"""
Put the modifications of b and c together.
"""

Start_criteria = time.time() #when the process starts

M=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if k_grid[j]<(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(((k_grid[i])**(1-Theta))*((h_grid[j])**(Theta))+(1-Delta)*k_grid[i]-k_grid[j])-((Kappa*(h_grid[j])**(1+(1/Nu)))/(1+(1/Nu))) 
        else:
            M[i,j]=-10000
#We observe how the first if is refered to the feasibility constrainst k' has a lower bound equal to since it cannot be negative, but the RHS of the expression is the upper bound:((k)**(1-Theta)+(1-Delta)*k).
#If the feasibility constraint of the economy holds we want that the cells of the return matrix are equal to the utility function.
#Once we have constructed this matrix, we observe there exists some cells that are NaN, this happens due to the fact that not all values for c, and hence for k' are feasible, so 
#the STEP 4 consist of taking into account the nonnegativity restriction on consumption and replace the elements in the matrix M that are negative for a very negative number.     
#In the STEP 5, we carry out the value function matrix iteration

tolerance_error=0.005
condition_nothold=True
Chi=np.empty((n,n))
intera=0
g=np.ones((n,1))
Vj = np.empty((n,1))
Vi=np.zeros(n)

while (condition_nothold==True):
    intera+=1
    for i in range(n):
          for j in range(n):
            if k_grid[j]>=k_grid[int(g[i])]: 
                Chi[i,j] = M[i,j] + Beta*Vi[j]
                if Chi[i,j]<Chi[i,j-1]: 
                    break
            
    for i in range(n):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((n,1))    
       
    else:
         condition_nothold=False
         
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.empty(n)
gc=np.empty(n)
gh=np.empty(n)
#We observe that nk is equal to nh so we can use one of them indistinctivily
for i in range(n):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))*(h_grid[i]**(Theta))+(1-Delta)*k_grid[i]-gk[i]
    gh[i]=h_grid[int(g[i])]
    
    

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals
print('We need', intera, 'iterations of the value function with mono and concavity to reach convergence and the time needed is', time_needed, 'seconds' )

#PLOTTING
#Graph for the Value Function with monotonicity and concavity
k=k_grid
plt.plot(k,Vj,color='tab:red',linewidth=3) 
plt.title(' Value Function with monotonicity and concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()
#Graph for the policy function for the capital with monotonicity and concavity
plt.plot(k,gk, color='tab:blue',linewidth=3) 
plt.title('Policy function for capital with monotonicity and concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_k$')
plt.show()
#Graph for the policy function for conumption with monotonicity and concavity
plt.plot(k, gc,color='tab:purple',linewidth=3) 
plt.title('Policy function for consumption with monotonicity and concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_c$')
plt.show()

#Graph for the policy function for labor with monotonicity and concavity
plt.plot(k_grid, gh,color='tab:green',linewidth=3) 
plt.title('Policy function for labor with with monotonicity and concavity',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_h$')
plt.show()





