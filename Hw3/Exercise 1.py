# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:27:08 2020

@author: ROSANNA_PC
"""
#========================================
#EXERCISE 1
#========================================
#========================================
#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import time


#First, we need to set the parameters of the model

Beta=0.988 #discounting factor
Theta=0.679 #labor share
Delta=0.013 #depreciation rate
h=1 #the labor is inellastically supplied


#The FIRST STEP consists of  define a grid for the values of capital, this is common for all the sections:

kss= ((1-Beta*(1-Delta))/(Beta*(1-Theta)))**(-1/Theta) #This is the capital of the steady state in this economy, look at the pdf
nk=200 #nk evenly spaced points, set density of the grid
k_grid=np.linspace(1,1.5*kss,nk) #we have established the set lower bound and upper bound for the capital grid, close to steady state,



#%%
#====================================================
#EXERCISE 1a:Bruce force Value Function Iteration
#====================================================

#THIRD step, we should obtain the feasible return matrix, called as M.

Start_criteria = time.time() #when the process starts
M=np.zeros((nk,nk)) #the return matri is zero at the beginning 
for i in range(0,nk):
    for j in range(0,nk):
        if k_grid[j]<=((k_grid[i])**(1-Theta)+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- k_grid[j]) 
        else:
            M[i,j]=-10000
#We observe how the first  condition is refered to the feasibility constrainst. The choice variable k', k_grid[j] in our notation has a lower bound equal to 0 since it cannot be negative, while the RHS of the expression is the upper bound:((k)**(1-Theta)+(1-Delta)*k).
#If the feasibility constraint of the economy holds, we want that the cells of the return matrix are equal to the utility function (we have put with equality in the constraint but if it is hold with equality, this means that the consumption would be equal to 0, and therefore we would be in a situation where the Inada conditions are violated, I have checked both options and this does not effect).
#Once we have constructed this matrix, we observe there exists some cells that are NaN, this happens due to the fact that not all values for c, and hence for k' are feasible in the economy, so 
#the STEP 4 consist of taking into account the nonnegativity restriction on consumption, and replace the elements in the matrix M that are negative for a very negative number (-10.000 in our case).

#In the STEP 5, we carry out the value function matrix iteration
tolerance_error=0.005 #we define the epsilon/tolerance_error
condition_nothold=True #this implies that if the difference between the updated value funcion and the value function in absolute values are higher than the tolerance error, we need to redo the process
Chi=np.empty((nk,nk))
intera=0 #we start with 0 interactions, and we are interested in analyzing how many interactions we need to reach convergence
g=np.ones((nk,1)) #initial guess for policy function
Vj = np.empty((nk,1))
Vi=np.zeros(nk) #SECOND STEP, guess of the initial value function, it is a vector

while (condition_nothold==True):
    intera+=1
    for i in range(nk):
        for j in range(nk):
            Chi[i,j]=M[i,j]+Beta*Vi[j]
    for i in range(nk):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((nk,1))    
       
    else:
         condition_nothold=False
         
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step

#We define the policy functions/decisions rules as:
gk=np.empty(nk)
gc=np.empty(nk)
for i in range(nk):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))+(1-Delta)*k_grid[i]-gk[i]

#We observe that for computing the policy function for consumption, it is relevant the policy function for capital, so the order matters.

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it has started. 
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

#%% 
#=======================================================
#EXERCISE 1b: MONOTONICITY
#=======================================================
"""
Here we pursue to reduce the number of points to be reached on the grid, to do so, we use the nknown property that the optimal decision rule increases in K, so

kj>ki, then g(kj)>g(ki)

Hence, if we want to find g(kj) when kj>ki, we can rule out searching for all grid values of capital that are smaller than g(ki), we need to modify the step 5
"""
#Im not going to explain every step again in the following parts since it has been explained carefully in section a.

Start_criteria = time.time() #when the process starts

M=np.zeros((nk,nk))
for i in range(0,nk):
    for j in range(0,nk):
        if k_grid[j]<=((k_grid[i])**(1-Theta)+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- k_grid[j]) 
        else:
            M[i,j]=-10000

tolerance_error=0.005
condition_nothold=True
Chi=np.empty((nk,nk))
intera=0
g=np.ones((nk,1))
Vj = np.empty((nk,1)) 
Vi=np.zeros(nk)  #initial guess value function
while (condition_nothold==True):
    intera+=1
    for i in range(nk):
        for j in range(nk):
            if k_grid[j]>=k_grid[int(g[i])]: 
                Chi[i,j]=M[i,j]+Beta*Vi[j]
            else:
                continue
    for i in range(nk):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((nk,1))       
    else:
         condition_nothold=False
         
#The condition for MONOTONICITY is introduced in line 149

        
#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.zeros(nk)
gc=np.zeros(nk)
for i in range(nk):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))+(1-Delta)*k_grid[i]-gk[i]

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


#%% 
#=======================================================
#EXERCISE 1c:  CONCAVITY
#=======================================================
"""
We use the known property that the maximand in the Bellmna equation, M_{i,j}+BetaVj, is strictly concave in k'
We have to modify again step 5.1

"""

Start_criteria=time.time()
M=np.zeros((nk,nk))
for i in range(0,nk):
    for j in range(0,nk):
        if k_grid[j]<=((k_grid[i])**(1-Theta)+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- k_grid[j]) 
        else:
            M[i,j]=-10000
            
tolerance_error=0.005
condition_nothold=True
Chi=np.empty((nk,nk))
intera=0
g=np.ones((nk,1))
Vj = np.empty((nk,1))
Vi=np.zeros([nk]) 

while (condition_nothold ==True):
    intera+=1
    for i in range(nk):
        for j in range(nk):
            Chi[i,j]=M[i,j]+Beta*Vi[j]
            if Chi[i,j]<Chi[i,j-1]: 
                break 
    for i in range(nk):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((nk,1))       
    else:
         condition_nothold=False
        
#the new step when we study the CONCAVITY is introduce in line 232

#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.empty(nk)
gc=np.empty(nk)
for i in range(nk):
    gk[i]=k_grid[int(g[i])]
    gc[i]=(k_grid[i]**(1-Theta))+(1-Delta)*k_grid[i]-gk[i]
    
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

#%% 
#=======================================================
#EXERCISE 1d:  LOCAL SEARCH
#=======================================================
"""
We exploit the property that the optimal decision rule is continuous, if kj=g(ki) and we have reasonable fine grids, then we hope that g(k_{i+1})is in a small neigborhood of kj. 
This restricts the elements of Chi that need to be computed.
"""

Start_criteria = time.time()
M=np.zeros((nk,nk))
for i in range(0,nk):
    for j in range(0,nk):
        if k_grid[j]<=((k_grid[i])**(1-Theta)+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- k_grid[j]) 
        else:
            M[i,j]=-10000
            
tolerance_error=0.005
condition_nothold=True
Chi=np.empty((nk,nk))
intera=0
g=np.ones((nk,1))
Vj = np.empty((nk,1))
Vi=np.zeros(nk)
while (condition_nothold ==True):
    intera+=1
    for i in range(nk):
        for j in range(nk):
             if (j >= g[i]) and (j <= g[i] + 5):  
                Chi[i,j]=M[i,j]+Beta*Vi[j]
         
    for i in range(nk):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((nk,1))       
    else:
         condition_nothold=False
         
#The new step for LOCAL SEARCH in line 308.

#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.zeros(nk)
gc=np.zeros(nk)
for i in range(nk):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))+(1-Delta)*k_grid[i]-gk[i]

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
#%% 
#=======================================================
#EXERCISE 1e: CONCAVITY AND MONOTONICITY
#=======================================================
"""
Put the modifications of b and c together.
"""

Start_criteria=time.time()
M=np.zeros((nk,nk))
for i in range(0,nk):
    for j in range(0,nk):
        if k_grid[j]<=((k_grid[i])**(1-Theta)+(1-Delta)*k_grid[i]): 
            M[i,j]=np.log(k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i]- k_grid[j]) 
        else:
            M[i,j]=-10000    
            
tolerance_error=0.005
condition_nothold=True
Chi=np.empty((nk,nk))
intera=0
g=np.ones((nk,1))
Vj = np.empty((nk,1))
Vi=np.zeros(nk)
while (condition_nothold ==True):
    intera+=1
    for i in range(nk):
        for j in range(nk):
            if k_grid[j]>=k_grid[int(g[i])]: 
                Chi[i,j] = M[i,j] + Beta*Vi[j]
                if Chi[i,j]<Chi[i,j-1]: 
                    break
         
    for i in range(nk):
        Vj[i]=np.max(Chi[i,:]) #updated value function Vs+1 as the maximum element in each row of Chi
        g[i] = np.argmax(Chi[i,:])
    if np.max(np.abs(Vj-Vi))>=tolerance_error:
       Vi = np.copy(Vj)
       Vj = np.empty((nk,1))       
    else:
         condition_nothold=False
         
#Monotonicity 383, concavity 385

#The step 6 consist of doing the difference between both value function and if this difference is smaller than epsilon we stop, if not we come back to the previous step
#We define the policy functions/decisions rules:
gk=np.zeros(nk)
gc=np.zeros(nk)
for i in range(nk):
    gk[i]= k_grid[int(g[i])] #a column vector that state the column number j that maximizes row i .
    gc[i]=(k_grid[i]**(1-Theta))+(1-Delta)*k_grid[i]-gk[i]

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

#%% 
#=======================================================
#EXERCISE 1f: HOWARD'S POLICY ITERATIONS
#=======================================================

Start_criteria=time.time()

# Initial guesses and empty matrices:
g0 = np.transpose(np.arange(0,nk))    # Initial decision rule guess
g = np.empty([nk, 1])  # empty matrix to update the guess

gki = np.empty([nk, 1])
gkj = np.empty([nk, 1])
gc = np.empty([nk, 1])

# Policy function iteration:
epsilon = 0.01 
convergence = False
s = 0

X = np.empty([nk, nk])
V0 = np.zeros([nk,1])
V = np.empty([nk,1])
# Policy iteration: 
while (convergence == False):
    for i in range(nk):
        gki[i] = k_grid[int(g0[i])]
        gc[i] = k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i] - gki[i]
        V[i] = np.max(np.log(gc[i]) + Beta*V0[i])
        g[i] = np.argmax(np.log(gc[i]) +Beta*V0[i])   # new decision rule
    for i in range(nk):
        gkj[i] = k_grid[int(g[i])]
        gc[i] = k_grid[i]**(1- Theta) + (1-Delta)*k_grid[i] - gkj[i]  
       
    if np.max(gkj - gki)>= epsilon:
        gki = np.copy(gkj)
        gkj = np.empty([nk,1])
        s += 1
    else:
        convergence = True

Stop_criteria = time.time() #when the process finishes
time_needed = Stop_criteria - Start_criteria #Obviously to know the time needed we have to do the difference when the program stop and when it starts. 
time_needed=round(time_needed,3) #We set that the execution time display only three decimals, if we don't use htis command to round, it has more than 12 decimals
print('We need', s, 'iterations of the value function with mono and concavity to reach convergence and the time needed is', time_needed, 'seconds' )

plt.plot(k_grid,V,color='tab:red',linewidth=3) 
plt.title(' Value Function Howard',fontsize=20)
plt.xlabel('k')
plt.ylabel('V(k)')
plt.show()


#Graph for the policy function for the capital with monotonicity and concavity
plt.plot(k_grid,gkj, color='tab:blue',linewidth=3) 
plt.title('Policy function for capital Howard',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_k$')
plt.show()
#Graph for the policy function for conumption with monotonicity and concavity
plt.plot(k_grid, gc,color='tab:purple',linewidth=3) 
plt.title('Policy function for consumption Howard',fontsize=20)
plt.xlabel('k')
plt.ylabel('$g_c$')
plt.show()
