# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:53:15 2020

@author: ROSANNA_PC
"""
"Import all the libraries that can be useful doing the Homework 2"
import numpy as np
import pandas as pd
import sympy as sy
import matplotlib.pyplot as plt #Default plotting in python
import mpmath as mp
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from scipy.optimize import minimize #Scipy builds on top of numpy to provide common tools for scientific programming such as:
#Linear algebra, numerical integration, interpolation, optimization, 
#distribution and random numbers gen, signal processing, etc.
import seaborn as sns #Seaborn: Tries to improve and enhance matplotlib. Specially good for data visualization or complex plots

"EXERCISE 1"
"Part a"
#Set the parameters of the model, h and theta are given, delta can be calculate since we know the ratio of investment-output and capital-output:
h=0.31
Theta=0.67
Delta=0.0625

#Normalize the output to 1 and the capital to 4
k=4
y=1
i= Delta*k #Since in steady state k_t=k_t+1=K
c=y-i #the consumption is equal to the production minus the investment
z= k**((Theta-1)/Theta) / h  #the algebra is in the pdf
#We define two  functions the Production Function that have constant returns to scale and the Euler Equation
def F(k,z):
    return(k)**(1-Theta)*(z*h)**(Theta)


def Euler(Beta):
    f=(1-Delta+(1-Theta)*(k)**(-Theta)*(z*h)**Theta)-(1/Beta)
    return f

def u_prime(c): #it is not used
    return 1/c

solution_Beta=[]
range_1 = np.linspace(0.01,1,10)
Beta_1=1/ ( (1-Theta)*(k)**(-Theta) * (z*h)**Theta + 1 - Delta) #Once we have solved for the other parameters, we obtain the value of beta
# Observe different starting points
#create a loop:
for i in range_1: # to store results from a loop, might be useful to create and empty list and then append. 
	tmp = optimize.fsolve(Euler, i)
	solution_Beta.append(tmp)

#We are going to print the parameters and the variables in steady state.
    
print ("The productivity parameter,z=", "{0:.4f}".format(z))
print ("Beta=", "{0:.4f}".format(np.mean(solution_Beta)))
print ("Steady State for capital=", "{0:.2f}".format(k))
print ("Steady State for output=", "{0:.2f}".format(y))
print ("Steady State for consumption=", "{0:.2f}".format(c))
print ("Delta", "{0:.4f}".format(Delta))

#This coincides with the analytical result, see pdf.

"Part b"
#There is a modification, now the productivity parameter is twice the previous parameter, so:
    
z_2=z*2

#The parameters values delta, beta, gamma and labour are the same that in the previous part, the shock does not affect to them.

def Euler_new(k_2): #we need to define a new euler equation since the value of z has changed
    f2=(1-Delta+(1-Theta)*(k_2)**(-Theta)*(z_2*h)**Theta)-(1/Beta_1)
    return f2
solution_Beta2=[]
range_2 = np.linspace(0.01,1,10)

for j in range_2: 
	tmp2 = optimize.fsolve(Euler_new, j)
	solution_Beta2.append(tmp2)
    
"""
The command: optimize.fsolve find the roots of a function.
Return the roots of the (non-linear) equations defined by func(x) = 0 given a starting estimate.

"""
#We define the new values for capital, production, investment and consumption, that are the ones that are affected by this shock.
k_2=np.mean(solution_Beta2)
y_2=(z_2*h)**Theta*k_2**(1-Theta)
i_2=k_2*Delta
c_2=y_2-i_2

#We print the results
print ("New productivity parameter,z=", "{0:.4f}".format(z_2))
print ("New Steady State for capital=", "{0:.2f}".format(k_2))
print ("Beta=", "{0:.4f}".format(np.mean(solution_Beta)))
print ("New steady State for output=", "{0:.2f}".format(y_2))
print ("New steady State for consumption=", "{0:.2f}".format(c))
print ("Delta", "{0:.4f}".format(Delta))

#I  have created a table to put together the results of part a and b.

Table = {'Parameters/Variables': [ 'z', 'c', 'i',  'y', 'k', 'h', 'Theta', 'Beta', 'Delta'], 'Steady State a': [z, c, i, y, k, h, Theta, Beta_1, Delta,], 'Steady State b': [z_2, c_2, i_2, y_2, k_2, h, Theta, Beta_1, Delta, ]}
Table_ab = pd.DataFrame(Table)
print(Table_ab)

"Part c"
"""
First of all, we define the functions that are relevant to study the transition: 
    - the production function
    - the resource constraint
    - the capital in steady state
    - the consumption in steady state
    - the Euler Equation
    - the constant capital

In all of them, first we defefine the parameters that we need to obtain the value and then the equation that is needed to be solved.
"""
def ProductionFunction(k,params):
        """ 
        The production function is defined as
        """
        # extract params
        
        theta = params['theta']
        delta  = params['delta']
        h = params ['h']
        z = params ['z']
       
        return k**(1-theta)*(z*h)**(theta)
def Steady_State(params):
        beta   = params['beta']
        theta = params['theta']
        delta = params['delta']
        h = params ['h']
        z = params ['z']
        
        k_star=z*h*((beta*(1-theta))/(1-beta*(1-delta)))**(1/theta) #(((1-theta)/((1/beta)+delta-1))**(1/theta))*z*h
        c_star= ProductionFunction(k,params)- delta*k_star
        return k_star,c_star
    
def ResourceConstraint (k,c,params):
        beta   = params['beta']
        theta = params['theta']
        delta = params['delta']
        h = params ['h']
        return ProductionFunction(k, params) + (1-delta)*k -c

def EulerEquation(k, c, params):
        beta  = params['beta']
        theta = params['theta']
        delta = params['delta']
        h = params ['h']
        z = params ['z']
        k_next=ResourceConstraint(k,c,params) #the capital in the next period its equal to the result of the resource constraint function. 
        
        if k_next > 0: #This means that if the capital is positive in the next period, the consumption of the next period is going to be calculated 
            c_next=c*beta*(1 - delta + (1-theta)*pow(z*h, theta)*pow(k, -theta))
            return c_next
        else: #if the capital in the next period is not positive, the consumption is equal to 0.
            return 0
        
def Constant_k(k,params):
        delta=params['delta']
        return ProductionFunction(k,params)-delta*k


# Again, we set the parameters of the model as in part a.
params = {'delta':0.25/4, 'beta':0.98,  'theta':0.67, 'z':1.6297, 'h':0.31}

# We want to obtain the values of the capital and the consumption in steady state
print(Steady_State(params))
k_star, c_star = Steady_State(params)
#This result is practically the same that in part a.
            

# One important step is setting the forward equations and the convergence criterion



def Path(c_0, k_0, params, T=1000): #1000 itenerations
    
    T += 1
    
    k_t = np.zeros(T) #first of all, we create a vectors of zeros that the length coincides with the itenerations
    c_t = np.zeros(T)
    
    k_t[0] = k_0 #For the first iteneration (recall that 0 means the first element of the vector/list), the first value is called as k_0, the same for consumption
    c_t[0] = c_0
    
    for t in range(T-1): #this is a loop since we want to calculate the capital in the following period, until the previous last period
        k_t[t+1] = ResourceConstraint(k_t[t], c_t[t], params)
        if k_t[t+1] > 0: #this is exactly the same as before, if capital is positive in the next period, I want to know the value of consumption. Otherwise, it takes 0.
            c_t[t+1] = EulerEquation(k_t[t], c_t[t], params)
        else:
            k_t[t+1] = 0
            c_t[t+1] = 0
            
    return k_t, c_t

def Path_crit(c_0, k_0, params, T=100): #100 itenerations
    
    k_t, c_t = Path(c_0, k_0, params, T)
    k_star, c_star = Steady_State(params)
    
    ss_diff = np.sqrt((k_t-k_star)**2 + (c_t-c_star)**2)
    
    return np.min(ss_diff) + ss_diff[-1]


k_0 = k_star / 20

# Find the function minimum, starting from an initial guess:
    
result = minimize(Path_crit, 0.34, args=(k_0, params, 100), method='Nelder-Mead')
"""
Minimization of scalar function of one or more variables using the Nelder-Mead algorithm.

"""
c_0 = result.x
c_00 = c_0

#The phase diagram is done a continuation:
k_t, c_t = Path(c_0, k_0, params, 50)

kk = np.linspace(0, 1.5*k_star, 1000)
cc = np.linspace(0, 1.5*c_star, 1000)

#This is for doing plot of the capital per effective labor, the consumption per effective labor, the sadel path and the steady state:
plt.plot(kk, Constant_k(kk, params),color='purple', lw=2, label='$\hat{k}_{t+1}=\hat{k}_{t}$')
plt.plot(kk**0 * k_star, cc, color='blue', lw=2, label='$\hat{c}_{t+1}=\hat{c}_{t}$')
plt.plot(k_star, c_star, 'ro', label='Steady state')
plt.plot(k_t, c_t, 'k-', label='Saddle path')

#put title to the graph:
plt.title('Phase Diagram_a', fontsize=20, weight='bold')
#Name to the axes:
plt.xlabel('Capital per effective labor $\hat{k}$')
plt.ylabel('Consumption per effective labor $\hat{c}$')
#Legend:
plt.legend(loc='lower right')

plt.show()


#Now, we are doing exactly the same for the part b, where productivity is double
# Parameters: double the productivity parameter z permanently :
params = {'delta':0.25/4, 'beta':0.98,  'theta':0.67, 'z':2*1.6297, 'h':0.31}
# Steady state
print(Steady_State(params))
k_star, c_star = Steady_State(params)

k_0 = 4.0

# Find the function minimum, starting from an initial guess
result = minimize(Path_crit, 0.34, args=(k_0, params, 50), method='Nelder-Mead')
print(result)

c_0 = result.x

k_t, c_t = Path(c_0, k_0, params, 50)
kk = np.linspace(0, 1.5*k_star, 1000)
cc = np.linspace(0, 1.5*c_star, 1000)

#Graph of the phase diagram:

plt.plot(kk, Constant_k(kk, params), lw=2, color='purple', label='$\hat{k}_{t+1}=\hat{k}_{t}$')
plt.plot(kk**0 * k_star, cc, lw=2, color='blue', label='$\hat{c}_{t+1}=\hat{c}_{t}$')
plt.plot(k_star, c_star, 'ro', label='Steady state')
plt.plot(k_t, c_t, 'k-', label='Saddle path')

plt.title('Phase Diagram_b', fontsize=20, weight='bold')
plt.xlabel('Capital per effective labor $\hat{k}$')
plt.ylabel('Consumption per effective labor $\hat{c}$')
plt.legend(loc='lower right')

plt.show()

#Once, the graphs of the phase diagram are represented for the two steady states founded in part a and b, now let's focus on time-path
k_int = k_t
c_int = c_t 
s_t = ProductionFunction(k_t, params)-c_t  #the savings are equal to the production function minus the marginal propensity to consume
h_t = np.full(np.shape(s_t),h)
y_t = s_t + c_t 

fig = plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.plot(s_t, label='Savings',color='red')
plt.title('Time-path for savings', fontsize=16)
plt.legend()

plt.subplot(222)
plt.plot(c_t, label='Consumption', color='red')
plt.title('Time-path for consumption', fontsize=16)
plt.legend()

plt.subplot(223)
plt.plot(h_t, label='Labor',color='red')
plt.title('Time-path for labor', fontsize=16)
plt.legend()

plt.subplot(224)
plt.plot(y_t, label='Output',color='red')
plt.title('Time-path for output', fontsize=16)

plt.legend()
plt.show()

"Part d"
"""
Now there is an unexpected shock that affects productivity, decreasing to the level of part a

"""
# Parameters: double the productivity parameter z permanently 
params = {'delta':0.25/4, 'beta':0.98,  'theta':0.67, 'z':2*1.6297, 'h':0.31}
# Steady state
print(Steady_State(params))
k_star, c_star = Steady_State(params)

k_0 = 4 # Old steady states

# Find the function minimum, starting from an initial guess
result = minimize(Path_crit, 0.34, args=(k_0, params, 10), method='Nelder-Mead')
print(result)

c_0 = result.x

k_t, c_t = Path(c_0, k_0, params, 10)  # 10 Periods shocks


k_00 = k_t[10]

#Once we get period 10, there is the productivity shock that effects on z, so z changes.
params = {'delta':0.25/4, 'beta':0.98,  'theta':0.67, 'z':1.6297, 'h':0.31} 

result = minimize(Path_crit, 0.34, args=(k_00, params, 40), method='Nelder-Mead')
c_00 = result.x
k_t1, c_t1 = Path(c_00, k_00, params, 40)
k_t = np.append(k_t, k_t1)
c_t = np.append(c_t, c_t1)


s_t = ProductionFunction(k_t, params)-c_t
h_t = np.full(np.shape(s_t),h)
alpha = 0.33
z = 1.6297
y_t = s_t + c_t

#We have done the same as before, I have not explained the steps. The only modification is divided into two periods, the one with high productivity and the other with low productivity
#GRAPHS:

fig = plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.plot(s_t, color='green', label='Savings')
plt.title('Time-path for savings', fontsize=16)
plt.legend()

plt.subplot(222)
plt.plot(c_t, color='green', label='Consumption')
plt.title('Time-path for consumption', fontsize=16)
plt.legend()

plt.subplot(223)
plt.plot(h_t, color='green',label='Labor')
plt.title('Time-path for labor', fontsize=16)
plt.legend()

plt.subplot(224)
plt.plot(y_t, color='green',label='Output')
plt.title('Time-path for output', fontsize=16)

plt.legend()
plt.show()

"Part e"

"""

def ProductionFunction_new(k,h, params):
        
      
        # extract params
        
        theta = params['theta']
        delta  = params['delta']
        z = params ['z']
        v= params['v']
        return k**(1-theta)*(z*h)**(theta)
    
def ResourceConstraint_new (k,c,h,params):
        beta   = params['beta']
        theta = params['theta']
        delta = params['delta']
        v= params['v']
        return ProductionFunction_new(k,h,params) + (1-delta)*k -c
    
def EulerEquation_new(k, c, h, params):
        beta  = params['beta']
        theta = params['theta']
        delta = params['delta']
        z = params ['z']
        v= params['v']
        k_next=ResourceConstraint_new(k,c,params) #the capital in the next period its equal to the result of the resource constraint function. 
        
        if k_next > 0: #This means that if the capital is positive in the next period, the consumption of the next period is going to be calculated 
            c_next=c*beta*(1 - delta + (1-theta)*pow(z*h, theta)*pow(k, -theta))
            return c_next
        else: #if the capital in the next period is not positive, the consumption is equal to 0.
            return 0
        
def Constant_k_new(k,h,params):
        delta=params['delta']
        return ProductionFunction_new(k,h,params)-delta*k
    
def Steady_State_new(params):
        beta   = params['beta']
        theta = params['theta']
        delta = params['delta']
        z = params ['z']
        v= params['v']
        
        k_star=z*h_star*((beta*(1-theta))/(1-beta*(1-delta)))**(1/theta)
        c_star= ProductionFunction_new(k,h,params)- delta*k_star
        h_star=((k_star**(1-theta)*z**-theta)/c_star)**(v/(1-theta*v+v))
        
        return k_star,c_star, h_star
    
# Again, we set the parameters of the model as in part a.
params = {'delta':0.25/4, 'beta':0.98,  'theta':0.67, 'z':1.6297, 'v':0.3}

# We want to obtain the values of the capital and the consumption in steady state
print(Steady_State_new(params))
k_star, c_star, h_star = Steady_State_new(params)

"""