# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:52:27 2020

@author: ROSANNA_PC
"""

#=============================================================================================================================
#IMPORT LIBRARIES
#=============================================================================================================================
#First of all, we import the libraries:
    
import numpy as np
import scipy.optimize as opt
from scipy.optimize import fsolve
import math
import matplotlib.pyplot as plt

#=============================================================================================================================
#PARAMETERS
#=============================================================================================================================

#I set the parameters:

beta = 0.9                    # discount factor
phi=0.7                       # intensity of pollutant capital in the production
x=0.05                        # the cost of investing in less pollutant capital
rho=0.5                       # pollutant and non-pollutant capital are imperfectsubstitutes in production
alpha = 0.3                   # the elasticity of output with respect input of capital/capital share in production,between 0/1
delta_p = 0.1                 # pollutant capital depreciation, between 0 and 1
delta_np=0.09                 # non pollutant-capital depreciation   
alpha_1= 0.5                  # impact of the non-pollutant capital on environment
alpha_2=  0.7                 # impact of the pollutant capital on environment
eta=0.9                       # impact of the environment in the utility function
A = 1.0                       # Total factor productivity, strictly positive
T = 20                        # As agents live for only three periods, assume that each period is 20 years.
n = np.array([1.0, 1.0, 0])   # exogenous labor supply, the individuals supply a unit of labor inellastically in the first two periods and 0.2 when they are retired.
k_init_np=0                   # the individuals are born without savings  k_{1,t}
k_init_p=0                    
k_last=0                      # the individuals don't save income in the last period of their life b_{4,t}
xi = 0.1                      # parameter for convergence of GE loop
r_init_np = 1 / beta - 1      # initial guess for the interest rate, the same for both
r_init_p = 1 / beta - 1  
#I assume that there is no population growth and no survival risk.

#The price of the output is equal to 1.

#=============================================================================================================================
#FUNCTIONS
#=============================================================================================================================

#Now, we need to define the different functions of the model.

#========================================
#HOUSEHOLDS
#========================================

'''
The utility function of the households is logarithmic and it depends on consumption and the environment.
'''
def environment(k_np,k_p,alpha_1,alpha_2):
    
    E=alpha_1*k_np-alpha_2*k_p
    
    return E

#The environment depends positively on non-pollutant capital and negatively of pollutant capital.

def utility(c, E): 
    
    u_c=np.log(c)+ eta*np.log(E)

    return u_c

#The utility has two arguments: the consumption and the environment.
#Then, the marginal utility can be writen as:

def mu(c, E):
    mu_c = (1/c)+ eta*(1/E)
    return mu_c

#The BC is defined as: c_{s,t}+k_np{s+1,t+1}+k_p{s+1,t+1}=w_t*n_{s,t}+(1+r_t_np)k_np{s,t} + (1+r_t_p)k_p{s,t} for all s and t.
#The wage and the interest rate don't have an age subindex, since both are the same for all individuals.
    

def budget_constraint(r_np,r_p, w, n, k_s_p,k_s_np, k_st1_p,k_st1_np, delta_p, delta_np):
  
    c = w * n + (1 + r_np-delta_np) * k_s_np + (1 + r_p-delta_p) * k_s_p  - k_st1_np-k_st1_p
    c>=0 #consumption cannot be negative
    return c

#The households have the capital instead of assets, for that reason the budget constraint is different from before. 
#Now, I adapt the first order condition to my problem:
def FOC(k_st1_p, k_st1_np, *args):
    beta, r_np,r_p, w, n, k_init_p,k_init_np,E = args
    k_s_p = np.append(k_init_p, k_st1_p)
    k_s_np= np.append(k_init_np, k_st1_np)
    k_sp1_p=np.append(k_st1_p, 0)
    k_sp1_np=np.append(k_st1_np, 0)
    c = budget_constraint(r_np,r_p, w, n, k_s_p,k_s_np, k_st1_p,k_st1_np, delta_p, delta_np)
    lhs_euler = eta*(1/E)*(alpha_1+alpha_2)
    rhs_euler = 1/c*((r_p-delta_p)-(r_np-delta_np))
    foc_errors = lhs_euler[:-1] - rhs_euler[1:]
    
    return foc_errors

#========================================
#FIRMS
#========================================

'''
This economy has perfectly competitive firms that rent investment capitals from individuals for real return and hire labor for real wage. 
We assume that firm use the total amount of labor and capital available to produce the otuput. Futhermore, the production is a Cobb-Douglas with CES. 
'''


#The total CES aggregate of capital:
def aggregate_capital(phi,k_s_p, k_s_np,rho):
    k=(phi*k_s_p**rho+(1-phi)*k_s_np**rho)**(1/rho)
    return k

#The labor supply is defined as L_t=n_{1,t}+n_{2,t}+n_{3,t} for all t. 
#Recall that the individuals supply inellastically the first two periods, in the last one they are retired and therefore they don't obtain any income from work. 
def labor(n):
    
    L = np.sum(n)

    return L

def production_function (alpha, K,L):
    Y=K**alpha*L**(1-alpha)
    return Y

def wage (r_np, r_p, alpha, delta,k,k_s_p,k_s_np):
    w = (k**alpha-r_p*k_s_p-(r_np+x)*k_s_np)
    return w

def capital(k_s):
    
    if k_s.ndim == 1:
        K = k_s.sum()
        
    if k_s.ndim == 2:
        K = k_s.sum(axis=1)
        
    return K

def interest_rate(phi, k_s_p, k_s_np,r_np, r_p, x, rho):
    r_p=((phi/(1-phi))*(k_s_np/k_s_p)**(1-rho))*(r_np+x)
    return r_p, r_np

#since there are two different interest rate
#========================================
#STEADY STATE
#========================================
'''
Once, I have studied the different equations for the firm's and HH's, we are going to solve for the steady state in this OLG economy. 
In steady state growth all variables, such as output, population, capital stock, saving, investment, and technical progress, grow at a constant rate.
'''
def solve_ss(r_init_np, r_init_p, beta, n, alpha, A, delta_np,delta_p, xi,k_s_np,k_s_p, r_np,r_p,k):
    
    
    distance = 0.01
    tolerance_error = 0.00001
    iteration = 0
    maximum_number_iterations = 300
    r_np = r_init_np #Initial guess
    r_p = r_init_p #Initial guess
    while (distance > tolerance_error) & (iteration < maximum_number_iterations):
       
        # We want to calculate the value for the wages:
        wage_ss = (r_np, r_p, alpha, delta_np, delta_p,k,k_s_p,k_s_np)
        
        #In this point we solve HH problem:
        
        k_sp1_p_guess = [0.05, 0.05] #choose an arbitrary initial guess for the steady-state distribution of wealth
        k_sp1_np_guess = [0.05, 0.05]
        result = opt.root(FOC, k_sp1_p_guess,k_sp1_np_guess, args=( beta, r_np, r_p, wage_ss, n, 0.0))
        #Find a root of a vector function.
        k_sp1 = result.x
        euler_errors = result.fun
    
        k_s = np.append(0.0, k_sp1)
        
        # Use market clearing condition for capital and labor, since the good market is redundant:
            
        L = labor(n)
        K = capital(k_s)
        
        # Obtain the interest rate for the following period:
        r_prime = interest_rate(L, K, alpha, A, delta_np, delta_p)
        
        
        # Look at the distance between the interest rate and the interest rate in the following period:
        distance = np.absolute(r_p - r_prime)
        distance_1= np.absolute(r_np - r_prime)
        print('Iteration = ', iteration, ', Distance = ', distance,
              ', r_p= ', r_p)
        print('Iteration = ', iteration, ', Distance = ', distance_1,
              ', r_np = ', r_np)
        # We update the interest rate each time.
        r_np= xi * r_prime + (1 - xi) * r_np #convex combination of the interest rate in the following period and the interest rate in this period, x_i is between  0 and 1
        r_p= xi * r_prime + (1 - xi) * r_p
        # We update the iteration counter each time.
        iteration += 1

    return r_np,r_p, k_sp1, euler_errors,L,K,wage_ss


#========================================
#TIME PATH
#========================================
'''
 Given time paths for interest rates and wages, this function generates matrices
 for the time path of the distribution of consumption, labor supply, savings,
 the corresponding Euler errors for the labor supply decision and the savings decisions...
 
Following the chapter 17 of Stokey and Lucas when they focus on overlapping generation model, 
we don't have to solve recursively for the policy functions iterating on individual value functions. 
We need recursively solve for the policy functions by iterating on the entire transition path 
of the endogenous objects in the economy.
'''

def solve_tp(r_path_init_np,r_path_init_p, beta, n, alpha, delta_np, delta_p, T, xi, k_sp1_p, k_sp_n, r_ss_np,r_ss,p
     k_sp1_p,k_sp1_np):
    
    #Here my mistake, it is supposed that the interest rate of both types of capital must be equal in equilibrium, but obviously I should obtain two types of capital,  therefore,
    #  I think that my mistake is in the line 180 of the code. I have checked in different ways but I obtain always the same problem.
    #So let the code in this way since I cannot find the mistake since the solution is only displayed if the whole code is correct as we have seen in
    #``OLG_3periods_RosannaGomar.py''!! 

    tp_distance = 7.0
    tp_tolerance = 0.00001
    tp_iterations = 0 
    tp_max_iterations = 300  #we set a maximum number of iterations, if after 300 iterations we don't get the expected result, we state that the time path is not solved.
                             #Imagine that this does not converge, we cannot have it running infinitely. 