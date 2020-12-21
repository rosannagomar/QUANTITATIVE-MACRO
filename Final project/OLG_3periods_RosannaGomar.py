# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 12:34:17 2020

@author: ROSANNA_PC
"""
#=============================================================================================================================
#IMPORT LIBRARIES
#=============================================================================================================================
#First of all, we import the libraries:
    
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

#The model is explained in the Appendix of the pdf. 

'''
My idea is introducing the climate change into the analysis of an overlapping generation model. In the two-period-lived agent,
young agents define the supply of capital in the following period due to the fact that old people will not be around. 
Then this greatly simplifies the two-period-lived agent model and makes its solution method much easier than OLG models with agents that
live for three or more periods.

As I want to distinguish between two types of capital in the production side and introduce a public good in the utility function(environment) that depends on
past decisions, I think that the best approximation can be programming an OLG of three periods. I have not had the opportunity to write an OLG model in Phyton,
so its better to start with the simple one with three generations and then, if I have time I will write the modifications of my model in Phyton.
'''
#=============================================================================================================================
#PARAMETERS
#=============================================================================================================================

#I set the parameters:

theta = 2                     # it can be interpreted in different ways:curvature of the utility funciton, co efficient of relative risk aversion (we have a CRRA utility function)
                              # or constant elasticity of intertemporal substitution
beta = 0.9                    # discount factor
alpha = 0.3                   # the elasticity of output with respect input of capital/capital share in production,between 0/1
delta = 0.1                   # capital depreciation, between 0 and 1
A = 1.0                       # Total factor productivity, strictly positive
T = 60                        # As agents live for only three periods, assume that each period is 20 years.
n = np.array([1.0, 1.0, 0])   # exogenous labor supply, the individuals supply a unit of labor inellastically in the first two periods and 0.2 when they are retired.
b_init=0                      # the individuals are born without savings  b_{1,t}
b_last=0                      # the individuals don't save income in the last period of their life b_{4,t}
xi = 0.1                      # parameter for convergence of GE loop
r_init = 1 / beta - 1         # initial guess for the interest rate

#I assume that there is no population growth and no survival risk.

#The price of the output is equal to 1.

#=============================================================================================================================
#FUNCTIONS
#=============================================================================================================================

#Now, we need to define the different functions of the model.

#========================================
#FIRMS
#========================================

'''
This economy has perfectly competitive firms that rent investment capital from individuals for real return and hire labor for real wage. 
We assume that firm use the total amount of labor and capital available to produce the otuput. Futhermore, the production is a Cobb-Douglas with CES. 
'''

#We calculate the FOC with respect to capital to obtain the interest rate, this can be seen in the Appendix.

def interest_rate(L, K, alpha, A, delta):
   
    #r = alpha * A * L** (1 - alpha) * K**(alpha-1)- delta     #I can write in a better way
    r = alpha * A * (L / K) ** (1 - alpha) - delta

    return r


#We state the other condition that characterizes the FOC's of the firm, that is the derivative with respect to labor. 

def wage(r, alpha, A, delta):
    
    #w=(1-alpha)*A*(K/L)**alpha                                #Rearranging:
    w = ((1 - alpha) * A * ((r + delta) / (alpha * A)) **(alpha / (alpha - 1)))

    return w

#The labor supply is defined as L_t=n_{1,t}+n_{2,t}+n_{3,t} for all t. 
#Recall that the individuals supply inellastically the first two periods, in the last one they are retired and therefore they don't obtain any income from work. 

def labor(n):
    
    L = np.sum(n)

    return L

#The aggregate capital is defined as K_t=b_{2,t}+b_{3,t} for all t.

def capital(b_s):
    
    if b_s.ndim == 1:
        K = b_s.sum()
        
    if b_s.ndim == 2:
        K = b_s.sum(axis=1)
        
    return K

#The Cobb Douglas production function is defined as:
    
def production(L,K,alpha):
    Y=A*K**alpha*L**(1-alpha)
    
    return Y

#where \alpha is the capital share and 1-\alpha the labor share.

#========================================
#HOUSEHOLDS
#========================================

'''
The utility function of the households is constant relative risk aversion (CRRA), as I have explained before 
the parameter theta can be interpreted in different ways such as: curvature of the utility function, 
coefficient of relative risk aversion or constant elasticity of intertemporal substitution.
I have followed the literature to give a value for the parameter theta.
'''

def utility(c, theta): 
    
    u_c=(c**(1-theta))/(1-theta)

    return u_c

#Then, the marginal utility can be writen as:

def mu(c, theta):
    mu_c = c ** -theta

    return mu_c


''' 
Now, let's analyze the FOC of the HH, as we can see in the pdf, we can simplify the problem substituting
the BC in the maximizatio problem and then do the derivative with respect to b_{2,t+1} and b_{3,t+2},
then we can reagrupate the expression substituting per consumption and we get something like this:
(1) u'(c1) = beta * (1 + r)* u'(c2) -> b_{2,t+1} #to obtain this we use the envelope theorem
(2) u'(c2) = beta * (1 + r)* u'(c3) -> b_{3,t+2}
'''


def FOC(b_st1, *args):
    beta, theta, r, w, n, b_init = args
    b_s = np.append(b_init, b_st1)
    b_sp1 = np.append(b_st1, 0)
    c = budget_constraint(r, w, n, b_s, b_sp1)
    mu_c = mu(c, theta)
    lhs_euler = mu_c
    rhs_euler = beta * (1+r) * mu_c
    foc_errors = lhs_euler[:-1] - rhs_euler[1:]
    
    return foc_errors

#The BC is defined as: c_{s,t}+b_{s+1,t+1}=w_t*n_{s,t}+(1+r_t)b_{s,t} for all s and t.
#The wage and the interest rate don't have an age subindex, since both are the same for all individuals.
    
def budget_constraint(r, w, n, b_s,b_st1):
  
    c = w * n + (1 + r) * b_s - b_st1
    c>=0 #consumption cannot be negative
    return c

#========================================
#STEADY STATE
#========================================

'''
Once, we have studied the different equations for the firm's and HH's, we are going to solve for the steady state in this OLG economy. 
In steady state growth all variables, such as output, population, capital stock, saving, investment, and technical progress, grow at a constant rate.
'''

def solve_ss(r_init, beta, theta, n, alpha, A, delta, xi):
    
    
    distance = 0.01
    tolerance_error = 0.00001
    iteration = 0
    maximum_number_iterations = 300
    r = r_init #Initial guess
    
    
    while (distance > tolerance_error) & (iteration < maximum_number_iterations):
       
        # We want to calculate the value for the wages:
        wage_ss = wage(r, alpha, A, delta)
        
        #In this point we solve HH problem:
        
        b_sp1_guess = [0.05, 0.05] #choose an arbitrary initial guess for the steady-state distribution of wealth
        result = opt.root(FOC, b_sp1_guess, args=( beta, theta, r, wage_ss, n, 0.0))
        #Find a root of a vector function.
        b_sp1 = result.x
        euler_errors = result.fun
    
        b_s = np.append(0.0, b_sp1)
        
        # Use market clearing condition for capital and labor, since the good market is redundant:
            
        L = labor(n)
        K = capital(b_s)
        
        # Obtain the interest rate for the following period:
        r_prime = interest_rate(L, K, alpha, A, delta)
        
        
        
        # Look at the distance between the interest rate and the interest rate in the following period:
        distance = np.absolute(r - r_prime)
        print('Iteration = ', iteration, ', Distance = ', distance,
              ', r = ', r)
        
        # We update the interest rate each time.
        r = xi * r_prime + (1 - xi) * r #convex combination of the interest rate in the following period and the interest rate in this period, x_i is between  0 and 1
       
        # We update the iteration counter each time.
        iteration += 1

    return r, b_sp1, euler_errors,L,K,wage_ss

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
def solve_tp(r_path_init, beta, theta, n, alpha, A, delta, T, xi, b_sp1_pre, r_ss,
     b_sp1_ss):

    tp_distance = 7.0
    tp_tolerance = 0.00001
    tp_iterations = 0 
    tp_max_iterations = 300  #we set a maximum number of iterations, if after 300 iterations we don't get the expected result, we state that the time path is not solved.
                             #Imagine that this does not converge, we cannot have it running infinitely. 
                             
    r_path = np.append(r_path_init, np.ones(2) * r_ss)
    
    #Once we have the transition path for aggregate capital, we can determine the transition path for the interest rate and the wage
    
    while (tp_distance > tp_tolerance) & (tp_iterations < tp_max_iterations):
        #Once we obtain the r_path immediately we obtain the w_path since the others elements are parameters.
        w_path = wage(r_path, alpha, A, delta)
        
        # Solve the problem of the households:
        
        b_sp1_mat = np.zeros((T + 2, 2))
        euler_errors_mat = np.zeros((T + 2, 2))
        
        # solve upper right corner
        foc_args = (beta, theta, r_path[:2], w_path[:2], n[-2:],
                    b_sp1_pre[0])
        
        b_sp1_guess = b_sp1_ss[-1] #the guess depends on the steady state
        
        result = opt.root(FOC, b_sp1_guess, args=foc_args) #to find a root of a vector function.
        b_sp1_mat[0, -1] = result.x
        euler_errors_mat[0, -1] = result.fun
        
        # solve all full lifetimes
        DiagMaskb = np.eye(2, dtype=bool)  #boolean identity matrix
        for t in range(T):
            foc_args = (beta, theta, r_path[t:t+3], w_path[t:t+3], n,
                        0.0)
            b_sp1_guess = b_sp1_ss #only other restriction on the initial transition path for aggregate capital is that it equal the steady-state level
            result = opt.root(FOC, b_sp1_guess, args=foc_args) #to find a root of a vector function.
            b_sp1_mat[t:t+2, :] = (DiagMaskb * result.x +
                                   b_sp1_mat[t:t+2, :])
            euler_errors_mat[t:t+2, :] = (
                DiagMaskb * result.fun + euler_errors_mat[t:t+2, :])
        
        # create a b_s_mat
        b_s_mat = np.zeros((T, 3))
        b_s_mat[0, 1:] = b_sp1_pre
        b_s_mat[1:, 1:] = b_sp1_mat[:T-1, :]
        
        # Use the market clearing for labor and capital, recall that the good market is redundant.
        L_path = np.ones(T) * labor(n)
        K_path = capital(b_s_mat) 
        
        #Once we have the path for labor and capital, we can calculate the income path:
        Y_path=A*K_path**(alpha)*L_path**(1-alpha)
        
        # find implied r
        r_path_prime = interest_rate(L_path, K_path, alpha, A, delta)
        
   
        # We need to check the distance in absolute value between r' and r:
        tp_distance = np.absolute(r_path[:T] - r_path_prime[:T]).max()
        print('Iteration = ', tp_iterations, ', Distance = ', tp_distance)
        
        # We update the new value of the interest rate:
        
        r_path[:T] = xi * r_path_prime[:T] + (1 - xi) * r_path[:T]
        
        # We update the iteration counter each time. 
        tp_iterations += 1

    if tp_iterations < tp_max_iterations:
        print('The time path SOLVED!!!!')
    else:
        print('The time path DID NOT SOLVE!!!!')


    return r_path[:T],w_path[:T], K_path[:T] , L_path[:T],  Y_path[:T], euler_errors_mat[:T, :]


# SOLUTION TO THE PARAMETERS OF THE STEADY SS:
ss_params = (beta, theta, n, alpha, A, delta, xi)


#SOLUTION TO THE VARIABLES OF THE STEADY SS:
r_ss, b_sp1_ss, euler_errors_ss,L_ss,K_ss,wage_ss = solve_ss(r_init, beta, theta, n, alpha, A, delta, xi)

#Once we have this, we can calculate the values of other variables such as:
    
I_ss=delta*K_ss                                      #investment
C_ss = wage_ss * L_ss + (1 + r_ss) * K_ss - K_ss     #consumption
Y_ss=C_ss+I_ss                                       #output
v=K_ss/Y_ss                                          #capital/output
m=delta* (K_ss/Y_ss )                                #investment/output

# SOLUTION TO THE TIME PATH:
r_path_init = np.ones(T) * r_ss                      #our initial guess depends on the steady state
b_sp1_pre = 1.1 * b_sp1_ss                           #initial distribiton of savings 

# NOTE: if the initial distribution of savings is the same as the SS value, then the path of interest rates should equal the SS value in
# each period!

tp_params = (beta, theta, n, alpha, A, delta, T, xi, b_sp1_pre, r_ss, b_sp1_ss)

r_path, w_path, K_path, L_path, Y_path,  euler_errors_path = solve_tp(r_path_init, beta, theta, n, alpha, A, delta, T, xi, b_sp1_pre, r_ss, b_sp1_ss)

#The same as before, now we can compute the consumption and the investment path:
C_path=w_path[:T]*L_path[:T]+(1 + r_path[:T]) * K_path[:T] - K_path[:T]
I_path=Y_path[:T]-C_path
U_path=utility(C_path,theta)
#We print the results to have a general overview:

print ("Coefficient of relative risk aversion=", "{0:.4f}".format(theta))
print ("Discount factor=", "{0:.4f}".format(beta))
print ("Capital share in production=", "{0:.2f}".format(alpha))
print ("Delta=", "{0:.4f}".format(delta))
print('SS interest rate is ', r_ss)
print('Maximum Euler error in the SS is ',
      np.absolute(euler_errors_ss).max())
print ("SS capital=", "{0:.2f}".format(K_ss))
print ("SS consumption=", "{0:.2f}".format(C_ss))
print ("SS investment=", "{0:.2f}".format(I_ss))
print ("SS output=", "{0:.2f}".format(Y_ss))
print ("capital/output=", "{0:.2f}".format(v))
print ("investment/output", "{0:.2f}".format(m))

print('Maximum Euler error along the time path is ', np.absolute(euler_errors_path).max())

#========================================
#PLOTS
#========================================

#We define a line with the value of the steady state for each variable:
    
line_k= np.ones(T)*K_ss             #capital
line_r=np.ones(T)*r_ss              #interest rate
line_w=np.ones(T)*wage_ss           #wage
line_L=np.ones(T)*L_ss              #labor
line_Y=np.ones(T)*Y_ss              #output
line_C=np.ones(T)*C_ss              #consumption
line_I=np.ones(T)*I_ss              #investment


#We define the PATHS for each variable:
   

    
#INTEREST RATE:
plt.plot(np.arange(T), r_path,color='tab:green',linewidth=3) 
plt.plot(np.arange(T), line_r, 'g--', linewidth=0.5)
plt.title('Path of real interest rates',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#WAGE:
plt.plot(np.arange(T), w_path,color='tab:cyan',linewidth=3) 
plt.plot(np.arange(T), line_w, 'c--', linewidth=0.5)
plt.title('Path of wages',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#CAPITAL:
plt.plot(np.arange(T), K_path,color='tab:blue',linewidth=3) 
plt.plot(np.arange(T), line_k, 'b--', linewidth=0.5)
plt.title('Path of capital',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#LABOR:
plt.plot(np.arange(T), L_path, color='tab:red',linewidth=3) 
plt.plot(np.arange(T), line_L, 'r--', linewidth=0.5)
plt.title('Path of labor',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#OUTPUT:
plt.plot(np.arange(T), Y_path, color='tab:purple',linewidth=3) 
plt.plot(np.arange(T), line_Y, 'm--', linewidth=0.5)
plt.title('Path of output',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#INVESTMENT:

plt.plot(np.arange(T), I_path, color='y',linewidth=3) 
plt.plot(np.arange(T), line_I, 'y--', linewidth=0.5)
plt.title('Path of investment',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()


#CONSUMPTION:

plt.plot(np.arange(T), C_path, color='k',linewidth=3) 
plt.plot(np.arange(T), line_C, 'k--', linewidth=0.5)
plt.title('Path of consumption',fontsize=10)
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()

#CONSUMPTION, PRODUCTION, INVESTMENT AND CAPITAL TOGETHER:

fig = plt.figure(figsize=(8,6))

plt.plot(np.arange(T), C_path, 'r', label='Consumption',linewidth=1.5)
plt.plot(np.arange(T), Y_path, color='y', label='Output',linewidth=1.5)
plt.plot(np.arange(T), I_path, color='b', label='Investment',linewidth=1.5)
plt.plot(np.arange(T), K_path, color='g', label='Capital',linewidth=1.5)
plt.plot(np.arange(T), line_k, 'g--', linewidth=0.5)
plt.plot(np.arange(T), line_Y, 'y--',linewidth=0.5 )
plt.plot(np.arange(T), line_C, 'r--',linewidth=0.5 )
plt.plot(np.arange(T), line_I, 'b--',linewidth=0.5 )
plt.xlabel('Time')
plt.ylabel('Savings, consumption, output and capital')
plt.xlim(0, 20)
plt.legend(loc = 'upper right')
plt.title('Transition', fontsize=16)
plt.show()