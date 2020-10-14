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
from scipy.optimize import fsolve #Scipy builds on top of numpy to provide common tools for scientific programming such as:
#Linear algebra, numerical integration, interpolation, optimization, 
#distribution and random numbers gen, signal processing, etc.
import seaborn as sns #Seaborn: Tries to improve and enhance matplotlib. Specially good for data visualization or complex plots

"EXERCISE 1"
"Part a"
#Set the parameters of the model, h and theta are given, delta can be calculate since we know the ratio of investment-output and capital-output:
h=0.31 #labor supply
Theta=0.67 #labor share in the production function
Delta=0.0625 #depreciation rate

#Normalize the output to 1 and the capital to 4
k_1=4
y_1=1
i_1= Delta*k_1 #investment rate, since in steady state k_t=k_t+1=K
c_1=y_1-i_1 #the consumption is equal to the production minus the investment
z= k_1**((Theta-1)/Theta) / h  #productivity shock, the algebra is in the pdf
#We define two  functions the Production Function that have constant returns to scale and the Euler Equation
def F(k,z):
    return(k_1)**(1-Theta)*(z*h)**(Theta)


def Euler(Beta):
    f=(1-Delta+(1-Theta)*(k_1)**(-Theta)*(z*h)**Theta)-(1/Beta)
    return f

def u_prime(c): #it is not used
    return 1/c

solution_Beta=[]
range_1 = np.linspace(0.01,1,10)
Beta_1=1/ ( (1-Theta)*(k_1)**(-Theta) * (z*h)**Theta + 1 - Delta) #Once we have solved for the other parameters, we obtain the value of beta
# Observe different starting points
#create a loop:
for i in range_1: # to store results from a loop, might be useful to create and empty list and then append. 
	tmp = optimize.fsolve(Euler, i)
	solution_Beta.append(tmp)


#We are going to print the parameters and the variables in steady state.
    
print ("The productivity parameter,z=", "{0:.4f}".format(z))
print ("Beta=", "{0:.4f}".format(np.mean(solution_Beta)))
print ("Steady State for capital=", "{0:.2f}".format(k_1))
print ("Steady State for output=", "{0:.2f}".format(y_1))
print ("Steady State for consumption=", "{0:.2f}".format(c_1))
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
print ("New steady State for consumption=", "{0:.2f}".format(c_2))
print ("Delta", "{0:.4f}".format(Delta))

#I  have created a table to put together the results of part a and b.

Table = {'Parameters/Variables': [ 'z', 'c', 'i',  'y', 'k', 'h', 'Theta', 'Beta', 'Delta'], 'Steady State a': [z, c_1, i_1, y_1, k_1, h, Theta, Beta_1, Delta,], 'Steady State b': [z_2, c_2, i_2, y_2, k_2, h, Theta, Beta_1, Delta, ]}
Table_ab = pd.DataFrame(Table)
print(Table_ab)

"Part c"

#The marginal utility is:

def  consumption(c):
    return c  #since it has a logarithmic form

#The production function is defined as:
def y(k,z):
    return k**(1-Theta)*(z*h)**(Theta)

n=125 #the simulation has this number of periods 

"""
The important fact here is that we have two different euler equation as we have seen in part a (initial Euler Equation) and part b (Final Euler equation).
If we study the transition, we need another Euler equation for all periods between both steady states. Summarizing, we have three Euler equation, we need a sequence of capitals
for which these three equation hold for every period of time.

"""
def transition(k,n=n):
    k_0=k_1
    k_last=k_2
    #Set the initial and final condition
    k[0]=k_1
    k[n-1]=k_2 #recall that our first element in phyton is the 0.
    k_trans=np.zeros(n) #we create a vector of 0's for the values of capital during the transition, let's complete it:
    for i in range(0,n-2): #we create a loop and distinguish the three euler equations:
        if i==0:
            k_trans[i+1]=Beta_1*consumption(y(k_0,z_2)+(1-Delta)*k_0-k[i+1])*((1-Delta)+(1-Theta)*(1/(k[i+1]))**(Theta)*((y(k_0,z_2))/(k_0**(1-Theta))))-consumption(y(k[i+1],z_2)+(1-Delta)*k[i+1]-k[i+2])
        elif i==(n-2):
            k_trans[i+1]=Beta_1*consumption(y(k[i],z_2)+(1-Delta)*k[i]-k[i+1])*((1-Delta)+(1-Theta)*(1/(k[i+1])**(Theta)*((y(k[i],z_2))/(k[i]**(1-Theta)))))-consumption(y(k[i+1],z_2)+(1-Delta)*k[i+1]-k_last)
        else:
            k_trans[i+1]=Beta_1*consumption(y(k[i],z_2)+(1-Delta)*k[i]-k[i+1])*((1-Delta)+(1-Theta)*(1/(k[i+1])**(Theta)*((y(k[i],z_2))/(k[i]**(1-Theta)))))-consumption(y(k[i+1],z_2)+(1-Delta)*k[i+1]-k[i+2])
            
    return (k_trans)
    
x0=np.linspace(4,8,n) #Initial values, recall that the capital is 4 for part a and 8 for part b. 

#TRANSITION PATH FOR CAPITAL, OUTPUT AND SAVINGS:
trans_path_k=fsolve(transition, x0) 
trans_path_y=y(trans_path_k,z_2) 
trans_path_s=np.zeros(n) #we need to complete the vector of zeros

for i in range(0,n-1):
        trans_path_s[i]=trans_path_k[i+1]-(1-Delta)*trans_path_k[i] #the savings are equal to the investment and depends on the capital of the current period and the following one

trans_path_s[n-1]=trans_path_s[n-2] #the savings are equal in the last period and in the antepenultimate period since the horizon of time is finite

#TRANSITION PATH FOR CONSUMPTION AND LABOR
trans_path_cons=trans_path_y-trans_path_s #consumption is equal to the investment minus consumption
trans_path_labor=np.ones(n)*h #labor is exactly the same for all period since it is exogeneous.

#Doing this, we add the values of each variable in the first steady state.
trans_path_k=np.insert(trans_path_k,0,k_1)
trans_path_y=np.insert(trans_path_y,0,y_1)
trans_path_s=np.insert(trans_path_s,0,i_1)  #iss because investment equals savings in this models.
trans_path_cons=np.insert(trans_path_cons,0,c_1)
trans_path_labor=np.insert(trans_path_labor,0,h)

#TIME
time=np.array(list(range(0,(n+1))))


#GRAPHS
f, axs = plt.subplots(2,2,figsize=(15,10))
ax1 = plt.subplot(231)
ax1.plot(time, trans_path_s,'.', color='blue', linewidth=2)   
ax1.set_title('Transition path for savings', fontsize=10)
ax1.set_ylabel('Savings')
ax1.set_xlabel('Time')
ax2 = plt.subplot(232)
ax2.plot(time, trans_path_cons,'.', color='blue', linewidth=2)     
ax2.set_title('Transition path for consumption')
ax2.set_ylabel('Consumption')
ax2.set_xlabel('Time')
ax3 = plt.subplot(233)
ax3.plot(time, trans_path_labor, 'b-', linewidth=2)   
ax3.set_title('Transition path for labor')
ax3.set_ylabel('Labor supply')
ax3.set_xlabel('Time')
ax4 = plt.subplot(234)
ax4.plot(time, trans_path_y,'.', color='blue', linewidth=2)   
ax4.set_title('Transition path for output')
ax4.set_ylabel('Output')
ax4.set_xlabel('Time')
ax5 = plt.subplot(236)
ax5.plot(time, trans_path_k,'.', color='blue', linewidth=2)   
ax5.set_title('Transition path for capital')
ax5.set_ylabel('Capital stock')
ax5.set_xlabel('Time')

plt.show()

"PART D"

n2=116 #before n1=125, as phyton starts with 0 and the first 9 periods agents think that productivity is z2, we sustract these periods
def second_transition(k,n2=n2):
    k_0=trans_path_k[9]
    k_last=k_1
    k[0]=trans_path_k[9]
    k[n2-1]=k_1
    
    k_second_trans=np.zeros(n2) #the same as before, but now taking into account that the productivity is z and not z_2:
    for i in range(0,n2-2):
        if i==0:
            k_second_trans[i+1]=Beta_1*consumption(y(k_0,z)+(1-Delta)*k_0-k[i+1])*((1-Delta)+(1-Theta)*(1/(k[i+1]))**(Theta)*((y(k_0,z))/(k_0**(1-Theta))))-consumption(y(k[i+1],z)+(1-Delta)*k[i+1]-k[i+2])
        elif i==(n2-2):
            k_second_trans[i+1]=Beta_1*consumption(y(k[i],z)+(1-Delta)*k[i]-k[i+1])*((1-Delta)+(1-Theta)*(1/(k[i+1])**(Theta)*((y(k[i],z))/(k[i]**(1-Theta)))))-consumption(y(k[i+1],z)+(1-Delta)*k[i+1]-k_last)
        else:
            k_second_trans[i+1]=Beta_1*consumption(y(k[i],z)+(1-Delta)*k[i]-k[i+1])*((1-Delta)+(1-Theta)*(1/(k[i+1])**(Theta)*((y(k[i],z))/(k[i]**(1-Theta)))))-consumption(y(k[i+1],z)+(1-Delta)*k[i+1]-k[i+2])
            
    return (k_second_trans)

x02=np.linspace(6,4,n2)

#NEW TRANSITION PATH FOR CAPITAL, INCOME, SAVINGS, CONSUMPTION AND LABOR:
    
trans_path_k_2=fsolve(second_transition,x02)
trans_path_y_2=y(trans_path_k_2,z)

trans_path_s_2=np.zeros(n2)
for i in range(0,n2-1):
        trans_path_s_2[i]=trans_path_k_2[i+1]-(1-Delta)*trans_path_k_2[i]


trans_path_s_2[n2-1]=trans_path_s_2[n2-2] 
trans_path_cons_2=trans_path_y_2-trans_path_s_2 
trans_path_labor_2=np.ones(n2)*h 

trans_path_k_2=np.concatenate((trans_path_k[0:10],trans_path_k_2))   
trans_path_y_2=np.concatenate((trans_path_y[0:10],trans_path_y_2)) 
trans_path_s_2=np.concatenate((trans_path_s[0:10],trans_path_s_2)) 
trans_path_cons_2=np.concatenate((trans_path_cons[0:10],trans_path_cons_2)) 
trans_path_labor_2=np.concatenate((trans_path_labor[0:10],trans_path_labor_2))


#GRAPHS
f, axs = plt.subplots(2,2,figsize=(15,10))
ax1 = plt.subplot(231)
ax1.plot(time, trans_path_s_2,'.', color='red', linewidth=2)   
ax1.set_title('New transition path for savings', fontsize=10)
ax1.set_ylabel('Savings')
ax1.set_xlabel('Time')
ax2 = plt.subplot(232)
ax2.plot(time, trans_path_cons_2,'.', color='red', linewidth=2)     
ax2.set_title('New transition path for consumption')
ax2.set_ylabel('Consumption')
ax2.set_xlabel('Time')
ax3 = plt.subplot(233)
ax3.plot(time, trans_path_labor_2, 'r-', linewidth=2)   
ax3.set_title('New transition path for labor')
ax3.set_ylabel('Labor supply')
ax3.set_xlabel('Time')
ax4 = plt.subplot(234)
ax4.plot(time, trans_path_y_2,'.', color='red', linewidth=2)   
ax4.set_title('New transition path for output')
ax4.set_ylabel('Output')
ax4.set_xlabel('Time')
ax5 = plt.subplot(236)
ax5.plot(time, trans_path_k_2,'.', color='red', linewidth=2)   
ax5.set_title('New transition path for capital')
ax5.set_ylabel('Capital stock')
ax5.set_xlabel('Time')

plt.show()


