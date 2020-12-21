# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:06:12 2020

@author: ROSANNA_PC
"""

#===================================================================
#II.5.1  GENERAL EQUILIBRIUM: The simple ABHI model economy
#===================================================================

'''For doing this exercise, it has been very useful:https://python.quantecon.org/aiyagari.html.
I have used their code but I have introduced some modifications to adapt to our exercise, for example,
we need to take into account that our utility function is CRRA and not logarithmic'''

#First, we import the libraries:
import numpy as np
import matplotlib.pyplot as plt
from quantecon.markov import DiscreteDP
from numba import jit

class Household:
    
    """
    This class takes the parameters that define a household asset accumulation
    problem and computes the corresponding reward and transition matrices R
    and Q required to generate an instance of DiscreteDP, and thereby solve
    for the optimal policy.

    Comments on indexing: We need to enumerate the state space S as a sequence
    S = {0, ..., n}.  To this end, (a_i, z_i) index pairs are mapped to s_i
    indices according to the rule

        s_i = a_i * z_size + z_i 

    To invert this map, use
    
        a_i = s_i // z_size  (integer division)
        z_i = s_i % z_size

    """

    rho=0.06
    sigma_y=0.4
    def __init__(self,
                 r=0.04,                                  # interest rate
                 w=1.0,                                   # wages
                 beta=1/(1+rho),                          # discount factor
                 a_min=1e-10,
                 pi=[[0.8, 0.2], [0.2, 0.8]],              # Markov chain
                 z_vals=[1-sigma_y, 1+sigma_y],           # exogenous states
                 a_max=18,
                 a_size=100):

        # Store values, set up grids over a and z
        
        self.r, self.w, self.beta = r, w, beta
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size

        self.pi = np.asarray(pi)
        self.z_vals = np.asarray(z_vals)
        self.z_size = len(z_vals)

        self.a_vals = np.linspace(a_min, a_max, a_size)
        self.n = a_size * self.z_size

        # Build the array Q
        self.Q = np.zeros((self.n, a_size, self.n))
        self.build_Q()

        # Build the array R
        self.R = np.empty((self.n, a_size))
        self.build_R()

    def set_prices(self, r, w):
        """
        Use this method to reset prices.  Calling the method will trigger a
        re-build of R.
        """
        self.r, self.w = r, w
        self.build_R()

    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.pi)

    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R, self.a_size, self.z_size, self.a_vals, self.z_vals, self.r, self.w)


# Do the hard work using JIT(JUST IN TIME)-ed functions:
    
sigma=5 
@jit(nopython=True)
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        a = a_vals[a_i]
        z = z_vals[z_i]
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = w * z + (1 + r) * a - a_new
            if c > 0:
                R[s_i, new_a_i] = (c**(1-sigma)-1)/(1-sigma) # CRRA Utility, this is a modification of the code
                
#R needs to be a matrix where R[s, a] is the reward at state s under action a.
@jit(nopython=True)
def populate_Q(Q, a_size, z_size, pi):
    n = a_size * z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for next_z_i in range(z_size):
                Q[s_i, a_i, a_i * z_size + next_z_i] = pi[z_i, next_z_i]

#Q needs to be a three-dimensional array where Q[s, a, s'] is the probability of transitioning to state s' when the current state is s and the current action is a

@jit(nopython=True)
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i * z_size + z_i]
    return a_probs

# Let’s compute the equilibrium, the following code draws aggregate supply and demand curves.
# Moreover, the intersection gives equilibrium interest rates and capital.

A = 1            # the technological progress is equal to 1
N = 1            # normalize the population size
α = 0.33         # alpha, contribution of the capital to the production function
rho=0.06
beta = 0.96      # discount factor
δ = 0.05         # depreciation rate


def r_to_w(r):
    
    """
    Equilibrium wages associated with a given interest rate r.
    """
    
    return A * (1 - α) * (A * α / (r + δ))**(α / (1 - α))

def rd(K):
    
    """
    Inverse demand curve for capital.  The interest rate associated with a
    given demand for capital K.
    """
    
    return A * α * (N / K)**(1 - α) - δ


def prices_to_capital_stock(am, r):
    
    """
    Map prices to the induced level of capital stock.
    
    Parameters:
    ----------
    
    am : Household
        An instance of an aiyagari_household.Household 
        
    r : float
        The interest rate
    """
    
    w = r_to_w(r)
    am.set_prices(r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, beta)
    
    # Compute the optimal policy
    results = aiyagari_ddp.solve(method='policy_iteration')
    
    # Compute the stationary distribution
    stationary_probs = results.mc.stationary_distributions[0]
    
    # Extract the marginal distribution for assets
    asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)
    
    # Return K
    return np.sum(asset_probs * am.a_vals)


# Create an instance of Household
am = Household(a_max=30)

# Use the instance to build a discrete dynamic program
am_ddp = DiscreteDP(am.R, am.Q, am.beta)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 100
r_vals = np.linspace(0.005, 0.04, num_points)

# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = prices_to_capital_stock(am, r)

##############################
#PLOT DEMAND AND SUPPLY OF K
#############################

fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='Supply of capital')
ax.plot(k_vals, rd(k_vals), color="red",lw=2, alpha=0.6, label='Demand for capital')
ax.grid()
ax.title.set_text('Demand and supply of capital')
ax.set_xlabel('Capital')
ax.set_ylabel('Interest rate')
ax.legend(loc='upper right')


plt.show()


# Report the endogenous distribution of wealth. 

#STEP 1: Stationary distribution of wealth. 
am_ddp = DiscreteDP(am.R, am.Q, am.beta)
results = am_ddp.solve(method='policy_iteration')

# Compute the stationary distribution
stationary_probs = results.mc.stationary_distributions[0]

# Extract the marginal distribution for assets
asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)

#############################
#PLOT
#############################
plt.figure()
plt.hist(asset_probs)
plt.title('Stationary distribution of assets(wealth)')


#Compare our result with the paper of Krueger, Mitman and Perri

#Gini's coefficient:
def gini(array):
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
        
    # Values cannot be 0:
    array += 0.0000001
    
    # Values must be sorted:
    array = np.sort(array)
    
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    
    # Number of array elements:
    n = array.shape[0]
    
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

#Now, we are going to observe how is the wealth distributed across the different groups of the population

order=sorted(asset_probs)
top_1=order[99]/np.sum(order)
top_10=np.sum(order[90:99])/np.sum(order)
bottom_50=np.sum(order[0:50])/np.sum(order)
middle=np.sum(order[50:90])/np.sum(order)

print('The GINI coefficient is '"{0:.4f}".format(gini(asset_probs)))
print('The 80/20 ratio is equal to '"{0:.4f}".format(np.percentile(asset_probs,80)/np.percentile(asset_probs,20)))
print('The aggregate wealth is equal to '"{0:.2f}".format(np.sum(asset_probs)))
print('The average assets is equal to '"{0:.4f}".format(np.mean(asset_probs)))
print('The standard deviation of assets is '"{0:.4f}".format(np.std(asset_probs)))
print('The top 1% shares '"{0:.2f}".format(top_1))
print('The top 10% shares '"{0:.2f}".format(top_10))
print('The bottom 50% shares '"{0:.2f}".format(bottom_50))
print('The middle class shares '"{0:.2f}".format(middle))



#%%

#===================================================================
#II.5.2  GENERAL EQUILIBRIUM: Aiyagari (1994)
#===================================================================

'''We need to repeat the same steps as before but now we use the parameters of Aiyagari (1994)'''


#First, we import the libraries:
    
import numpy as np
import matplotlib.pyplot as plt
from quantecon.markov import DiscreteDP
from numba import jit

class Household:
    
    """
    This class takes the parameters that define a household asset accumulation
    problem and computes the corresponding reward and transition matrices R
    and Q required to generate an instance of DiscreteDP, and thereby solve
    for the optimal policy.

    Comments on indexing: We need to enumerate the state space S as a sequence
    S = {0, ..., n}.  To this end, (a_i, z_i) index pairs are mapped to s_i
    indices according to the rule

        s_i = a_i * z_size + z_i 

    To invert this map, use
    
        a_i = s_i // z_size  (integer division)
        z_i = s_i % z_size

    """

    def __init__(self,
                 r=0.01,                # interest rate
                 w=1.0,                 # wages
                 beta=0.96,                # discount factor
                 a_min=0,
                 Π=[[0.5, 0.2, 0.1, 0.075, 0.075, 0.025, 0.025], [0.1, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05], [0.1, 0.1, 0.5, 0.1, 0.1, 0.05, 0.05], [0.05, 0.1, 0.1, 0.5, 0.1, 0.1, 0.05],[0.05, 0.05, 0.1, 0.1, 0.5, 0.1, 0.1],[0.05, 0.05, 0.1, 0.1, 0.1, 0.5, 0.1],[0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.5]],  # Markov chain
                 z_vals=[0.01,0.1,0.2,0.3,0.4,0.5,0.6],           # exogenous states
                 a_max=18,
                 a_size=100):

        # Store values, set up grids over a and z
        self.r, self.w, self.beta = r, w, beta
        self.a_min, self.a_max, self.a_size = a_min, a_max, a_size

        self.Π = np.asarray(Π)
        self.z_vals = np.asarray(z_vals)
        self.z_size = len(z_vals)

        self.a_vals = np.linspace(a_min, a_max, a_size)
        self.n = a_size * self.z_size

        # Build the array Q
        self.Q = np.zeros((self.n, a_size, self.n))
        self.build_Q()

        # Build the array R
        self.R = np.empty((self.n, a_size))
        self.build_R()

    def set_prices(self, r, w):
        """
        Use this method to reset prices.  Calling the method will trigger a
        re-build of R.
        """
        
        self.r, self.w = r, w
        self.build_R()

    def build_Q(self):
        populate_Q(self.Q, self.a_size, self.z_size, self.Π)

    def build_R(self):
        self.R.fill(-np.inf)
        populate_R(self.R, self.a_size, self.z_size, self.a_vals, self.z_vals, self.r, self.w)


# Do the hard work using JIT-ed functions

sigma=3 #In the paper, there are three diferent values {1,3,5}

@jit(nopython=True)
def populate_R(R, a_size, z_size, a_vals, z_vals, r, w):
    n = a_size * z_size
    for s_i in range(n):
        a_i = s_i // z_size
        z_i = s_i % z_size
        a = a_vals[a_i]
        z = z_vals[z_i]
        for new_a_i in range(a_size):
            a_new = a_vals[new_a_i]
            c = w * z + (1 + r) * a - a_new
            if c > 0:
                R[s_i, new_a_i] = (pow(c,1-sigma)-1)/(1-sigma)   # Utility
                

@jit(nopython=True)
def populate_Q(Q, a_size, z_size, Π):
    n = a_size * z_size
    for s_i in range(n):
        z_i = s_i % z_size
        for a_i in range(a_size):
            for next_z_i in range(z_size):
                Q[s_i, a_i, a_i * z_size + next_z_i] = Π[z_i, next_z_i]


@jit(nopython=True)
def asset_marginal(s_probs, a_size, z_size):
    a_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            a_probs[a_i] += s_probs[a_i * z_size + z_i]
    return a_probs

@jit(nopython=True)
def consumption_marginal(s_probs, a_size, z_size):
    c_probs = np.zeros(a_size)
    for a_i in range(a_size):
        for z_i in range(z_size):
            c_probs[a_i] += s_probs[a_i * z_size + z_i]
    return c_probs

# Let’s compute the equilibrium, the following code draws aggregate supply and demand curves.
# Moreover, the intersection gives equilibrium interest rates and capital.

A = 1            # the technological progress is equal to 1
N = 1            # normalize the population size
α = 0.33         # alpha, contribution of the capital to the production function
rho=0.06
beta = 0.96      # discount factor
δ = 0.05         # depreciation rate


def r_to_w(r):
    
    """
    Equilibrium wages associated with a given interest rate r.
    """
    
    return A * (1 - α) * (A * α / (r + δ))**(α / (1 - α))

def rd(K):
    
    """
    Inverse demand curve for capital.  The interest rate associated with a
    given demand for capital K.
    """
    
    return A * α * (N / K)**(1 - α) - δ


def prices_to_capital_stock(am, r):
    
    """
    Map prices to the induced level of capital stock.
    
    Parameters:
    ----------
    
    am : Household
        An instance of an aiyagari_household.Household 
    r : float
        The interest rate
    """
    
    w = r_to_w(r)
    am.set_prices(r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, beta)
    
    # Compute the optimal policy
    results = aiyagari_ddp.solve(method='policy_iteration')
    
    # Compute the stationary distribution
    stationary_probs = results.mc.stationary_distributions[0]
    
    # Extract the marginal distribution for assets
    asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)
    
    # Return K
    return np.sum(asset_probs * am.a_vals)


# Create an instance of Household
am = Household(a_max=30)

# Use the instance to build a discrete dynamic program
am_ddp = DiscreteDP(am.R, am.Q, am.beta)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 20
r_vals = np.linspace(0.005, 0.04, num_points)

# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = prices_to_capital_stock(am, r)


##############################
#PLOT DEMAND AND SUPPLY OF K
##############################

fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital',color = "green")
ax.plot(k_vals, rd(k_vals), lw=2, alpha=0.6, label='demand for capital',color = "black")
ax.title.set_text('Demand and supply of capital')
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.legend(loc='upper right')

plt.show()


#We need to find the distribution of assets and the stationary distribution
am_ddp = DiscreteDP(am.R, am.Q, am.beta)
results = am_ddp.solve(method='policy_iteration')

# Compute the stationary distribution
stationary_probs = results.mc.stationary_distributions[0]

# Extract the marginal distribution for assets
asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)

#############################
#PLOT
#############################

plt.figure()
plt.hist(asset_probs, color = "green")
plt.title('Stationary distribution of assets')


#Compare our result with the paper of Krueger, Mitman and Perri

#Gini's coefficient:


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
        
    # Values cannot be 0:
    array += 0.0000001
    
    # Values must be sorted:
    array = np.sort(array)
    
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    
    # Number of array elements:
    n = array.shape[0]
    
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))



#Now, we are going to observe how is the wealth distributed across the different groups of the population

order=sorted(asset_probs)
top_1=order[99]/np.sum(order)
top_10=np.sum(order[90:99])/np.sum(order)
bottom_50=np.sum(order[0:50])/np.sum(order)
middle=np.sum(order[50:90])/np.sum(order)

print('The GINI coefficient is '"{0:.4f}".format(gini(asset_probs)))
print('The 80/20 ratio is equal to '"{0:.4f}".format(np.percentile(asset_probs,80)/np.percentile(asset_probs,20)))
print('The aggregate wealth is equal to '"{0:.2f}".format(np.sum(asset_probs)))
print('The average assets is equal to '"{0:.4f}".format(np.mean(asset_probs)))
print('The standard deviation of assets is '"{0:.4f}".format(np.std(asset_probs)))
print('The top 1% shares '"{0:.2f}".format(top_1))
print('The top 10% shares '"{0:.2f}".format(top_10))
print('The bottom 50% shares '"{0:.2f}".format(bottom_50))
print('The middle class shares '"{0:.2f}".format(middle))


