import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def func1(x):
    return (x-2)**9

def func2(x):
    return 1/(1+x**2)

def chebyshev_mesh(a,b,n):
    z = np.array([np.cos((2*i+1)*np.pi/(2*n)) for i in range(n)]) # formula is slightly altered since n represents number of nodes
    l_term = (a+b)/2
    r_term = (b-a)/2
    return np.array([l_term + z[i]*r_term for i in range(n)])

def monomial(x_arr,y_arr):
    n = len(x_arr)
    matrix = np.array([[x_arr[i]**j for j in range (n)] for i in range(n)])
    return np.linalg.solve(matrix,y_arr)

def binom_coeff(n,k):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

# preprocessing for barycentric 2
def barycentric_weights(h,n,mesh_type):
    β = np.zeros(n)
    if mesh_type == 'equal':
        denom = math.factorial(n-1)*h**(n-1)
    for i in range(n):
        if mesh_type == 'chebyshev':
            β[i] = ((-1)**i)*np.sin((2*i+1)*np.pi/(2*n)) # formula is slightly altered since n represents number of nodes
        elif mesh_type == 'equal':
            β[i] = (binom_coeff(n-1,i)*(-1)**(n-1-i))/denom
            # if i == 0:
            #     β[i] = 1
            # else:
            #     β[i] = -β[i-1]*(n-1-i)/(i+1) # formula is slightly altered since n represents number of nodes
    return β

def barycentric_lagrange(x_arr,y_arr,mesh_type):    
    n = len(x_arr)
    h = x_arr[1] - x_arr[0]
    β_arr = barycentric_weights(h,n,mesh_type)
    # print('weights:',β_arr)
    return lambda z: np.sum(y_arr*β_arr/(z-x_arr))/np.sum(β_arr/(z-x_arr)) if z not in x_arr else y_arr[np.where(x_arr==z)[0][0]]

# x = np.array([1,2,3],dtype='float')
# y = np.array([1,4,9],dtype='float')
# bary = barycentric_lagrange(x,y,'equal')

a = -5
b = 5
n = 3
eq_mesh = np.linspace(a,b,n)
ch_mesh = chebyshev_mesh(a,b,n)

x = ch_mesh
y = x**2
bary = barycentric_lagrange(x,y,'chebyshev')

x = np.linspace(a,b,100)
y = np.array([bary(x[i]) for i in range(len(x))])
plt.plot(x,y)

def divided_diff(x_arr,y_arr):
    n = len(x_arr)
    α = np.zeros(n)
    return None

