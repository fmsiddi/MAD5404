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

# x = ch_mesh
# y = x**2
# bary = barycentric_lagrange(x,y,'chebyshev')
# x = np.linspace(a,b,100)
# y = np.array([bary(x[i]) for i in range(len(x))])
# plt.plot(x,y)

# x = eq_mesh
# y = x**2
# bary = barycentric_lagrange(x,y,'equal')
# x = np.linspace(a,b,100)
# y = np.array([bary(x[i]) for i in range(len(x))])
# plt.plot(x,y)

# x_nodes = np.array([-.5,0,.5,1])
# y_nodes = np.array([.8,1,3,8])
# bary = barycentric_lagrange(x_nodes,y_nodes,'equal')
# x = np.linspace(a,b,100)
# y = np.array([bary(x[i]) for i in range(len(x))])
# plt.plot(x,y)

def divided_diff(x_arr,y_arr):
    n = len(x_arr)
    α = np.zeros(n)
    α[0] = y_arr[0]
    f = y_arr
    i = 1
    while i < n:
        f = np.array([(f[j]-f[j-1])/(x_arr[j+i-1]-x_arr[j-1]) for j in range(1,len(f))])
        α[i] = f[0]
        i += 1
    return α

def horner(x,α,x_arr):
    n = len(x_arr)
    # p = α[-1] + (x - x_arr[-1])
    p = α[-1]
    for i in range(n-2,-1,-1):
        p = p * (x - x_arr[i]) + α[i]
    return p

# x_nodes = np.array([-2,0,.5,1])
# y_nodes = np.array([-1,1,3,8])
# α = divided_diff(x_nodes,y_nodes)
# x = np.linspace(a,b,100)
# y = np.array([horner(x[i],α,x_nodes) for i in range(len(x))])
# plt.plot(x,y)

# x_nodes = np.array([1,2,3],dtype='float')
# y_nodes = np.array([1,4,9],dtype='float')
# α = divided_diff(x_nodes,y_nodes)
# x = np.linspace(a,b,100)
# y = np.array([horner(x[i],α,x_nodes) for i in range(len(x))])
# plt.plot(x,y)

def hermite_divided_diff(x_arr,y_arr,y_prime):
    z_arr = np.repeat(x_arr,2)
    f = np.repeat(y_arr,2)
    y_p = np.repeat(y_prime,2)
    n = len(z_arr)
    α = np.zeros(n)
    α[0] = y_arr[0]
    i = 1
    while i < n:
        f = np.array([(f[j]-f[j-1])/(z_arr[j+i-1]-z_arr[j-1]) if z_arr[j+i-1] != z_arr[j-1] else y_p[j] for j in range(1,len(f))])
        α[i] = f[0]
        i += 1
    return α

def hermite(x,α,x_arr):
    n = len(α)
    z_arr = np.repeat(x_arr,2)
    # p = α[-1] + (x - x_arr[-1])
    p = α[-1]
    for i in range(n-2,-1,-1):
        p = p * (x - z_arr[i]) + α[i]
    return p

# x_nodes = np.array([0,1,3])
# y_nodes = np.array([2,4,5])
# y_p_nodes = np.array([1,-1,-2])
# α = hermite_divided_diff(x_nodes,y_nodes,y_p_nodes)
# print(hermite(2,α,x_nodes))

def compute_M(x_arr,y_arr):
    n = len(x_arr)
    A = np.zeros((n,n))
    np.fill_diagonal(A,2)
    h = x_arr[1] - x_arr[0]
    μ = .5 # we set as constant since nodes are equidistant
    μ = np.tile(.5,n-1)
    μ[-1] = 0
    λ = np.tile(.5,n-1) # we set as constant since nodes are equidistant
    λ[0] = 0
    A += np.diag(λ,1) + np.diag(μ,-1)
    z = np.zeros(n)
    for i in range(1,n-1):
        z[i] = (6/(2*h)) * ((y_arr[i+1]-y_arr[i])/h - (y_arr[i]-y_arr[i-1])/h)
    return np.linalg.solve(A,z)

def cubic_interp(x,x_arr,y_arr,M):
    n = len(x_arr)
    h = x_arr[1] - x_arr[0]
    # a = np.array([M[i+1] - M[i]/(6*h) for i in range(0,n-1)])
    # b = np.array([M[i]/2 for i in range(0,n-1)])
    # c = np.array([(y_arr[i+1] - y_arr[i])/h - (M[i+1] + 2*M[i])*h/6 for i in range(0,n-1)])
    # d = y_arr[:-1]
    if (x < x_arr[0]) or (x > x_arr[-1]):
        return print('desired input value {} is outside of interpolation range'.format(x))
    elif x in x_arr:
        return y_arr[np.where(x==x_arr)][0]
    else:
        diff = x - x_arr
        i = np.where(diff>0)[0][-1]
        # return a[i]*(x-x_arr[i])**3 + b[i]*(x-x_arr[i])**2 + c[i]*(x-x_arr[i]) + d[i]
        first_term = M[i]/6*(((x-x_arr[i+1])**3)/(-h) - (x- x_arr[i+1])*(x_arr[i]-x_arr[i+1]))
        second_term = M[i+1]/6*(((x-x_arr[i])**3)/(-h) - (x- x_arr[i])*(x_arr[i]-x_arr[i+1]))
        third_term = (y_arr[i]*(x-x_arr[i+1]) - y_arr[i+1]*(x-x_arr[i]))/(-h)
        return first_term - second_term + third_term
    
x_nodes = np.array([1,2,3,4,5])
y_nodes = np.array([0,1,0,1,0])
M = compute_M(x_nodes,y_nodes)
p = cubic_interp(1.5,x_nodes,y_nodes,M)

a = 1
b = 5
plot_x = np.linspace(a,b,100)
y = [cubic_interp(x,x_nodes,y_nodes,M) for x in plot_x]
plt.plot(plot_x,y)



