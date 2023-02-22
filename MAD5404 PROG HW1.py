import numpy as np
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# FIRST TEST FUNCTION
def func1(x):
    return (x-2)**9

# SECOND TEST FUNCTION
def func2(x):
    return 1/(1+x**2)

def chebyshev_mesh(a,b,n):
    z = np.array([np.cos((2*i+1)*np.pi/(2*n)) for i in range(n)]) # formula is slightly altered since n represents number of nodes
    l_term = (a+b)/2
    r_term = (b-a)/2
    return np.array([l_term + z[i]*r_term for i in range(n)])

# MONOMIAL METHOD: TAKES ARRAY OF X NODES AND Y VALUES AND AN INPUT X AND RETURNS VALUE OF INTERPOLATION POLYNOMIAL AT X
def monomial(x,x_arr,y_arr):
    n = len(x_arr)
    matrix = np.array([[x_arr[i]**j for j in range (n)] for i in range(n)])
    a = np.linalg.solve(matrix,y_arr) # NOTE: I USE THE NUMPY LIBRARY TO SOLVE THE LINEAR SYSTEM
    p = 0
    for i in range(len(a)):
        p += a[i]*(x**i)
    return p

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

# RETURNS ANONYMOUS FUNCTION THAT WILL TAKE AN INPUT X AND RETURN VALUE OF INTERPOLATION POLYNOMIAL AT X
def barycentric_lagrange(x_arr,y_arr,mesh_type):    
    n = len(x_arr)
    h = x_arr[1] - x_arr[0]
    β_arr = barycentric_weights(h,n,mesh_type) # COMPUTE BARYCENTRIC WEIGHTS
    return lambda z: np.sum(y_arr*β_arr/(z-x_arr))/np.sum(β_arr/(z-x_arr)) if z not in x_arr else y_arr[np.where(x_arr==z)[0][0]]

# x = np.array([1,2,3],dtype='float')
# y = np.array([1,4,9],dtype='float')
# bary = barycentric_lagrange(x,y,'equal')

# a = -5
# b = 5
# n = 3
# eq_mesh = np.linspace(a,b,n)
# ch_mesh = chebyshev_mesh(a,b,n)

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

# COMPUTES EACH f[...] AT EACH LEVEL, KEEP THE ONE WE WILL USE AS THE COEFFICIENT AND STORE IT IN α, THEN REPEAT
# RETURNS AN ARRAY OF COEFFICIENTS WE WILL USE IN THE NEWTON FORM OF INTERPOLATING POLYNOMIAL
def divided_diff(x_arr,y_arr):
    n = len(x_arr)
    α = np.zeros(n)
    α[0] = y_arr[0]
    f = y_arr
    i = 1
    while i < n:
        f = np.array([(f[j]-f[j-1])/(x_arr[j+i-1]-x_arr[j-1]) for j in range(1,len(f))]) # DIVIDED DIFFERENCE
        α[i] = f[0]
        i += 1
    return α

# TAKES INPUT X, ARRAY OF COEFFICIENTS, AND ARRAY OF X NODES AND RETURNS VALUE OF INTERPOLATING POLYNOMIAL AT X
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

# TAKES AN ARRAY OF X NODES, VALUES OF FUNCTION, AND VALUES OF DERIVATIVES, ALL AT THE X NODES
# THEN COMPUTES DIVIDED DIFFERENCE USING THE HERMITE RULE
def hermite_divided_diff(x_arr,y_arr,y_prime):
    z_arr = np.repeat(x_arr,2) # DUPLICATE X VALUES TO GET 2N VALUES
    f = np.repeat(y_arr,2) # DUPLICATE Y VALUES TO GET 2N VALUES
    y_p = np.repeat(y_prime,2)
    n = len(z_arr)
    α = np.zeros(n)
    α[0] = y_arr[0]
    i = 1
    # DIVIDED DIFFERENCE
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

# COMPUTE M VALUES FOR NATURAL CUBIC SPLINES. RETURNS ARRAY OF M VALUES
def compute_M(x_arr,y_arr):
    n = len(x_arr)
    A = np.zeros((n,n))
    np.fill_diagonal(A,2)
    h = x_arr[1] - x_arr[0] # ASSUMES CONSTANT STEP SIZE
    μ = .5 # we set as constant since nodes are equidistant
    μ = np.tile(.5,n-1)
    μ[-1] = 0
    λ = np.tile(.5,n-1) # we set as constant since nodes are equidistant
    λ[0] = 0
    A += np.diag(λ,1) + np.diag(μ,-1)
    z = np.zeros(n)
    for i in range(1,n-1):
        z[i] = (6/(2*h)) * ((y_arr[i+1]-y_arr[i])/h - (y_arr[i]-y_arr[i-1])/h)
    return np.linalg.solve(A,z) # NOTE: I USE THE NUMPY LIBRARY TO SOLVE THE LINEAR SYSTEM

# TAKES X INPUT, X NODES, FUNCTION VALUES, AND M VALUES.
# RETURNS VALUE OF CORRESPONDING CUBIC SPLINE AT X. THE CORRECT SPLINE FUNCTION TO USE IS DETERMINED IN THE IF STATEMENT
def cubic_interp(x,x_arr,y_arr,M):
    h = x_arr[1] - x_arr[0]
    if (x < x_arr[0]) or (x > x_arr[-1]):
        return print('desired input value {} is outside of interpolation range'.format(x))
    elif x in x_arr:
        return y_arr[np.where(x==x_arr)][0]
    else:
        diff = x - x_arr
        i = np.where(diff>0)[0][-1] # WE DETERMINE THE INDEX OF THE CUBIC SPLINE TO USE
        # THE BELOW COMPOSITION WAS TAKEN FROM https://towardsdatascience.com/numerical-interpolation-natural-cubic-spline-52c1157b98ac
        first_term = M[i]/6*(((x-x_arr[i+1])**3)/(-h) - (x- x_arr[i+1])*(x_arr[i]-x_arr[i+1]))
        second_term = M[i+1]/6*(((x-x_arr[i])**3)/(-h) - (x- x_arr[i])*(x_arr[i]-x_arr[i+1]))
        third_term = (y_arr[i]*(x-x_arr[i+1]) - y_arr[i+1]*(x-x_arr[i]))/(-h)
        return first_term - second_term + third_term
    
# x_nodes = np.array([1,2,3,4,5])
# y_nodes = np.array([0,1,0,1,0])
# M = compute_M(x_nodes,y_nodes)
# p = cubic_interp(1.5,x_nodes,y_nodes,M)

# a = 1
# b = 5
# plot_x = np.linspace(a,b,100)
# y = [cubic_interp(x,x_nodes,y_nodes,M) for x in plot_x]
# plt.plot(plot_x,y)

#%%

# PARAMETERS

a = -5
b = 5
n_small = 3
n_med = 5
n_big = 10

#%%

# FUNCTION 1, SMALL NUMBER OF EQUIDISTANT NODES
f_mesh = np.linspace(a,b,101)
f1_fine = func1(f_mesh)

fig1, ((ax1,ax2,ax3)) = plt.subplots(1,3, sharey=True, figsize=(20,7))
fig1.suptitle('f(x) = (x-2)^9 (Equidistant Mesh)', fontsize=14)

ax1.set_title('n = {}'.format(n_small))
ax1.plot(f_mesh,f1_fine, label='f(x)')

eq_mesh = np.linspace(a,b,n_small)
f1 = func1(eq_mesh)

mono_f1 = np.array([monomial(x,eq_mesh,f1) for x in f_mesh])
ax1.plot(f_mesh, mono_f1, '-D', label='monomial', markevery=np.searchsorted(f_mesh,eq_mesh))

bary = barycentric_lagrange(eq_mesh,f1,'equal')
lag_f1 = np.array([bary(x) for x in f_mesh])
ax1.plot(f_mesh, lag_f1, '-D', label='barycentric lagrange', markevery=np.searchsorted(f_mesh,eq_mesh))

α = divided_diff(eq_mesh,f1)
newt_f1 = np.array([horner(x,α,eq_mesh) for x in f_mesh])
ax1.plot(f_mesh, newt_f1, '-D', label='newton', markevery=np.searchsorted(f_mesh,eq_mesh))

f1_prime = np.array([9*(x-2)**8 for x in eq_mesh])
α = hermite_divided_diff(eq_mesh,f1,f1_prime)
herm_f1 = np.array([hermite(x,α,eq_mesh) for x in f_mesh])
ax1.plot(f_mesh, herm_f1, '-D', label='hermite', markevery=np.searchsorted(f_mesh,eq_mesh))

M = compute_M(eq_mesh,f1)
cub_f1 = [cubic_interp(x,eq_mesh,f1,M) for x in f_mesh]
ax1.plot(f_mesh, cub_f1, '-D', label='natural cubic splines', markevery=np.searchsorted(f_mesh,eq_mesh))

ax1.legend()

plt.tight_layout()

# FUNCTION 1, MEDIUM NUMBER OF EQUIDISTANT NODES
ax2.set_title('n = {}'.format(n_med))
ax2.plot(f_mesh,f1_fine, label='f(x)')

eq_mesh = np.linspace(a,b,n_med)
f1 = func1(eq_mesh)

mono_f1 = np.array([monomial(x,eq_mesh,f1) for x in f_mesh])
ax2.plot(f_mesh, mono_f1, '-D', label='monomial', markevery=np.searchsorted(f_mesh,eq_mesh))

bary = barycentric_lagrange(eq_mesh,f1,'equal')
lag_f1 = np.array([bary(x) for x in f_mesh])
ax2.plot(f_mesh, lag_f1, '-D', label='barycentric lagrange', markevery=np.searchsorted(f_mesh,eq_mesh))

α = divided_diff(eq_mesh,f1)
newt_f1 = np.array([horner(x,α,eq_mesh) for x in f_mesh])
ax2.plot(f_mesh, newt_f1, '-D', label='newton', markevery=np.searchsorted(f_mesh,eq_mesh))

f1_prime = np.array([9*(x-2)**8 for x in eq_mesh])
α = hermite_divided_diff(eq_mesh,f1,f1_prime)
herm_f1 = np.array([hermite(x,α,eq_mesh) for x in f_mesh])
ax2.plot(f_mesh, herm_f1, '-D', label='hermite', markevery=np.searchsorted(f_mesh,eq_mesh))

M = compute_M(eq_mesh,f1)
cub_f1 = [cubic_interp(x,eq_mesh,f1,M) for x in f_mesh]
ax2.plot(f_mesh, cub_f1, '-D', label='natural cubic splines', markevery=np.searchsorted(f_mesh,eq_mesh))

ax2.legend()

plt.tight_layout()

# FUNCTION 1, LARGE NUMBER OF EQUIDISTANT NODES
ax3.set_title('n = {}'.format(n_big))
ax3.plot(f_mesh,f1_fine, label='f(x)')

eq_mesh = np.linspace(a,b,n_big)
f1 = func1(eq_mesh)

mono_f1 = np.array([monomial(x,eq_mesh,f1) for x in f_mesh])
ax3.plot(f_mesh, mono_f1, '-D', label='monomial', markevery=np.searchsorted(f_mesh,eq_mesh))

bary = barycentric_lagrange(eq_mesh,f1,'equal')
lag_f1 = np.array([bary(x) for x in f_mesh])
ax3.plot(f_mesh, lag_f1, '-D', label='barycentric lagrange', markevery=np.searchsorted(f_mesh,eq_mesh))

α = divided_diff(eq_mesh,f1)
newt_f1 = np.array([horner(x,α,eq_mesh) for x in f_mesh])
ax3.plot(f_mesh, newt_f1, '-D', label='newton', markevery=np.searchsorted(f_mesh,eq_mesh))

f1_prime = np.array([9*(x-2)**8 for x in eq_mesh])
α = hermite_divided_diff(eq_mesh,f1,f1_prime)
herm_f1 = np.array([hermite(x,α,eq_mesh) for x in f_mesh])
ax3.plot(f_mesh, herm_f1, '-D', label='hermite', markevery=np.searchsorted(f_mesh,eq_mesh))

M = compute_M(eq_mesh,f1)
cub_f1 = [cubic_interp(x,eq_mesh,f1,M) for x in f_mesh]
ax3.plot(f_mesh, cub_f1, '-D', label='natural cubic splines', markevery=np.searchsorted(f_mesh,eq_mesh))

ax3.legend()

plt.tight_layout()




# FUNCTION 2, SMALL NUMBER OF EQUIDISTANT NODES
f_mesh = np.linspace(a,b,101)
f2_fine = func2(f_mesh)

fig2, ((ax4,ax5,ax6)) = plt.subplots(1,3, sharey=True, figsize=(20,7))
fig2.suptitle('f(x) = 1/(1 + x^2) (Equidistant Mesh)', fontsize=14)

ax4.set_title('n = {}'.format(n_small))
ax4.plot(f_mesh,f2_fine, label='f(x)')

eq_mesh = np.linspace(a,b,n_small)
f2 = func2(eq_mesh)

mono_f2 = np.array([monomial(x,eq_mesh,f2) for x in f_mesh])
ax4.plot(f_mesh, mono_f2, '-D', label='monomial', markevery=np.searchsorted(f_mesh,eq_mesh))

bary = barycentric_lagrange(eq_mesh,f2,'equal')
lag_f2 = np.array([bary(x) for x in f_mesh])
ax4.plot(f_mesh, lag_f2, '-D', label='barycentric lagrange', markevery=np.searchsorted(f_mesh,eq_mesh))

α = divided_diff(eq_mesh,f2)
newt_f2 = np.array([horner(x,α,eq_mesh) for x in f_mesh])
ax4.plot(f_mesh, newt_f2, '-D', label='newton', markevery=np.searchsorted(f_mesh,eq_mesh))

# f2_prime = np.array([9*(x-2)**8 for x in eq_mesh])
# α = hermite_divided_diff(eq_mesh,f2,f2_prime)
# herm_f2 = np.array([hermite(x,α,eq_mesh) for x in f_mesh])
# ax4.plot(f_mesh, herm_f2, '-D', label='hermite', markevery=np.searchsorted(f_mesh,eq_mesh))

M = compute_M(eq_mesh,f2)
cub_f2 = [cubic_interp(x,eq_mesh,f2,M) for x in f_mesh]
ax4.plot(f_mesh, cub_f2, '-D', label='natural cubic splines', markevery=np.searchsorted(f_mesh,eq_mesh))

ax4.legend()

plt.tight_layout()

# FUNCTION 2, MEDIUM NUMBER OF EQUIDISTANT NODES
ax5.set_title('n = {}'.format(n_med))
ax5.plot(f_mesh,f2_fine, label='f(x)')

eq_mesh = np.linspace(a,b,n_med)
f2 = func2(eq_mesh)

mono_f2 = np.array([monomial(x,eq_mesh,f2) for x in f_mesh])
ax5.plot(f_mesh, mono_f2, '-D', label='monomial', markevery=np.searchsorted(f_mesh,eq_mesh))

bary = barycentric_lagrange(eq_mesh,f2,'equal')
lag_f2 = np.array([bary(x) for x in f_mesh])
ax5.plot(f_mesh, lag_f2, '-D', label='barycentric lagrange', markevery=np.searchsorted(f_mesh,eq_mesh))

α = divided_diff(eq_mesh,f2)
newt_f2 = np.array([horner(x,α,eq_mesh) for x in f_mesh])
ax5.plot(f_mesh, newt_f2, '-D', label='newton', markevery=np.searchsorted(f_mesh,eq_mesh))

# f2_prime = np.array([9*(x-2)**8 for x in eq_mesh])
# α = hermite_divided_diff(eq_mesh,f2,f2_prime)
# herm_f2 = np.array([hermite(x,α,eq_mesh) for x in f_mesh])
# ax5.plot(f_mesh, herm_f2, '-D', label='hermite', markevery=np.searchsorted(f_mesh,eq_mesh))

M = compute_M(eq_mesh,f2)
cub_f2 = [cubic_interp(x,eq_mesh,f2,M) for x in f_mesh]
ax5.plot(f_mesh, cub_f2, '-D', label='natural cubic splines', markevery=np.searchsorted(f_mesh,eq_mesh))

ax5.legend()

plt.tight_layout()

# FUNCTION 2, LARGE NUMBER OF EQUIDISTANT NODES
ax6.set_title('n = {}'.format(n_big))
ax6.plot(f_mesh,f2_fine, label='f(x)')

eq_mesh = np.linspace(a,b,n_big)
f2 = func2(eq_mesh)

mono_f2 = np.array([monomial(x,eq_mesh,f2) for x in f_mesh])
ax6.plot(f_mesh, mono_f2, '-D', label='monomial', markevery=np.searchsorted(f_mesh,eq_mesh))

bary = barycentric_lagrange(eq_mesh,f2,'equal')
lag_f2 = np.array([bary(x) for x in f_mesh])
ax6.plot(f_mesh, lag_f2, '-D', label='barycentric lagrange', markevery=np.searchsorted(f_mesh,eq_mesh))

α = divided_diff(eq_mesh,f2)
newt_f2 = np.array([horner(x,α,eq_mesh) for x in f_mesh])
ax6.plot(f_mesh, newt_f2, '-D', label='newton', markevery=np.searchsorted(f_mesh,eq_mesh))

# f2_prime = np.array([9*(x-2)**8 for x in eq_mesh])
# α = hermite_divided_diff(eq_mesh,f2,f2_prime)
# herm_f2 = np.array([hermite(x,α,eq_mesh) for x in f_mesh])
# ax6.plot(f_mesh, herm_f2, '-D', label='hermite', markevery=np.searchsorted(f_mesh,eq_mesh))

M = compute_M(eq_mesh,f2)
cub_f2 = [cubic_interp(x,eq_mesh,f2,M) for x in f_mesh]
ax6.plot(f_mesh, cub_f2, '-D', label='natural cubic splines', markevery=np.searchsorted(f_mesh,eq_mesh))

ax6.legend()

plt.tight_layout()