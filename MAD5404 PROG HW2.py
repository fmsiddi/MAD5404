import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')


def comp_trapezoidal_error(func, a, b, m, sol, fine):
    H_m = (b-a)/m
    inner_points = np.linspace(a+H_m, b-H_m, m-1)
    inner_points_eval = func(inner_points)
    I_m = H_m/2*(func(a) + func(b) + 2*np.sum(inner_points_eval))
    if fine == True:
        new_points = np.linspace((a + inner_points[0])/2, (b + inner_points[-1])/2, m)
        new_points_eval = func(new_points)
        I_2m = (I_m + H_m*np.sum(new_points_eval))/2
        error = (I_2m - I_m)/3 # denominator is 2**r - 1
    else:
        error = sol - I_m
    return error
    
def comp_midpoint_error(func, a, b, m, sol, fine):
    H_m = (b-a)/m
    prelim_points = np.linspace(a,b,m+1)
    midpoints = np.linspace((prelim_points[0] + prelim_points[1])/2, (prelim_points[-2] + prelim_points[-1])/2, m)
    midpoints_eval = func(midpoints)
    I_m = H_m * np.sum(midpoints_eval)
    Hdiv6 = H_m/6 # precomputed to save on computational time in the loop below
    H5div6 = 5*H_m/6 # precomputed to save on computational time in the loop below
    if fine == True:
        I_3m = (I_m + H_m * np.sum([func(a + i*H_m + Hdiv6) + func(a + i*H_m + H5div6) for i in np.arange(m)]))
        error = (I_3m - I_m)/8 # denominator is 2**r - 1
    else:
        error = sol - I_m
    return error

def comp_simpsons_error(func, a, b, m, sol, fine=False):
    H_m = (b-a)/m
    h_sf = H_m/2
    prelim_points = np.linspace(a,b,m+1)
    c_i = np.array([prelim_points[i] + h_sf for i in range(m)])
    b_i = prelim_points[1:-1]
    I_m = H_m * (func(a) + func(b) + 2*np.sum(func(b_i)) + 4*np.sum(func(c_i)))/6
    return sol - I_m
#%%

# Composite methods, Task 1

def task1(func, tol, method, a, b, sol, fine):
    m = 0
    error = 1
    while error > tol:
        m += 1
        error = abs(method(func, a, b, m, sol, fine))
    H_m = (b-a)/m
    return error, H_m

# func 1:
func = lambda x: np.exp(np.sin(2*x)) * np.cos(2*x)
sol = (np.exp(np.sqrt(3)/2) - 1)/2
a = 0
b = np.pi/3

func1 = np.zeros((3,2))

for method in range(3):
    if method == 0:
        func1[method][0] = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=False)[1]
        func1[method][1] = task1(func, .0001, comp_trapezoidal_error, a, b, sol, fine=False)[1]
    if method == 1:
        func1[method][0] = task1(func, .01, comp_midpoint_error, a, b, sol, fine=False)[1]
        func1[method][1] = task1(func, .0001, comp_midpoint_error, a, b, sol, fine=False)[1]
    if method == 2:
        func1[method][0] = task1(func, .01, comp_simpsons_error, a, b, sol, fine=False)[1]
        func1[method][1] = task1(func, .0001, comp_simpsons_error, a, b, sol, fine=False)[1]
        
# func 2:
func = lambda x: x * np.cos(2*np.pi*x)
sol = -1/(2 * np.pi**2)
a = 0
b = 3.5

func2 = np.zeros((3,2))

for method in range(3):
    if method == 0:
        func2[method][0] = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=False)[1]
        func2[method][1] = task1(func, .0001, comp_trapezoidal_error, a, b, sol, fine=False)[1]
    if method == 1:
        func2[method][0] = task1(func, .01, comp_midpoint_error, a, b, sol, fine=False)[1]
        func2[method][1] = task1(func, .0001, comp_midpoint_error, a, b, sol, fine=False)[1]
    if method == 2:
        func2[method][0] = task1(func, .01, comp_simpsons_error, a, b, sol, fine=False)[1]
        func2[method][1] = task1(func, .0001, comp_simpsons_error, a, b, sol, fine=False)[1]
        
# func 3:
func = lambda x: x + 1/x
sol = (2.5**2 - .1**2)/2 + np.log(2.5/.1)
a = 0.1
b = 2.5

func3 = np.zeros((3,2))

for method in range(3):
    if method == 0:
        func3[method][0] = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=False)[1]
        func3[method][1] = task1(func, .0001, comp_trapezoidal_error, a, b, sol, fine=False)[1]
    if method == 1:
        func3[method][0] = task1(func, .01, comp_midpoint_error, a, b, sol, fine=False)[1]
        func3[method][1] = task1(func, .0001, comp_midpoint_error, a, b, sol, fine=False)[1]
    if method == 2:
        func3[method][0] = task1(func, .01, comp_simpsons_error, a, b, sol, fine=False)[1]
        func3[method][1] = task1(func, .0001, comp_simpsons_error, a, b, sol, fine=False)[1]
