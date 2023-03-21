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
        if m == 1:
            new_points = np.array([(a+b)/2])
        else:
            new_points = np.linspace((a + inner_points[0])/2, (b + inner_points[-1])/2, m)
        new_points_eval = func(new_points)
        I_2m = (I_m + H_m*np.sum(new_points_eval))/2
        approx_error = (I_2m - I_m)/3 # denominator is 2**r - 1
        true_error = sol - I_2m
        return approx_error, true_error
    else:
        error = sol - I_m
        return error

    
def comp_midpoint_error(func, a, b, m, sol, fine):
    H_m = (b-a)/m
    prelim_points = np.linspace(a,b,m+1)
    midpoints = np.linspace((prelim_points[0] + prelim_points[1])/2, (prelim_points[-2] + prelim_points[-1])/2, m)
    midpoints_eval = func(midpoints)
    I_m = H_m * np.sum(midpoints_eval)
    if fine == True:
        new_points1 = np.array([a + i*H_m + H_m/6 for i in range(m)])
        new_points2 = np.array([a + i*H_m + 5*H_m/6 for i in range(m)])
        I_3m = (I_m + H_m * np.sum([func(new_points1[i]) + func(new_points2[i]) for i in range(m)]))/3
        approx_error = (I_3m - I_m)/8 # denominator is 3**r - 1
        true_error = sol - I_3m
        return approx_error, true_error
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
    error = 1
    if fine == False:
        m = 0
        while error > tol:
            m += 1
            error = abs(method(func, a, b, m, sol, fine))
        H_m = (b-a)/m
        return error, H_m, m
    else:
        error_comparison = np.zeros((10,2))
        # if method == comp_trapezoidal_error:
        #     m_array = np.array([2**k for k in range(1,11)])
        # elif method == comp_midpoint_error:
        #     m_array = np.array([3**k for k in range(1,11)])
        m_array = np.arange(1,11)
        for i in range(len(m_array)):
            errors = method(func, a, b, m_array[i], sol, fine)
            error_comparison[i][0] = abs(errors[0]) # approximate error
            error_comparison[i][1] = abs(errors[1]) # true error
        return error_comparison
#%%
# Part 1 of Task 1

# func 1:
func = lambda x: np.exp(np.sin(2*x)) * np.cos(2*x)
sol = (np.exp(np.sqrt(3)/2) - 1)/2
a = 0
b = np.pi/3

func1 = np.zeros((3,2,3))

for method in range(3):
    if method == 0:
        quad = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=False)
        func1[method][0][0] = quad[0]
        func1[method][0][1] = quad[1]
        func1[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_trapezoidal_error, a, b, sol, fine=False)
        func1[method][1][0] = quad[0]
        func1[method][1][1] = quad[1]
        func1[method][1][2] = quad[2]
    if method == 1:
        quad = task1(func, .01, comp_midpoint_error, a, b, sol, fine=False)
        func1[method][0][0] = quad[0]
        func1[method][0][1] = quad[1]
        func1[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_midpoint_error, a, b, sol, fine=False)
        func1[method][1][0] = quad[0]
        func1[method][1][1] = quad[1]
        func1[method][1][2] = quad[2]
    if method == 2:
        quad = task1(func, .01, comp_simpsons_error, a, b, sol, fine=False)
        func1[method][0][0] = quad[0]
        func1[method][0][1] = quad[1]
        func1[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_simpsons_error, a, b, sol, fine=False)
        func1[method][1][0] = quad[0]
        func1[method][1][1] = quad[1]
        func1[method][1][2] = quad[2]
        
# func 2:
func = lambda x: x * np.cos(2*np.pi*x)
sol = -1/(2 * np.pi**2)
a = 0
b = 3.5

func2 = np.zeros((3,2,3))

for method in range(3):
    if method == 0:
        quad = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=False)
        func2[method][0][0] = quad[0]
        func2[method][0][1] = quad[1]
        func2[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_trapezoidal_error, a, b, sol, fine=False)
        func2[method][1][0] = quad[0]
        func2[method][1][1] = quad[1]
        func2[method][1][2] = quad[2]
    if method == 1:
        quad = task1(func, .01, comp_midpoint_error, a, b, sol, fine=False)
        func2[method][0][0] = quad[0]
        func2[method][0][1] = quad[1]
        func2[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_midpoint_error, a, b, sol, fine=False)
        func2[method][1][0] = quad[0]
        func2[method][1][1] = quad[1]
        func2[method][1][2] = quad[2]
    if method == 2:
        quad = task1(func, .01, comp_simpsons_error, a, b, sol, fine=False)
        func2[method][0][0] = quad[0]
        func2[method][0][1] = quad[1]
        func2[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_simpsons_error, a, b, sol, fine=False)
        func2[method][1][0] = quad[0]
        func2[method][1][1] = quad[1]
        func2[method][1][2] = quad[2]
        
# func 3:
func = lambda x: x + 1/x
sol = (2.5**2 - .1**2)/2 + np.log(2.5/.1)
a = 0.1
b = 2.5

func3 = np.zeros((3,2,3))

for method in range(3):
    if method == 0:
        quad = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=False)
        func3[method][0][0] = quad[0]
        func3[method][0][1] = quad[1]
        func3[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_trapezoidal_error, a, b, sol, fine=False)
        func3[method][1][0] = quad[0]
        func3[method][1][1] = quad[1]
        func3[method][1][2] = quad[2]
    if method == 1:
        quad = task1(func, .01, comp_midpoint_error, a, b, sol, fine=False)
        func3[method][0][0] = quad[0]
        func3[method][0][1] = quad[1]
        func3[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_midpoint_error, a, b, sol, fine=False)
        func3[method][1][0] = quad[0]
        func3[method][1][1] = quad[1]
        func3[method][1][2] = quad[2]
    if method == 2:
        quad = task1(func, .01, comp_simpsons_error, a, b, sol, fine=False)
        func3[method][0][0] = quad[0]
        func3[method][0][1] = quad[1]
        func3[method][0][2] = quad[2]
        
        quad = task1(func, .0001, comp_simpsons_error, a, b, sol, fine=False)
        func3[method][1][0] = quad[0]
        func3[method][1][1] = quad[1]
        func3[method][1][2] = quad[2]
#%%

# Part 2 of Task 1

# func 1:
func = lambda x: np.exp(np.sin(2*x)) * np.cos(2*x)
sol = (np.exp(np.sqrt(3)/2) - 1)/2
a = 0
b = np.pi/3

func1_fine = np.zeros((2,10,2))

func1_fine[0] = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=True)
func1_fine[1] = task1(func, .01, comp_midpoint_error, a, b, sol, fine=True)
    
        
# func 2:
func = lambda x: x * np.cos(2*np.pi*x)
sol = -1/(2 * np.pi**2)
a = 0
b = 3.5

func2_fine = np.zeros((2,10,2))

func2_fine[0] = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=True)
func2_fine[1] = task1(func, .01, comp_midpoint_error, a, b, sol, fine=True)
        

# func 3:
func = lambda x: x + 1/x
sol = (2.5**2 - .1**2)/2 + np.log(2.5/.1)
a = 0.1
b = 2.5

func3_fine = np.zeros((2,10,2))

func3_fine[0] = task1(func, .01, comp_trapezoidal_error, a, b, sol, fine=True)
func3_fine[1] = task1(func, .01, comp_midpoint_error, a, b, sol, fine=True)
        
#%%

func = lambda x: x**2 
sol = (12**3)/3

a = 0
b = 12
m = 2

H_m = (b-a)/m
prelim_points = np.linspace(a,b,m+1)
midpoints = np.linspace((prelim_points[0] + prelim_points[1])/2, (prelim_points[-2] + prelim_points[-1])/2, m)
midpoints_eval = func(midpoints)
I_m = H_m * np.sum(midpoints_eval)

new_points1 = np.array([a + i*H_m + H_m/6 for i in range(m)])
new_points2 = np.array([a + i*H_m + 5*H_m/6 for i in range(m)])
I_3m = (I_m + H_m * np.sum([func(new_points1[i]) + func(new_points2[i]) for i in range(m)]))/3
approx_error = (I_3m - I_m)/8 # denominator is 3**r - 1
true_error = sol - I_3m

