import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def forward_euler(y,h,f,λ,t,F,F_prime):
    return y + h*f(λ,y,t,F,F_prime)

def AB2(y1,y2,h,f,λ,t1,t2,F,F_prime):
    return y1 + .5*h*(3*f(λ,y1,t1,F,F_prime) - f(λ,y2,t2,F,F_prime))

def backward_euler(y,h,λ,t,F,F_prime):
    return (y - h*(λ*F(t) - F_prime(t)))/(1-h*λ)

def AM1(y1,h,f,λ,t,t1,F,F_prime):
    return (y1 - .5*h*(λ*F(t) - F_prime(t) - f(λ,y1,t1,F,F_prime)))/(1-.5*h*λ)

def f(λ,y,t,F,F_prime):
    return λ*(y - F(t)) + F_prime(t)

def y(y_0, F, t, λ):
    return (y_0 - F(0))*np.exp(λ*t) + F(t)

#%%

# 0.2: F(t) = 0 and y(0) = 1, λ = ±1

M = 50
h = 10/M
t_n = np.linspace(0,10,M+1)
t = np.linspace(0,10,10000)
    
F = lambda t: 0
F_prime = lambda t: 0
y_0 = 1
λ = 1

y_t = np.zeros(len(t))
for n in range(len(t)):
    y_t[n] = y(y_0, F, t[n], λ)
    
plt.plot(t,y_t,label='solution')

# forward euler
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = forward_euler(y_n[n-1],h,f,λ,t_n[n-1],F,F_prime)
    
plt.plot(t_n,y_n,label='forward euler')

# AB2
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = AB2(y_n[n-1],y_n[n-2],h,f,λ,t_n[n-1],t_n[n-2],F,F_prime)
    
plt.plot(t_n,y_n,label='AB2')

# backward euler
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = backward_euler(y_n[n-1],h,λ,t_n[n],F,F_prime)
    
plt.plot(t_n,y_n,label='backward euler')

# AM1
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = AM1(y_n[n-1],h,f,λ,t_n[n],t_n[n-1],F,F_prime)
    
plt.plot(t_n,y_n,label='AM1')

plt.legend()

#%%

# 0.3: F(t) = sin(ωt), ω = 10 and .01 and y(0) = 0, λ = -1 and -.01

M = 100
h = 10/M
t_n = np.linspace(0,10,M+1)
t = np.linspace(0,10,10000)
    
ω = 10
# ω = .01
y_0 = 0
λ = -1
# λ = -.01
F = lambda t: np.sin(ω*t)
F_prime = lambda t: ω*np.cos(ω*t)


y_t = np.zeros(len(t))
for n in range(len(t)):
    y_t[n] = y(y_0, F, t[n], λ)
    
plt.plot(t,y_t,label='solution')

# forward euler
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = forward_euler(y_n[n-1],h,f,λ,t_n[n-1],F,F_prime)
    
plt.plot(t_n,y_n,label='forward euler')

# AB2
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = AB2(y_n[n-1],y_n[n-2],h,f,λ,t_n[n-1],t_n[n-2],F,F_prime)
    
plt.plot(t_n,y_n,label='AB2')

# backward euler
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = backward_euler(y_n[n-1],h,λ,t_n[n],F,F_prime)
    
plt.plot(t_n,y_n,label='backward euler')

# AM1
y_n = np.zeros(len(t_n))
y_n[0] = y_0
for n in range(1,len(t_n)):
    y_n[n] = AM1(y_n[n-1],h,f,λ,t_n[n],t_n[n-1],F,F_prime)
    
plt.plot(t_n,y_n,label='AM1')

plt.legend()