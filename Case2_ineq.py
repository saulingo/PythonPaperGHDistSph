import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import time

# Constants
n=3
zeta = np.arccos(-1/(n+1))
pi = np.pi
d = 0.001

def G(k, beta):
    term1 = (-1/(n+1)) * np.cos((pi - zeta) * beta)
    term2 = (np.sqrt((n+1)**2-1) / (n+1)) * np.sin((pi - zeta) * beta) * ((((n+1)**2) * np.cos(pi - zeta + k) + 1) / ((n+1)**2-1))
    return np.arccos(term1 + term2)

# Main Functions. The variable x now represents α, while y represents k.
def LBdx1(x, y):
    return np.arccos(np.cos(x) + (np.sin(x / 2)**2) * (1 + np.cos(y)))



def LBdx2(x, y):# If k≥pi/2, we can assume α=0, so d(x,x')=alpha'. 
    bound = np.where(y >= np.pi / 2, x, np.arcsin(np.sin(x)*np.sin(y)))
    return bound

def LBdx3(x,y):
    return np.amax([LBdx1(x, y), LBdx2(x, y)],axis=0)




def INEQ(x, y):
    return G(y,x-pi/2+1)-LBdx3(x,y)-zeta



# Generate grid points
x_vals = np.arange(pi/2-1, pi/2, d)
y_vals = np.arange(0.3, pi, d)










# Function to find the maximum value of INEQ(x, y) in chunks
def find_max_ineq(x_vals, y_vals, chunk_size):
    max_ineq = -np.inf
    for x_start in range(0, len(x_vals), chunk_size):
        x_end = min(x_start + chunk_size, len(x_vals))
        for y_start in range(0, len(y_vals), chunk_size):
            y_end = min(y_start + chunk_size, len(y_vals))
            
            x_chunk = x_vals[x_start:x_end]
            y_chunk = y_vals[y_start:y_end]
            
            X, Y = np.meshgrid(x_chunk, y_chunk)
            INEQ_values = INEQ(X, Y)
            max_ineq_chunk = np.max(INEQ_values)
            max_ineq = max(max_ineq, max_ineq_chunk)
    
    return max_ineq

# Adjust chunk_size based on your memory constraints
chunk_size = 1000



time0=time.perf_counter()
# Find the maximum value of INEQ
max_ineq_value = find_max_ineq(x_vals, y_vals, chunk_size)
print("Maximum value of INEQ:", max_ineq_value)
print("Tiempo:",time.perf_counter()-time0)








