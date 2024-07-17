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

# Main Functions
def LBdx(x, y):
    return np.arccos(np.cos(x) + (np.sin(x / 2)**2) * (1 + np.cos(y)))



def LBdx2(x, y):# If k≥pi/2, we can assume α=0, so d(x,x')=alpha'. 
    bound = np.where(y >= np.pi / 2, x, np.arcsin(np.sin(x)*np.sin(y)))
    return bound

def LBdx3(x,y):
    return np.amax([LBdx(x, y), LBdx2(x, y)],axis=0)




def INEQ(x, y):
    return G(y,x-pi/2+1)-LBdx3(x,y)-zeta



# Generate grid points
x_vals = np.arange(pi/2-1, pi/2, d)
y_vals = np.arange(0.3, pi, d)






X, Y = np.meshgrid(x_vals, y_vals)

Z = INEQ(X, Y)


colorscale = [
    [0.0, 'blue'],   # Color for minimum value
    [0.53, 'blue'],   # Color for values around 0
    [0.53, 'red'],    # Transition color at 0
    [1.0, 'red']     # Color for maximum value
]


# Plot the result using Plotly
surface = go.Surface(z=Z, x=X, y=Y, colorscale=colorscale)
layout = go.Layout(
    title='INEQ(x, y)',
    scene=dict(
        xaxis_title='x',
        yaxis_title='y',
        zaxis_title='INEQ(x, y)'
    )
)
fig = go.Figure(data=[surface], layout=layout)

# Render the plot
pio.show(fig)


