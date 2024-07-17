import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import time

# Constants
n=3
zeta = np.arccos(-1/(n+1))
pi = np.pi
d = 0.001


# Main Functions
def LBdx(x, y):
    return np.arccos(np.cos(x) + (np.sin(x / 2)**2) * (1 + np.cos(y)))

def UBdF6(x,y):
    return y+(pi-zeta)*(pi-x)

def UBdF7(x,y):
    return zeta+(pi-zeta)*(x-pi+2)




def UBdF(x, y):#Find minimum of all upper bounds.
    return np.amin([UBdF6(x, y), UBdF7(x, y)],axis=0)

def INEQ(x, y):
    return UBdF(x, y) - LBdx(x, y)-zeta

# Generate grid points
x_vals = np.arange(pi-2, 2, d)
y_vals = np.arange(1.4, pi, d)










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




