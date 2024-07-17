import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import time

# Constants
n=3
zeta = np.arccos(-1/(n+1))
pi = np.pi
d = 0.0001

# Helper Functions
def C(k):
    return (-np.sqrt(1 - 2 * np.cos(zeta - k) + ((n+1)**2) * np.cos(zeta - k)**2)) / np.sqrt((n+1)**2 - 1)


def G(k, beta):
    term1 = (-1/(n+1)) * np.cos((pi - zeta) * beta)
    term2 = (np.sqrt((n+1)**2-1) / (n+1)) * np.sin((pi - zeta) * beta) * ((((n+1)**2) * np.cos(pi - zeta + k) + 1) / ((n+1)**2-1))
    return np.arccos(term1 + term2)

# Main Functions
def LBdx(x, y):
    return np.arccos(np.cos(x) + (np.sin(x / 2)**2) * (1 + np.cos(y)))

def k_2(x, y):
    return y + (pi - zeta) * (pi/2 - x/2)

def UBdF2(x, y):
    return (pi - zeta) * (x/2 - pi/2 + 1) + G(y, x/2 - pi/2 + 1)

def UBdF1(x, y): #This outputs UBdF1 if the condition is not met, pi if it is. Will complain about not being able to compute arccosines but that only happens when the condition below is met (so the arccos does not matter/does not make geometric sense).
    k2 = k_2(x, y)
    C_y = C(y)
    condition = (pi - zeta + np.arccos(C_y) + k2 >= 2 * pi)
    argument = (-(n+1) / np.sqrt((n+1)**2-1)) * np.sqrt(C_y**2 + np.cos(k2)**2 - (2/(n+1)) * C_y * np.cos(k2))
    result = np.where(
        condition,
        pi,
        np.arccos(argument)
    )
    return result

def UBdF3(x,y):
    return y + (pi - zeta) * (pi - x)

#def UBF2(x,y):
#    return zeta + (pi - zeta) * (x - pi + 2)

#def UBF3(x,y):
#    return np.arccos(C(y)) + (pi - zeta)*(x/2 - pi/2 + 1)


def UBdF(x, y):#Find minimum of all upper bounds.
    return np.amin([UBdF1(x, y), UBdF2(x, y), UBdF3(x, y)],axis=0)

def INEQ(x, y):
    return UBdF(x, y) - LBdx(x, y)-zeta

# Generate grid points
x_vals = np.arange(pi-2, 2, d)
y_vals = np.arange(0.7, 1.4, d)













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




