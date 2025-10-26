import numpy as np

eta3 = np.arccos(-1/4)

C  = lambda κ: -np.sqrt(1 - 2*np.cos(eta3 - κ) + 16*np.cos(eta3 - κ)**2) / np.sqrt(15)
L  = lambda κ: np.arccos(np.cos(2)*(np.cos(κ)-1)/2 + (1+np.cos(κ))/2)
U3 = lambda κ: np.arccos((-4/np.sqrt(15)) * np.sqrt(C(κ)**2 + np.cos(np.pi - eta3 + κ)**2 - 0.5*C(κ)*np.cos(np.pi - eta3 + κ)))

step = 1e-7
κ, fmax = 0.3, -np.inf

while κ <= 0.7:
    f = U3(κ) - L(κ) - eta3
    if f > fmax:
        fmax = f
    κ += step

print("Maximum value of L - eta3 - U3:", fmax)