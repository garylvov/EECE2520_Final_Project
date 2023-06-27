# from sympy import  fourier_transform, exp
# from sympy.abc import t
# import sympy as sp
# # Define the symbol
# k = sp.symbols('ω')
# t = sp.symbols('t')
# # Define the signal
# signal = sp.cos(t)

# # Calculate the Fourier transform without evaluation
# transform = fourier_transform(signal, t, k)

# # Print the symbolic expression for the Fourier transform
# print(transform)

import sympy as sp

# Define the symbols
t, f = sp.symbols('t f')

# Define the pulse function
def pulse(t):
    return sp.Piecewise((1, sp.And(t >= -0.5, t <= 0.5)), (0, True))

# Compute the Fourier transform
fourier_transform = sp.fourier_transform(pulse(t), t, f)

# Print the result
print(fourier_transform)
