import numpy as np
from scipy.optimize import fsolve

# Define the system of equations based on the step response formula
def equations(p):
    tau1, tau2 = p
    # Equation for the 28.3% rise point
    eq1 = 1 - (tau1 * np.exp(-526 / tau1) - tau2 * np.exp(-526 / tau2)) / (tau1 - tau2) - 0.283
    # Equation for the 63.2% rise point
    eq2 = 1 - (tau1 * np.exp(-1138 / tau1) - tau2 * np.exp(-1138 / tau2)) / (tau1 - tau2) - 0.632
    return (eq1, eq2)

# Provide an initial guess for the solver
initial_guess = [1000, 500]

# Use fsolve to find the values of tau1 and tau2 that satisfy the equations
solution = fsolve(equations, initial_guess)

tau1, tau2 = solution

print(f"Estimated tau1: {tau1:.2f} seconds")
print(f"Estimated tau2: {tau2:.2f} seconds")
