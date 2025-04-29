import jax
import jax.numpy as jnp

# Constants
D0 = 1e28         # cm^2 / s
delta = 0.6
zmax = 3e3        # parsec
zmax_cm = zmax * 3.086e18  # cm
zcr = 4 * zmax_cm / jnp.pi
b0 = 1e-16        # GeV^{-1} s^{-1}
Q0 = 7.8e49       # GeV^{-1}
gamma = 2.2
c = 3e13


def E_s(E, t):
    return E / (1 - b0 * E * t)

def l2(E, t):
    Es = E_s(E, t)
    term1 = E**(delta - 1)
    term2 = Es**(delta - 1)
    return (4 * D0 / (b0 * (1 - delta))) * (term1 - term2)



def theta3(z, q, N=100):
    n = jnp.arange(1, N + 1)
    series = jnp.sum(q ** (n**2) * jnp.cos(2 * n * z))
    return 1 + 2 * series

def chi_function(l2_hat, N=100):
    q = jnp.exp(-l2_hat)
    theta3_0 = theta3(0.0, q, N)
    theta3_pi2 = theta3(jnp.pi / 2, q, N)
    return (theta3_0 - theta3_pi2) / jnp.pi


def green_function(E, r, t):
    Es = E_s(E, t)
    l2_val = l2(E, t)
    l2_hat = l2_val / zcr**2
    
    prefactor = Q0 / (jnp.pi * l2_val*zcr)
    exp_term = jnp.exp(-r**2 / l2_val)
    energy_term = Es**(-gamma)*(1 - b0 * E * t)**(-2)
    chi_val = chi_function(l2_hat)
    print(f"E_s = {Es}")
    print(f"l2_val = {l2_val}")
    
   
    
    print(f"Prefactor = {prefactor}")
    
    
   
    print(f"Energy_term = {energy_term}")
    
    
    print(f"Chi_val = {chi_val}")
    return prefactor * exp_term * energy_term * chi_val

# Example of use
E = 100.0  # GeV
r = 100.0 * 3.086e18  # 100 parsecs in cm
t = 1e5 * 3.154e7     # 100 000 ans in secondes

flux = E**(3)*green_function(E, r, t)*c/(4*jnp.pi)
print(f"Flux: {flux:.10e}")



