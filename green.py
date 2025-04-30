import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

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
  
    return prefactor * exp_term * energy_term * chi_val

def flux(E,r,t):
    g = green_function(E, r, t)
    return g*c*E**(3)/(4*jnp.pi)

# Example of use

energies = jnp.logspace(1, 5, 100)  # 10 GeV to 100 000 GeV


t = 1e5 * 3.154e7  # 100 000 years in seconds
distances_pc = [10, 100, 1000]  # parsecs
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8,6))

for d_pc, color in zip(distances_pc, colors):
    r_cm = d_pc * 3.086e18
    G_vals = jnp.array([green_function(E, r_cm, t) for E in energies])
    plt.plot(energies, G_vals, label=f"{d_pc} pc", color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("Green function G(E)")
plt.title("G(E) at t = 100 Myr")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))

for d_pc, color in zip(distances_pc, colors):
    r_cm = d_pc * 3.086e18
    flux_vals = jnp.array([flux(E, r_cm, t) for E in energies])
    plt.plot(energies, flux_vals, label=f"{d_pc} pc", color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("E^3 J(E) [GeV cm-2 s-1]")
plt.title("E^3 J(E) at t = 100 Myr")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


