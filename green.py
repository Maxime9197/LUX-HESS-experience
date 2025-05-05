import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap

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
Ecut = 1000

def E_s(E, t):
    return E / (1 - b0 * E * t)

def l2(E, t):
    Es = E_s(E, t)
    term1 = E**(delta - 1)
    term2 = Es**(delta - 1)
    return (4 * D0 / (b0 * (1 - delta))) * (term1 - term2)

def l2_d(E,t):
    return 4*D0*E**(delta)*t

def theta3(z, q, N=100):
    n = jnp.arange(1, N + 1)
    series = jnp.sum(q ** (n**2) * jnp.cos(2 * n * z))
    return 1 + 2 * series

def chi_function(l2_hat, N=100):
    q = jnp.exp(-l2_hat)
    theta3_0 = theta3(0.0, q, N)
    theta3_pi2 = theta3(jnp.pi / 2, q, N)
    return (theta3_0 - theta3_pi2) / jnp.pi


# Green function with loss of energy
def green_function(E, r, t, verbose=True):
   
    Es = E_s(E, t)
    l2_val = l2(E, t)
    l2_hat = l2_val / zcr**2

    prefactor = Q0 / (jnp.pi * l2_val * zcr)
    exp_term = jnp.exp(-r**2 / l2_val)
    energy_term = Es**(-gamma) * (1 - b0 * E * t)**(-2)
    chi_val = chi_function(l2_hat)
    return prefactor * exp_term * energy_term * chi_val

# Green function without loss of energy
def green_function_d(E, r, t, verbose=True):
      
      l2_val = l2_d(E, t)
      l2_hat = l2_val / zcr**2

      prefactor = Q0 / (jnp.pi * l2_val * zcr)
      exp_term = jnp.exp(-r**2 / l2_val)
      energy_term = E**(-gamma)
      chi_val = chi_function(l2_hat)  
      return prefactor * exp_term * energy_term * chi_val
      

# Example of use
energies = jnp.logspace(1, 3, 100)  # 10 to ... GeV

t = 1e6 * 3.154e7  # 100 000 years in seconds
r = 100 * 3.086e18  # 100 parsecs in cm

# Plot G(E) with loss of energy
g_values = jnp.array([green_function(E, r, t, verbose=True) for E in energies])
plt.figure(figsize=(8, 6))
plt.plot(energies, g_values, label="G(E)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("Green function G(E)")
plt.title("G(E) at t = 100 kyr and r = 100 pc")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# Plot G(E) without loss of energy
g_values = jnp.array([green_function_d(E, r, t, verbose=True) for E in energies])
plt.figure(figsize=(8, 6))
plt.plot(energies, g_values, label="G(E)")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("Green function G(E)")
plt.title("G(E) without energy loss at t = 100 kyr and r = 100 pc")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()


# Parameters for Heatmap
E_fixed_1 = 1e3  # GeV
t_fixed_1 = 1e5 * 3.154e7  # seconds
z_fixed_1 = 0  # cm
grid_size_1 = 200
extent_pc_1 =10000  # +/- in parsecs
extent_cm_1 = extent_pc_1 * 3.086e18  # convert to cm

# 2D grid
x_vals_1 = jnp.linspace(-extent_cm_1, extent_cm_1, grid_size_1)
y_vals_1 = jnp.linspace(-extent_cm_1, extent_cm_1, grid_size_1)
X_1, Y_1 = jnp.meshgrid(x_vals_1, y_vals_1)
R_1 = jnp.sqrt(X_1**2 + Y_1**2 + z_fixed_1**2)

# Evaluate Green function for heatmap
G_vals = jnp.vectorize(lambda r: green_function(E_fixed_1, r, t_fixed_1, verbose=False))(R_1)

# Plotting Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(G_vals,
           extent=[-extent_pc_1, extent_pc_1, -extent_pc_1, extent_pc_1],
           origin='lower',
           norm='log',
           cmap='inferno')
plt.colorbar(label='G(E, r, t)')
plt.xlabel('x [pc]')
plt.ylabel('y [pc]')
plt.title(f'Heatmap of G(E={E_fixed_1:.0f} GeV, z=0, t= 10 kyr)')
plt.tight_layout()
plt.show()


E_fixed_2 = 1e3  # GeV
t_fixed_2 = 1e5 * 3.154e7  # seconds
z_fixed_2 = 0  # cm
grid_size_2 = 200
extent_pc_2 =10000  # +/- in parsecs
extent_cm_2 = extent_pc_2 * 3.086e18  # convert to cm

# 2D grid
x_vals_2 = jnp.linspace(-extent_cm_2, extent_cm_2, grid_size_2)
y_vals_2 = jnp.linspace(-extent_cm_2, extent_cm_2, grid_size_2)
X_2, Y_2 = jnp.meshgrid(x_vals_2, y_vals_2)
R_2 = jnp.sqrt(X_2**2 + Y_2**2 + z_fixed_2**2)

G_vals2 = jnp.vectorize(lambda r: green_function_d(E_fixed_2, r, t_fixed_2, verbose=False))(R_2)

# Plotting Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(G_vals2,
           extent=[-extent_pc_2, extent_pc_2, -extent_pc_2, extent_pc_2],
           origin='lower',
           norm='log',
           cmap='inferno')
plt.colorbar(label='G(E, r, t)')
plt.xlabel('x [pc]')
plt.ylabel('y [pc]')
plt.title(f'Heatmap of G(E={E_fixed_2:.0f} GeV, z=0, t= 10 kyr) without energy loss')
plt.tight_layout()
plt.show()

# Comparison

r_cm = 100 * 3.086e18  # 100 pc en cm
plt.figure(figsize=(8,6))


G_val = jnp.array([green_function(E, r_cm, t) for E in energies])    
G_val2 = jnp.array([green_function_d(E, r_cm, t) for E in energies])  


plt.plot(energies, G_val, label="With loss of energy")
plt.plot(energies, G_val2, label="Without loss of energy")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("Green Function G(E)")
plt.title("Comparison of G(E) at t = 1 Myr and r = 100 pc")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()






"""

# G(E) at different distances

distances_pc = [10, 100, 1000]  # parsecs
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8,6))

for d_pc, color in zip(distances_pc, colors):
    r_cm = d_pc * 3.086e18
    G_val = jnp.array([green_function(E, r_cm, t) for E in energies])
    plt.plot(energies, G_val, label=f"{d_pc} pc", color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("Green function G(E)")
plt.title("G(E) at t = 10 Myr")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
for d_pc, color in zip(distances_pc, colors):
    r_cm = d_pc * 3.086e18
    G_val2 = jnp.array([green_function_d(E, r_cm, t) for E in energies])
    plt.plot(energies, G_val2, label=f"{d_pc} pc", color=color)

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Energy [GeV]")
plt.ylabel("Green function G(E) ")
plt.title("G(E) at t = 10 Myr without energy loss")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

"""


