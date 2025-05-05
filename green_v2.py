import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import vmap, jit, lax
import matplotlib.colors as mcolors



jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)

# Constants
D0 = 1e28
delta = 0.6
zmax = 1e3  # pc
zmax_cm = zmax * 3.086e18
zcr = 4 * zmax_cm / jnp.pi
b0 = 1e-16
Q0 = 7.8e49
gamma = 2.2
E_fixed = 1e3  # GeV
t_now = 1e5* 3.154e7  # 1 Myr in seconds
N_sources = 10000
c = 3e10
Ecut = 1e4

# Grid
grid_size = 200
extent_pc = 15000
extent_cm = extent_pc * 3.086e18

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

def green_core(E, r, t):
    Es = E_s(E, t)
    l2_val = l2(E, t)
    l2_hat = l2_val / zcr**2
    prefactor = Q0 / (jnp.pi * l2_val * zcr)
    exp_term = jnp.exp(-r**2 / l2_val)
    energy_term = Es**(-gamma) * (1 - b0 * E * t)**(-2)
    chi_val = chi_function(l2_hat)
    return prefactor * exp_term * energy_term * chi_val


def green_function(E, r, t):
    return lax.cond(
        b0 * E * t < 1.0,
        lambda _: green_core(E, r, t),
        lambda _: 0.0,
        operand=None
    )


def sample_disk(key, N, R_max):
    key_r, key_theta = jax.random.split(key)
    r = R_max * jnp.sqrt(jax.random.uniform(key_r, (N,)))
    theta = 2 * jnp.pi * jax.random.uniform(key_theta, (N,))
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return x, y

# Sources sampling
subkey1, subkey2 = jax.random.split(key)
x_i, y_i = sample_disk(subkey1, N_sources, extent_cm)
t_i = t_now * jax.random.uniform(subkey2, (N_sources,))

# Green function for each source
def green_i(x, y, t0, xi, yi, ti):
    r = jnp.sqrt((x - xi)**2 + (y - yi)**2)
    dt = t0 - ti
    return jnp.where(dt > 0, green_function(E_fixed, r, dt), 0.0)

v_green = vmap(green_i, in_axes=(None, None, None, 0, 0, 0))

# Grid
x_vals = jnp.linspace(-extent_cm, extent_cm, grid_size)
y_vals = jnp.linspace(-extent_cm, extent_cm, grid_size)
X, Y = jnp.meshgrid(x_vals, y_vals)

# Density
@jit
def compute_ncr(xg, yg):
    return v_green(xg, yg, t_now, x_i, y_i, t_i).sum()

ncr_grid = vmap(
    lambda x_row, y_row: vmap(compute_ncr)(x_row, y_row)
)(X, Y)

# Mask outside the disk
R_grid = jnp.sqrt(X**2 + Y**2)
mask = R_grid > extent_cm +100*3.086e18  
ncr_masked = jnp.where(mask, jnp.nan, ncr_grid)  


plt.figure(figsize=(8, 6))
plt.imshow(ncr_masked,
           extent=[-extent_pc, extent_pc, -extent_pc, extent_pc],
           origin='lower',
           cmap='inferno',
           norm=mcolors.LogNorm(vmin=1e-22, vmax=2.8813092161518162e-18)) 
plt.colorbar(label='n_CR [GeV$^{-1}$ cm$^{-3}$]')
plt.xlabel('x [pc]',fontsize=13)
plt.ylabel('y [pc]',fontsize=13)
plt.title(f'CR Density at z=0, E={E_fixed} GeV, t={t_now / 3.154e13:.2f} Myr')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(x_i / 3.086e18, y_i / 3.086e18, color='black', s=0.1, label='Sources position')
plt.xlabel('x [pc]',fontsize = 13)
plt.ylabel('y [pc]',fontsize = 13)
plt.title(f' Random sources position in a radius {extent_pc /1000:.0f} kpc ')


print(ncr_grid.min(), ncr_grid.max())


#################################################################################
#################################################################################

# AVEC ECUT

def green_core_Ecut(E, r, t):
    Es = E_s(E, t)
    l2_val = l2(E, t)
    l2_hat = l2_val / zcr**2
    prefactor = Q0 / (jnp.pi * l2_val * zcr)
    exp_term = jnp.exp(-r**2 / l2_val)
    energy_term = Es**(-gamma) * (1 - b0 * E * t)**(-2)
    chi_val = chi_function(l2_hat)
    return prefactor * exp_term * energy_term * chi_val*jnp.exp(-Es/Ecut)


def green_function_Ecut(E, r, t):
    return lax.cond(
        b0 * E * t < 1.0,
        lambda _: green_core_Ecut(E, r, t),
        lambda _: 0.0,
        operand=None
    )

def green_i_Ecut(x, y, t0, xi, yi, ti):
    r = jnp.sqrt((x - xi)**2 + (y - yi)**2)
    dt = t0 - ti
    return jnp.where(dt > 0, c*green_function_Ecut(E_fixed, r, dt)/(4*jnp.pi), 0.0)
v_green2 = vmap(green_i_Ecut, in_axes=(None, None, None, 0, 0, 0))

@jit
def compute_ncr_Ecut(xg, yg):
    return v_green2(xg, yg, t_now, x_i, y_i, t_i).sum()

ncr_grid2 = vmap(
    lambda x_row, y_row: vmap(compute_ncr)(x_row, y_row)
)(X, Y)

ncr_masked2 = jnp.where(mask, jnp.nan, ncr_grid2) 
"""
plt.figure(figsize=(8, 6))
plt.imshow(ncr_masked2,
           extent=[-extent_pc, extent_pc, -extent_pc, extent_pc],
           origin='lower',
           cmap='inferno',
           norm='log',
           vmin=1e-10, vmax= ncr_grid.max())
plt.colorbar(label='n_CR [GeV$^{-1}$ cm$^{-3}$]')
plt.xlabel('x [pc]',fontsize=13)
plt.ylabel('y [pc]',fontsize=13)
plt.title(f'CR Density at z=0, E={E_fixed} GeV Ecut = {Ecut} GeV, t={t_now / 3.154e7:.1f} yr')
plt.tight_layout()
plt.show()

"""

"""

tnow_list_myr = [0.1,1,10]  # in Myr
tnow_list = [tnow_myr * 3.154e13 for tnow_myr in tnow_list_myr]  # in seconds

t_i_base = jax.random.uniform(subkey2, (N_sources,)) 

for tnow, tnow_myr in zip(tnow_list, tnow_list_myr):
    t_i = tnow * t_i_base  

    @jit
    def compute_ncr(xg, yg):
        return v_green(xg, yg, tnow, x_i, y_i, t_i).sum()

    ncr_grid = vmap(
        lambda x_row, y_row: vmap(compute_ncr)(x_row, y_row)
    )(X, Y)

    # Mask outside the disk
    R_grid = jnp.sqrt(X**2 + Y**2)
    mask = R_grid > extent_cm + 100 * 3.086e18  
    ncr_masked = jnp.where(mask, jnp.nan, ncr_grid)  

    # Affichage
    plt.figure(figsize=(8, 6))
    plt.imshow(ncr_masked,
           extent=[-extent_pc, extent_pc, -extent_pc, extent_pc],
           origin='lower',
           cmap='inferno',
           norm=mcolors.LogNorm(vmin=1e-10, vmax=ncr_grid.max())) 
    plt.colorbar(label='n_CR [GeV$^{-1}$ cm$^{-3}$]')
    plt.xlabel('x [pc]',fontsize=13)
    plt.ylabel('y [pc]',fontsize=13)
    plt.title(f'CR Density at z=0, E={E_fixed} GeV, t_now={tnow_myr} Myr')
    plt.tight_layout()
    plt.show()

"""

