import numpy as np
from scipy.special import lambertw
import timeit

# Gaussian beam parameters
P0 = 1e-3  # Power (W)
wavelength = 1.064e-6  # Wavelength (m)
rho0_true = 100e-6  # Beam waist (m)
z_pos = 2 * (np.pi * rho0_true**2 / wavelength)  # Position along z-axis (m)

# Calculate true beam radius at z (for reference only)
def calculate_true_rho(z, rho0, wavelength):
    zR = np.pi * rho0**2 / wavelength  # Rayleigh range
    rho_z = rho0 * np.sqrt(1 + (z / zR)**2)  # Beam radius at z
    return rho_z

true_rho_z = calculate_true_rho(z_pos, rho0_true, wavelength)

# Function to calculate Gaussian beam intensity
def gaussian_intensity(z, r, P0, rho_z_val):
    """Gaussian intensity function at radial distance r."""
    I_zr = (2 * P0 / (np.pi * rho_z_val**2)) * np.exp(-2 * r**2 / rho_z_val**2)
    return I_zr

# Compute intensity at r = rho(z) / sqrt(2) (simulated measurement)
r_radial = true_rho_z / np.sqrt(2)  # Radial distance r = rho(z) / sqrt(2)
intensity_reference = gaussian_intensity(z_pos, r_radial, P0, true_rho_z)

# Initial guess for Newton-Raphson based on intensity measurement
def initial_guess_from_intensity(I, P0):
    """Estimate initial rho from intensity I, ignoring exponential decay."""
    rho_approx = np.sqrt(2 * P0 / (np.pi * I))  # Simple on-axis approximation
    return rho_approx

rho_initial = initial_guess_from_intensity(intensity_reference, P0)

# Lambert W Function Method
def lambert_w_method():
    return np.sqrt(-2 * r_radial**2 / lambertw(-np.pi * r_radial**2 * intensity_reference / P0, k=0).real)

# Newton-Raphson Method with Optimized Implementation
def derivative_intensity_equation_for_newton_raphson(rho_val, r, P0):
    """Corrected derivative f'(rho) of the intensity equation with respect to rho."""
    rho2 = rho_val**2
    r2 = r**2
    exp_term = np.exp(-2 * r2 / rho2)
    term1 = (4 * P0 / (np.pi * rho_val**3)) * exp_term
    term2 = -(2 * P0 / (np.pi * rho_val**2)) * exp_term * (4 * r2 / rho_val**3)
    return term1 + term2

def newton_raphson_method():
    rho_numerical_nr = rho_initial  # Use intensity-based initial guess
    tolerance = 1e-9  # Tight tolerance for precision
    max_iterations = 100  # Sufficient iterations for convergence
    iteration_nr = 0

    while iteration_nr < max_iterations:
        f_val = intensity_reference - gaussian_intensity(z_pos, r_radial, P0, rho_numerical_nr)
        f_deriv_val = derivative_intensity_equation_for_newton_raphson(rho_numerical_nr, r_radial, P0)
        
        # Check for numerical issues
        if np.isnan(f_deriv_val) or np.isinf(f_deriv_val) or np.abs(f_deriv_val) < 1e-12:
            break

        # Standard Newton-Raphson update (no damping)
        rho_numerical_next = rho_numerical_nr - (f_val / f_deriv_val)

        # Safeguard against negative or tiny values
        if rho_numerical_next < 1e-12:
            rho_numerical_next = 1e-12

        # Check convergence
        if np.abs(rho_numerical_next - rho_numerical_nr) < tolerance:
            break

        rho_numerical_nr = rho_numerical_next
        iteration_nr += 1

    return rho_numerical_nr

# Profile Lambert W Function
time_lw = timeit.timeit(lambert_w_method, number=1000)  # Run 1000 times for accuracy

# Profile Newton-Raphson Method
time_nr = timeit.timeit(newton_raphson_method, number=1000)  # Run 1000 times for accuracy

# Print results
print("-----------------------------------")
print("Comparison of Beam Radius Calculation Methods:")
print("-----------------------------------")
print(f"True Beam Radius (rho(z)) at z={z_pos*1e3:.2f} mm: {true_rho_z*1e6:.3f} µm")
print(f"Intensity at r = rho(z)/sqrt(2): {intensity_reference:.6f} W/m^2")
print(f"Initial Guess for N-R: {rho_initial*1e6:.3f} µm")

print("\nLambert W Function Method:")
print(f"  Extracted rho_LW: {lambert_w_method()*1e6:.3f} µm")
print(f"  Execution Time (per call): {time_lw / 1000 * 1e6:.2f} µs")

print("\nNewton-Raphson Method:")
print(f"  Extracted rho_numerical_NR: {newton_raphson_method()*1e6:.3f} µm")
print(f"  Execution Time (per call): {time_nr / 1000 * 1e6:.2f} µs")

time_ratio_nr = time_nr / time_lw
print(f"\nComputational Time Ratio (Newton-Raphson / LambertW): {time_ratio_nr:.2f}")
print("-----------------------------------")