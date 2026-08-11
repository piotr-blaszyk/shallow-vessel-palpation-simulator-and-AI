import json

import numpy as np

from difftactile.main.constants import *
from difftactile.main.paths import repo_path


def calculate_wave_speed(density, E, nu):
    lambda_param = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.sqrt((lambda_param + 2 * mu) / density)


def calculate_cfl_timestep(
        phantom_healthy_youngs_modulus,
        vitactip_youngs_modulus,
        courant_number,
        verbose,
):
    materials = [
        {
            "name": "phantom",
            "density": SYSTEM_PARAMS.phantom.silicone.density,
            "youngs_modulus": phantom_healthy_youngs_modulus,
            "poissons_ratio": SYSTEM_PARAMS.phantom.silicone.poissons_ratio,
            "particle_spacing": SYSTEM_PARAMS.phantom.mpm_grid_cube_size,
        },
        {
            "name": "vitactip",
            "density": SYSTEM_PARAMS.vitactip.single_material.density,
            "youngs_modulus": vitactip_youngs_modulus,
            "poissons_ratio": SYSTEM_PARAMS.vitactip.single_material.poissons_ratio,
            "particle_spacing": SYSTEM_PARAMS_COMPUTED.vitactip_mean_particle_spacing,
        },
    ]
    dt_values = {}
    for material in materials:
        c = calculate_wave_speed(
            material["density"], material["youngs_modulus"], material["poissons_ratio"]
        )
        dt = courant_number * material["particle_spacing"] / c
        dt_values[material["name"]] = dt
    dt = min(dt_values.values())
    if verbose:
        print(f"uncapped dt: {dt:0.3e}")
    dt = min(dt, 5.0e-5)
    return dt


def calculate_critical_damping(contact_stiffness, effective_mass):
    return 2 * np.sqrt(contact_stiffness * effective_mass)


def estimate_normal_stiffness(E1, E2, contact_area):
    E_effective = (2 * E1 * E2) / (E1 + E2)
    return E_effective * contact_area


def estimate_tangential_stiffness(k_n):
    return 0.7 * k_n


def calculate_contact_parameters():
    contact_params = SYSTEM_PARAMS.contact
    SYSTEM_PARAMS.vitactip = SYSTEM_PARAMS.vitactip
    phantom_params = SYSTEM_PARAMS.phantom
    if SYSTEM_PARAMS.vitactip.number_of_materials == 1:
        E_vitactip = SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
        rho_vitactip = SYSTEM_PARAMS.vitactip.single_material.density
    else:
        E_vitactip = SYSTEM_PARAMS.vitactip.shell.youngs_modulus
        rho_vitactip = SYSTEM_PARAMS.vitactip.shell.density
    E_phantom = phantom_params.silicone.youngs_modulus
    rho_phantom = phantom_params.silicone.density
    contact_area = SYSTEM_PARAMS_COMPUTED.contact_surface_area
    phantom_min_max_particle_spacing = (
        SYSTEM_PARAMS_COMPUTED.phantom_min_max_particle_spacing
    )
    vitactip_mean_particle_spacing_all = (
        SYSTEM_PARAMS_COMPUTED.vitactip_mean_particle_spacing
    )
    phantom_contact_volume = contact_area * (phantom_min_max_particle_spacing * 2)
    vitactip_contact_volume = contact_area * vitactip_mean_particle_spacing_all
    m1 = vitactip_contact_volume * rho_vitactip
    m2 = phantom_contact_volume * rho_phantom
    effective_mass = (m1 * m2) / (m1 + m2)
    k_n_estimated = estimate_normal_stiffness(E_vitactip, E_phantom, contact_area)
    k_t_estimated = estimate_tangential_stiffness(k_n_estimated)
    c_critical_estimated = calculate_critical_damping(k_n_estimated, effective_mass)
    target_damping_ratio = 0.7
    c_recommended = target_damping_ratio * c_critical_estimated
    damping_ratio_estimated = target_damping_ratio
    mu_estimated = 0.85
    k_n_current = contact_params.normal_stiffness
    k_t_current = contact_params.tangential_stiffness
    c_current = contact_params.normal_damping
    mu_current = contact_params.coulomb_friction_coeff
    c_critical_current = calculate_critical_damping(k_n_current, effective_mass)
    damping_ratio_current = c_current / c_critical_current
    print("\nEstimated Parameters (Rule of Thumb):")
    print(f"Normal stiffness: {k_n_estimated:.2e} N/m")
    print(f"Tangential stiffness: {k_t_estimated:.2e} N/m")
    print(f"Critical damping coefficient: {c_critical_estimated:.2e} N*s/m")
    print(f"Recommended normal damping: {c_recommended:.2e} N*s/m")
    print(f"Target damping ratio (ζ): {damping_ratio_estimated:.3f}")
    print(
        f"Recommended friction coefficient: {mu_estimated:.2f} (silicone-silicone contact)"
    )
    print("\nDamping Assessment (based on estimated parameters):")
    if damping_ratio_estimated > 1:
        print("System is overdamped - slower response, no oscillations")
    elif damping_ratio_estimated < 1:
        print("System is underdamped - faster response but with oscillations")
    else:
        print("System is critically damped - fastest response without oscillations")

    # Load the current system parameters
    system_params_file = repo_path("difftactile/system_params/system-params.json")
    with open(system_params_file, 'r') as f:
        system_params = json.load(f)

    # Update the contact parameters with computed values
    system_params['contact']['normal_stiffness'] = float(k_n_estimated)
    system_params['contact']['tangential_stiffness'] = float(k_t_estimated)
    system_params['contact']['normal_damping'] = float(c_recommended)
    system_params['contact']['coulomb_friction_coeff'] = float(mu_estimated)

    # Write the updated parameters back to file
    with open(system_params_file, 'w') as f:
        json.dump(system_params, f, indent=4)

    print("\nSystem parameters file updated with computed contact parameters.")


def main():
    calculate_cfl_timestep(
        phantom_healthy_youngs_modulus=SYSTEM_PARAMS.phantom.silicone.youngs_modulus,
        vitactip_youngs_modulus=SYSTEM_PARAMS.vitactip.single_material.youngs_modulus,
        courant_number=SYSTEM_PARAMS.meta.target_courant_number,
        verbose=True,
    )
    calculate_contact_parameters()
