import numpy as np
from difftactile.main.constants import *


def calculate_wave_speed(density, E, nu):
    lambda_param = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.sqrt((lambda_param + 2 * mu) / density)


def calculate_cfl_timestep(
        phantom_youngs_modulus,
        vitactip_youngs_modulus,
        verbose,
):
    materials = [
        {
            "name": "phantom",
            "density": SYSTEM_PARAMS.phantom.silicone.density,
            "youngs_modulus": phantom_youngs_modulus,
            "poissons_ratio": SYSTEM_PARAMS.phantom.silicone.poissons_ratio,
            "particle_spacing": SYSTEM_PARAMS_COMPUTED.phantom_min_max_particle_spacing,
        },
        {
            "name": "vitactip",
            "density": SYSTEM_PARAMS.vitactip.single_material.density,
            "youngs_modulus": vitactip_youngs_modulus,
            "poissons_ratio": SYSTEM_PARAMS.vitactip.single_material.poissons_ratio,
            "particle_spacing": SYSTEM_PARAMS_COMPUTED.vitactip_min_particle_spacing.all,
        },
    ]
    cfl_number = SYSTEM_PARAMS.meta.target_courant_number
    dt_values = {}
    for material in materials:
        c = calculate_wave_speed(
            material["density"], material["youngs_modulus"], material["poissons_ratio"]
        )
        dt = cfl_number * material["particle_spacing"] / c
        dt_values[material["name"]] = dt
        num_frames_per_second = 1 / (dt * SYSTEM_PARAMS.contact.num_sub_frames)
        if verbose:
            print(f"dt_{material['name']}: {dt:0.3e}")
            print(f"num_frames_per_second_{material['name']}: {num_frames_per_second:0.0f}")
    dt = min(dt_values.values())
    num_frames_per_second = 1 / (dt * SYSTEM_PARAMS.contact.num_sub_frames)
    if verbose:
        print(f"\nrequired dt: {dt:0.3e}")
        print(f"num_frames_per_second: {num_frames_per_second:0.0f}")
    dt = min(dt, 1.0e-5)
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
    vitactip_min_particle_spacing_all = (
        SYSTEM_PARAMS_COMPUTED.vitactip_min_particle_spacing.all
    )
    phantom_contact_volume = contact_area * (phantom_min_max_particle_spacing * 2)
    vitactip_contact_volume = contact_area * vitactip_min_particle_spacing_all
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
    return {
        "estimated": {
            "normal_stiffness": k_n_estimated,
            "tangential_stiffness": k_t_estimated,
            "critical_damping": c_critical_estimated,
            "normal_damping": c_recommended,
            "damping_ratio": damping_ratio_estimated,
            "friction_coefficient": mu_estimated,
        },
        "current": {
            "normal_stiffness": k_n_current,
            "tangential_stiffness": k_t_current,
            "normal_damping": c_current,
            "critical_damping": c_critical_current,
            "damping_ratio": damping_ratio_current,
            "friction_coefficient": mu_current,
        },
    }


def main():
    calculate_cfl_timestep(
        phantom_youngs_modulus=SYSTEM_PARAMS.phantom.silicone.youngs_modulus,
        vitactip_youngs_modulus=SYSTEM_PARAMS.vitactip.single_material.youngs_modulus,
        verbose=True,
    )
    calculate_contact_parameters()
