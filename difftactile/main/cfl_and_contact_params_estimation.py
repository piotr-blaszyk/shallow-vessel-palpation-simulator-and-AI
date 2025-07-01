import numpy as np

from difftactile.main.constants import *

# Helper function to calculate wave speed for a material
def calculate_wave_speed(density, E, nu):
    lambda_param = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.sqrt((lambda_param + 2 * mu) / density)

def calculate_cfl_timestep():
    """
    Calculate the stable time step based on CFL condition for mixed MPM-FEM system.
    
    Returns:
        float: Recommended maximum stable time step in seconds
    """
    
    if SYSTEM_PARAMS.vitactip.number_of_materials == 1:
        materials = [
            {
                'name': 'phantom',
                'density': SYSTEM_PARAMS.phantom.silicone.density,
                'youngs_modulus': SYSTEM_PARAMS.phantom.silicone.youngs_modulus,
                'poissons_ratio': SYSTEM_PARAMS.phantom.silicone.poissons_ratio,
                'particle_spacing': SYSTEM_PARAMS_COMPUTED.phantom_min_max_particle_spacing
            },
            {
                'name': 'vitactip',
                'density': SYSTEM_PARAMS.vitactip.single_material.density,
                'youngs_modulus': SYSTEM_PARAMS.vitactip.single_material.youngs_modulus,
                'poissons_ratio': SYSTEM_PARAMS.vitactip.single_material.poissons_ratio,
                'particle_spacing': SYSTEM_PARAMS_COMPUTED.vitactip_min_particle_spacing.all
            }
        ]
    else:
        # Define all materials and their properties
        materials = [
            {
                'name': 'phantom',
                'density': SYSTEM_PARAMS.phantom.silicone.density,
                'youngs_modulus': SYSTEM_PARAMS.phantom.silicone.youngs_modulus,
                'poissons_ratio': SYSTEM_PARAMS.phantom.silicone.poissons_ratio,
                'particle_spacing': SYSTEM_PARAMS_COMPUTED.phantom_min_max_particle_spacing
            },
            {
                'name': 'vitactip_shell',
                'density': SYSTEM_PARAMS.vitactip.shell.density,
                'youngs_modulus': SYSTEM_PARAMS.vitactip.shell.youngs_modulus,
                'poissons_ratio': SYSTEM_PARAMS.vitactip.shell.poissons_ratio,
                'particle_spacing': SYSTEM_PARAMS_COMPUTED.vitactip_min_particle_spacing.shell
            },
            {
                'name': 'vitactip_gel',
                'density': SYSTEM_PARAMS.vitactip.gel.density,
                'youngs_modulus': SYSTEM_PARAMS.vitactip.gel.youngs_modulus,
                'poissons_ratio': SYSTEM_PARAMS.vitactip.gel.poissons_ratio,
                'particle_spacing': SYSTEM_PARAMS_COMPUTED.vitactip_min_particle_spacing.gel
            }
        ]

    # Calculate dt for each material
    # C can be between 0.0 (safest simulation) and 1.0 (fastest simulation)
    cfl_number = SYSTEM_PARAMS.meta.target_courant_number
    dt_values = {}
    
    for material in materials:
        # Calculate wave speed
        c = calculate_wave_speed(
            material.density,
            material.youngs_modulus,
            material.poissons_ratio
        )
        
        # Calculate dt
        dt = cfl_number * material.particle_spacing / c
        dt_values[material.name] = dt
        num_frames_per_second = 1 / (dt * SYSTEM_PARAMS.contact.num_sub_frames)
        print(f"dt_{material.name}: {dt:0.3e}")
        print(f"num_frames_per_second_{material.name}: {num_frames_per_second:0.0f}")

    # Take minimum dt for stability
    dt = min(dt_values.values())
    num_frames_per_second = 1 / (dt * SYSTEM_PARAMS.contact.num_sub_frames)
    print(f'\nrequired dt: {dt:0.3e}')
    print(f'num_frames_per_second: {num_frames_per_second:0.0f}')

def calculate_critical_damping(contact_stiffness, effective_mass):
    """
    Calculate critical damping coefficient for contact.
    
    Args:
        contact_stiffness: Normal contact stiffness (N/m)
        effective_mass: Effective mass of the contact point (kg)
        
    Returns:
        float: Critical damping coefficient (N*s/m)
    """
    return 2 * np.sqrt(contact_stiffness * effective_mass)

def estimate_normal_stiffness(E1, E2, contact_area):
    """
    Estimate normal contact stiffness using average Young's modulus
    and contact area. Based on Hertz contact theory simplification.
    """
    # Use harmonic mean of Young's moduli
    E_effective = (2 * E1 * E2) / (E1 + E2)
    # Rule of thumb: k_n ≈ E_effective * contact_area
    return E_effective * contact_area

def estimate_tangential_stiffness(k_n):
    """
    Estimate tangential stiffness based on normal stiffness.
    Common rule: k_t ≈ 0.7 * k_n
    """
    return 0.7 * k_n

def calculate_contact_parameters():
    """
    Calculate contact parameters including critical damping.
    Uses actual contact surface area for more accurate mass calculation.
    Also estimates initial contact parameters using rule-of-thumb approaches.
    """
    contact_params = SYSTEM_PARAMS.contact
    SYSTEM_PARAMS.vitactip = SYSTEM_PARAMS.vitactip
    phantom_params = SYSTEM_PARAMS.phantom
    
    # Get material properties
    if SYSTEM_PARAMS.vitactip.number_of_materials == 1:
        E_vitactip = SYSTEM_PARAMS.vitactip.single_material.youngs_modulus
        rho_vitactip = SYSTEM_PARAMS.vitactip.single_material.density
    else:
        # Use shell properties as it dominates contact behavior
        E_vitactip = SYSTEM_PARAMS.vitactip.shell.youngs_modulus
        rho_vitactip = SYSTEM_PARAMS.vitactip.shell.density
    
    E_phantom = phantom_params.silicone.youngs_modulus
    rho_phantom = phantom_params.silicone.density
    
    # Get contact surface area
    contact_area = SYSTEM_PARAMS_COMPUTED.contact_surface_area  # m²
    phantom_min_max_particle_spacing = SYSTEM_PARAMS_COMPUTED.phantom_min_max_particle_spacing
    vitactip_min_particle_spacing_all = SYSTEM_PARAMS_COMPUTED.vitactip_min_particle_spacing.all
    
    # Estimate contact volumes for both objects
    phantom_contact_volume = contact_area * (phantom_min_max_particle_spacing * 2)
    vitactip_contact_volume = contact_area * vitactip_min_particle_spacing_all
    
    # Calculate masses of contact regions
    m1 = vitactip_contact_volume * rho_vitactip  # kg
    m2 = phantom_contact_volume * rho_phantom   # kg
    
    # Calculate effective mass using harmonic mean
    effective_mass = (m1 * m2) / (m1 + m2)
    
    # Estimate contact parameters
    k_n_estimated = estimate_normal_stiffness(E_vitactip, E_phantom, contact_area)
    k_t_estimated = estimate_tangential_stiffness(k_n_estimated)
    
    # Calculate critical damping using estimated normal stiffness
    c_critical_estimated = calculate_critical_damping(k_n_estimated, effective_mass)
    # Calculate recommended normal damping (slightly underdamped with ζ = 0.7)
    target_damping_ratio = 0.7
    c_recommended = target_damping_ratio * c_critical_estimated
    damping_ratio_estimated = target_damping_ratio  # By definition
    mu_estimated = 0.85
    
    # Get current parameters (for comparison)
    k_n_current = contact_params.normal_stiffness
    k_t_current = contact_params.tangential_stiffness
    c_current = contact_params.normal_damping
    mu_current = contact_params.coulomb_friction_coeff

    # Calculate current damping ratio for comparison
    c_critical_current = calculate_critical_damping(k_n_current, effective_mass)
    damping_ratio_current = c_current / c_critical_current
    
    print("\nEstimated Parameters (Rule of Thumb):")
    print(f"Normal stiffness: {k_n_estimated:.2e} N/m")
    print(f"Tangential stiffness: {k_t_estimated:.2e} N/m")
    print(f"Critical damping coefficient: {c_critical_estimated:.2e} N*s/m")
    print(f"Recommended normal damping: {c_recommended:.2e} N*s/m")
    print(f"Target damping ratio (ζ): {damping_ratio_estimated:.3f}")
    print(f"Recommended friction coefficient: {mu_estimated:.2f} (silicone-silicone contact)")
    
    # Assess damping state based on estimated parameters
    print("\nDamping Assessment (based on estimated parameters):")
    if damping_ratio_estimated > 1:
        print("System is overdamped - slower response, no oscillations")
    elif damping_ratio_estimated < 1:
        print("System is underdamped - faster response but with oscillations")
    else:
        print("System is critically damped - fastest response without oscillations")
    
    return {
        'estimated': {
            'normal_stiffness': k_n_estimated,
            'tangential_stiffness': k_t_estimated,
            'critical_damping': c_critical_estimated,
            'normal_damping': c_recommended,
            'damping_ratio': damping_ratio_estimated,
            'friction_coefficient': mu_estimated
        },
        'current': {
            'normal_stiffness': k_n_current,
            'tangential_stiffness': k_t_current,
            'normal_damping': c_current,
            'critical_damping': c_critical_current,
            'damping_ratio': damping_ratio_current,
            'friction_coefficient': mu_current
        }
    }

if __name__ == '__main__':
    calculate_cfl_timestep()
    calculate_contact_parameters()
