import numpy as np
import json

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
    with open('../tasks/system-params.json', 'r') as f:
        system_params = json.load(f)
    with open('../tasks/system-params-computed.json', 'r') as f:
        system_params_computed = json.load(f)
    
    vitactip_params = system_params['vitactip']
    if vitactip_params['number_of_materials'] == 1:
        materials = [
            {
                'name': 'phantom',
                'density': system_params['phantom']['silicone']['density'],
                'youngs_modulus': system_params['phantom']['silicone']['youngs_modulus'],
                'poissons_ratio': system_params['phantom']['silicone']['poissons_ratio'],
                'particle_spacing': system_params_computed['phantom_particle_spacing']
            },
            {
                'name': 'vitactip',
                'density': system_params['vitactip']['single_material']['density'],
                'youngs_modulus': system_params['vitactip']['single_material']['youngs_modulus'],
                'poissons_ratio': system_params['vitactip']['single_material']['poissons_ratio'],
                'particle_spacing': system_params_computed['vitactip_minimum_particle_spacing']['all']
            }
        ]
    else:
        # Define all materials and their properties
        materials = [
            {
                'name': 'phantom',
                'density': system_params['phantom']['silicone']['density'],
                'youngs_modulus': system_params['phantom']['silicone']['youngs_modulus'],
                'poissons_ratio': system_params['phantom']['silicone']['poissons_ratio'],
                'particle_spacing': system_params_computed['phantom_particle_spacing']
            },
            {
                'name': 'vitactip_shell',
                'density': system_params['vitactip']['shell']['density'],
                'youngs_modulus': system_params['vitactip']['shell']['youngs_modulus'],
                'poissons_ratio': system_params['vitactip']['shell']['poissons_ratio'],
                'particle_spacing': system_params_computed['vitactip_minimum_particle_spacing']['shell']
            },
            {
                'name': 'vitactip_gel',
                'density': system_params['vitactip']['gel']['density'],
                'youngs_modulus': system_params['vitactip']['gel']['youngs_modulus'],
                'poissons_ratio': system_params['vitactip']['gel']['poissons_ratio'],
                'particle_spacing': system_params_computed['vitactip_minimum_particle_spacing']['gel']
            }
        ]

    # Calculate dt for each material
    cfl_number = 0.3
    dt_values = {}
    
    for material in materials:
        # Calculate wave speed
        c = calculate_wave_speed(
            material['density'],
            material['youngs_modulus'],
            material['poissons_ratio']
        )
        
        # Calculate dt
        dt = cfl_number * material['particle_spacing'] / c
        dt_values[material['name']] = dt
        num_frames_per_second = 1 / (dt * system_params['contact']['num_sub_frames'])
        print(f"dt_{material['name']}: {dt:0.3e}")
        print(f"num_frames_per_second_{material['name']}: {num_frames_per_second:0.0f}")

    # Take minimum dt for stability
    dt = min(dt_values.values())
    num_frames_per_second = 1 / (dt * system_params['contact']['num_sub_frames'])
    print(f'\nrequired dt: {dt:0.3e}')
    print(f'num_frames_per_second: {num_frames_per_second:0.0f}')

if __name__ == '__main__':
    calculate_cfl_timestep()
