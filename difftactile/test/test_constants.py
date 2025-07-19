from difftactile.main.constants import SYSTEM_PARAMS

def main():
    # Test dot notation
    dot_value = SYSTEM_PARAMS.optimisation.update_steps.vitactip.youngs_modulus
    print("Dot notation value:", dot_value)
    
    # Test bracket notation
    bracket_value = SYSTEM_PARAMS.optimisation.update_steps.vitactip['youngs_modulus']
    print("Bracket notation value:", bracket_value)
    
    # Test mixed notation
    mixed_value = SYSTEM_PARAMS.optimisation.update_steps['vitactip'].youngs_modulus
    print("Mixed notation value:", mixed_value)
    
    # Verify all values are equal
    assert dot_value == bracket_value == mixed_value, "All access methods should return the same value"
    print("All tests passed!")