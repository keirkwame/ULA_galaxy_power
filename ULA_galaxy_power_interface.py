#!python3
import ULA_galaxy_power as ULA

def setup(options):
    """Setup likelihood class."""
    theory_instance = ULA.RSD_model()
    return theory_instance

def execute(block, config):
    """Execute theory calculation (galaxy clustering multipoles) for input linear cosmology."""
    theory_instance = config

    #Load linear cosmology from block
    #Get cosmological parameters
    omega_m = block['cosmological_parameters', 'omega_m']
    h = block['cosmological_parameters', 'h0']
    #Get angular diameter distance
    z_distance = block['distances', 'z']
    h_z = block['distances', 'h']
    d_a = block['distances', 'd_a'] * h #Mpc/h
    #Get linear matter power spectrum
    k, z_power, pk = block.get_grid('matter_power_lin', 'k_h', 'z', 'p_k') #Check order #h/Mpc, (Mpc/h)^3
