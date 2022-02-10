#!python3
import copy as cp
import numpy as np
from pybird import pybird as pyb
#import ULA_galaxy_power as ULA

def setup(options):
    """Setup likelihood class."""
    #theory_instance = ULA.RSD_model()
    #Get ULA galaxy power parameters
    nz = options['EFT_of_LSS', 'nz']
    omega_m_AP = options.get('EFT_of_LSS', 'omega_m_AP')
    k_M = options.get('EFT_of_LSS', 'k_M') #, default=0.70)
    b4_fix_to_b2 = options.get('EFT_of_LSS', 'b4_fix_to_b2')

    k_min = options.get('EFT_of_LSS', 'k_min') #, default=0.001)
    k_max = options.get('EFT_of_LSS', 'k_max') #, default=0.5)
    nk = options.get('EFT_of_LSS', 'nk') #, default=50)
    k = np.linspace(k_min, k_max, num=nk)

    return (nz, omega_m_AP, k_M, b4_fix_to_b2, k) #theory_instance

def execute(block, config):
    """Execute theory calculation (galaxy clustering multipoles) for input linear cosmology."""
    #theory_instance = config
    nz, omega_m_AP, k_M, b4_fix_to_b2, k = config
    #Get ULA galaxy power parameters
    #nz = block['EFT_of_LSS', 'nz']
    #omega_m_AP = config #block.get('EFT_of_LSS', 'omega_m_AP') #, default=0.31)
    #k_M = block.get('EFT_of_LSS', 'k_M') #, default=0.70)
    #k_min = block.get('EFT_of_LSS', 'k_min') #, default=0.001)
    #k_max = block.get('EFT_of_LSS', 'k_max') #, default=0.5)
    #nk = block.get('EFT_of_LSS', 'nk') #, default=50)
    #k = np.linspace(k_min, k_max, num=nk)

    # Load linear cosmology from block
    #Get cosmological parameters
    #omega_c_h2 = block['cosmological_parameters', 'omch2']
    #h = block['cosmological_parameters', 'h0']
    #Get axion parameters
    #m_ax = 10. ** block['axion_parameters', 'm'] #eV
    #omega_ax_h2 = block['axion_parameters', 'omaxh2']
    #omega_dm_h2 = omega_c_h2 + omega_ax_h2
    #def k_jeans(m_ax, omega_dm_h2, z, h):
    #    """Get the axion Jeans scale."""
    #Get cosmological distances
    z_distance = block['distances', 'z']
    h_z = block['distances', 'h']
    d_a = block['distances', 'd_a'] #Mpc
    idx_distance_0 = np.where(z_distance == 0.)[0][0]
    print(idx_distance_0)

    #Get logarithmic growth rate
    k_growth, z_growth, f = block.get_grid('linear_cdm_transfer', 'k_h', 'z', 'growth_factor_f') #h/Mpc

    #Get linear matter power spectrum
    k_power, z_power, pk = block.get_grid('matter_power_lin', 'k_h', 'z', 'p_k') #Check order #h/Mpc, (Mpc/h)^3
    print('Matter power:', k_power.shape, z_power.shape, pk.shape)
    ##Check k_power values

    #Set-up Pybird RSD calculation
    z_multipoles = np.zeros(nz)
    monopole = np.zeros((k.shape[0], nz))
    quadrupole = np.zeros_like(monopole)

    for i in range(nz):
        # Get ULA galaxy power parameters
        param_suffix = '_z%i'%(i+1)

        z = block['EFT_of_LSS', param_suffix[1:]]
        b1 = block['EFT_of_LSS', 'b1'+param_suffix]
        b2 = block['EFT_of_LSS', 'b2'+param_suffix]
        b3 = block['EFT_of_LSS', 'b3'+param_suffix]
        if b4_fix_to_b2:
            b4 = cp.deepcopy(b2)
        else:
            b4 = block['EFT_of_LSS', 'b4' + param_suffix]
        # b4 = block['EFT_of_LSS', 'b4']
        # if b4 == 'fix_to_b2':
        #b4 = 0.  # cp.deepcopy(b2)
        c_ct = block['EFT_of_LSS', 'c_ct'+param_suffix]
        c_r1 = block['EFT_of_LSS', 'c_r1'+param_suffix]
        c_r2 = block['EFT_of_LSS', 'c_r2'+param_suffix]
        c_e1 = block['EFT_of_LSS', 'c_e1_1000'+param_suffix] * 1000.
        c_e2 = block['EFT_of_LSS', 'c_e2_1000'+param_suffix] * 1000.

        print('z =', z, np.where(z_distance == z), np.where(np.absolute(z_distance - z) < 1.e-4))
        idx_distance = np.where(np.absolute(z_distance - z) < 1.e-4)[0][0]
        print('z =', z_distance[idx_distance])
        idx_growth = np.where(np.absolute(z_growth - z) < 1.e-4)[0][0]
        print('z =', z_growth[idx_growth])
        idx_power = np.where(np.absolute(z_power - z) < 1.e-4)[0][0]
        print('z =', z_power[idx_power])

        common = pyb.Common(optiresum=True)
        nonlinear = pyb.NonLinear(load=True, save=True, co=common)
        resum = pyb.Resum(co=common)
        projection = pyb.Projection(k, omega_m_AP, z, co=common)
        print('f, d_A, H(z)/H_0', f[0, idx_growth], d_a[idx_distance] * h_z[idx_distance_0],
              h_z[idx_distance] / h_z[idx_distance_0])
        np.savez('matter_power.npz', k_power, pk[:, idx_power])
        bird = pyb.Bird(k_power, pk[:, idx_power], f[0, idx_growth], d_a[idx_distance] * h_z[idx_distance_0],
                        h_z[idx_distance] / h_z[idx_distance_0], z, which='full', co=common)

        #Do RSD calculation
        nonlinear.PsCf(bird)
        bird.setPsCf([b1, b2, b3, b4, c_ct, c_r1, c_r2])
        bird.setfullPs()
        resum.Ps(bird)
        projection.AP(bird)
        projection.kdata(bird)

        #Stochastic term
        stochastic_term = c_e1 + (c_e2 * ((k / k_M) ** 2.))

        #Create output arrays
        z_multipoles[i] = z
        monopole[:, i] = bird.fullPs[0] + stochastic_term
        quadrupole[:, i] = bird.fullPs[1] + (4. * f[0, idx_growth] * stochastic_term / 15.)

    #Save galaxy multipoles to block
    block.put_grid('EFT_of_LSS', 'k_multipoles', k, 'z_multipoles', z_multipoles, 'monopole', monopole) #, 'quadrupole',
    #                quadrupole)
    block.put('EFT_of_LSS', 'quadrupole', quadrupole)

    return 0

def cleanup(config):
    """Cleanup."""
    pass
