#!python3
import copy as cp
import numpy as np
from pybird import pybird as pyb
#import ULA_galaxy_power as ULA

def setup(options):
    """Setup likelihood class."""
    #theory_instance = ULA.RSD_model()
    return 0 #theory_instance

def execute(block, config):
    """Execute theory calculation (galaxy clustering multipoles) for input linear cosmology."""
    #theory_instance = config
    #Get ULA galaxy power parameters
    b1 = block['EFT_of_LSS', 'b1']
    b2 = block['EFT_of_LSS', 'b2']
    b3 = block['EFT_of_LSS', 'b3']
    #b4 = block['EFT_of_LSS', 'b4']
    #if b4 == 'fix_to_b2':
    b4 = 0. #cp.deepcopy(b2)
    c_ct = block['EFT_of_LSS', 'c_ct']
    c_r1 = block['EFT_of_LSS', 'c_r1']
    c_r2 = block['EFT_of_LSS', 'c_r2']
    c_e1 = block['EFT_of_LSS', 'c_e1_1000'] * 1000.
    c_e2 = block['EFT_of_LSS', 'c_e2_1000'] * 1000.

    ##How to input array in ini file?
    #z_survey = block['EFT_of_LSS', 'z_survey']
    z_survey = np.array([0.32, 0.57])

    omega_m_AP = block.get('EFT_of_LSS', 'omega_m_AP') #, default=0.31)
    k_M = block.get('EFT_of_LSS', 'k_M') #, default=0.70)
    k_min = block.get('EFT_of_LSS', 'k_min') #, default=0.001)
    k_max = block.get('EFT_of_LSS', 'k_max') #, default=0.5)
    nk = block.get('EFT_of_LSS', 'nk') #, default=50)
    k = np.linspace(k_min, k_max, num=nk)

    # Load linear cosmology from block
    #Get cosmological parameters
    #omega_c_h2 = block['cosmological_parameters', 'omch2']
    h = block['cosmological_parameters', 'h0']

    #Get axion parameters
    #m_ax = 10. ** block['axion_parameters', 'm'] #eV
    #omega_ax_h2 = block['axion_parameters', 'omaxh2']
    #omega_dm_h2 = omega_c_h2 + omega_ax_h2
    #def k_jeans(m_ax, omega_dm_h2, z, h):
    #    """Get the axion Jeans scale."""
    #Get cosmological distances
    z_distance = block['distances', 'z']
    h_z = block['distances', 'h']
    d_a = block['distances', 'd_a'] * h #Mpc/h
    idx_distance_0 = np.where(z_distance == 0.)[0][0]
    print(idx_distance_0)

    #Get logarithmic growth rate
    k_growth, z_growth, f = block.get_grid('linear_cdm_transfer', 'k_h', 'z', 'growth_factor_f') #h/Mpc

    #Get linear matter power spectrum
    k_power, z_power, pk = block.get_grid('matter_power_lin', 'k_h', 'z', 'p_k') #Check order #h/Mpc, (Mpc/h)^3
    print('Matter power:', k_power.shape, z_power.shape, pk.shape)
    ##Check k_power values

    #Set-up Pybird RSD calculation
    monopole = np.zeros((nk, z_survey.shape[0]))
    quadrupole = np.zeros_like(monopole)

    for i, z in enumerate(z_survey):
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
        print('f, d_A, H(z)/H_0', f[0, idx_growth], d_a[idx_distance], h_z[idx_distance] / h_z[idx_distance_0])
        bird = pyb.Bird(k_power, pk[:, idx_power], f[0, idx_growth], d_a[idx_distance],
                        h_z[idx_distance] / h_z[idx_distance_0], z, which='full', co=common)

        #Do RSD calculation
        nonlinear.PsCf(bird)
        bird.setPsCf([b1, b2, b3, b4, c_ct, c_r1, c_r2])
        bird.setfullPs()
        resum.Ps(bird)
        projection.AP(bird)
        projection.kdata(bird)

        #Add stochastic terms
        stochastic_term = c_e1 + (c_e2 * ((k / k_M) ** 2.))
        monopole[:, i] = bird.fullPs[0] + stochastic_term
        quadrupole[:, i] = bird.fullPs[1] + (4. * f[0, idx_growth] * stochastic_term / 15.)

    #Save galaxy multipoles to block
    block.put_grid('EFT_of_LSS', 'k_multipoles', k, 'z_multipoles', z_survey, 'monopole', monopole) #, 'quadrupole',
    #                quadrupole)
    block.put('EFT_of_LSS', 'quadrupole', quadrupole)

    return 0

def cleanup(config):
    """Cleanup."""
    pass
