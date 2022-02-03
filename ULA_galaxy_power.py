from classy import Class
from pybird import pybird

class RSD_model:
    """Class to do galaxy clustering redshift-space distortion calculations in the presence of ultra-light axions."""
    def __init__(self):

        # Output: monopole or quadrupole
        self.Pk_ell = []
        self.sigma8 = 0  # derived parameter

    def calc_rsd(self, ell, rsd_params, cosmo_params, redshift):
        '''With PyBird code'''

        # cosmology and k range
        k_array = np.linspace(0.001, 0.5, 50)
        Om_AP = 0.31  # (cosmo_params[2]/100 + cosmo_params[3]) / cosmo_params[0]**2

        if redshift == 'high_z':
            zpk = 0.57
        elif redshift == 'low_z':
            zpk = 0.32
        else:
            print("Invalid redshift choice")

        z_AP = zpk

        try:
            M = Class()
            M.set({'h': cosmo_params[0],
                   'ln10^{10}A_s': cosmo_params[1],
                   'omega_b': cosmo_params[2] / 100,
                   'omega_cdm': cosmo_params[3],
                   'n_s': 0.9611,
                   'N_ncdm': 0})

            M.set({'output': 'mPk',
                   'P_k_max_1/Mpc': 10.0,
                   'z_max_pk': zpk})

            M.compute()

            DA = M.angular_distance(zpk) * M.Hubble(0.)
            H = M.Hubble(zpk) / M.Hubble(0.)
            f = M.scale_independent_growth_factor_f(zpk)
            kk = np.logspace(-4, 1, 200)
            Pk = [M.pk(ki * M.h(), zpk) * M.h() ** 3 for ki in kk]

            self.sigma8 = M.sigma8()

            if include_axions:
                # Adjust Pk with square of axion transfer function
                if redshift == 'high_z':
                    T2 = np.array([final_T_high_z(k_val, cosmo_params[4], 100 * cosmo_params[0], cosmo_params[2] / 100,
                                                  cosmo_params[3]) for k_val in kk])
                elif redshift == 'low_z':
                    T2 = np.array([final_T_low_z(k_val, cosmo_params[4], 100 * cosmo_params[0], cosmo_params[2] / 100,
                                                 cosmo_params[3]) for k_val in kk])

                Pk *= T2

                # New sigma8 calc
                Pk_z0 = [M.pk(ki * M.h(), 0.) * M.h() ** 3 for ki in kk]
                Pk_z0 *= T2  # need to change to have a z=0 version of T_ax interpolation... if it matters
                R8 = 8.
                Ws = 3 * (np.sin(kk * R8) - kk * R8 * np.cos(kk * R8)) / kk ** 3 / R8 ** 3
                self.sigma8 = np.sqrt(simps(Pk_z0 * kk ** 2 * Ws ** 2, kk) / 2 / np.pi ** 2)

            # now that the linear theory part is done, on to the RSD
            # initialize PyBird object
            common = pybird.Common(optiresum=True)
            nonlinear = pybird.NonLinear(load=True, save=True, co=common)
            resum = pybird.Resum(co=common)
            projection = pybird.Projection(k_array, Om_AP, z_AP, co=common)

            # now deprecated
            # path       = '/home/r/rbond/alague/scratch/RSD/pybird/'
            '''
            if dataset == 'NGC' and redshift == 'high':
                projection = pybird.Projection(k_array, Om_AP, z_AP,
                                               window_fourier_name='pynest_BOSS_CMASS_NGC_z057',
                                               path_to_window=path+'./montepython_tree/data/pybird/Window',
                                               window_configspace_file=path+'./montepython_tree/data/pybird/window_BOSS_CMASS_NGC_z057.dat')

            elif dataset == 'SGC' and redshift == 'high':
                projection = pybird.Projection(k_array, Om_AP, z_AP,
                                               window_fourier_name='pynest_BOSS_CMASS_SGC_z057',
                                               path_to_window=path+'./montepython_tree/data/pybird/Window',
                                               window_configspace_file=path+'./montepython_tree/data/pybird/window_BOSS_CMASS_SGC_z057.dat')

            elif dataset == 'NGC' and redshift == 'low':
                projection = pybird.Projection(k_array, Om_AP, z_AP,
                                               window_fourier_name='pynest_BOSS_LOWZ_NGC_z038',
                                               path_to_window=path+'./montepython_tree/data/pybird/Window',
                                               window_configspace_file=path+'./montepython_tree/data/pybird/window_BOSS_LOWZ_NGC_z038.dat')
            else:
                projection = pybird.Projection(k_array, Om_AP, z_AP,
                                               window_fourier_name='pynest_BOSS_LOWZ_SGC_z038',
                                               path_to_window=path+'./montepython_tree/data/pybird/Window',
                                               window_configspace_file=path+'./montepython_tree/data/pybird/window_BOSS_LOWZ_SGC_z038.dat')

            '''
            bird = pybird.Bird(kk, Pk, f, DA, H, zpk, which='full', co=common)
            bs = [rsd_params[0], rsd_params[1], rsd_params[2], rsd_params[1], rsd_params[4], rsd_params[5],
                  0]  # rsd_params [b1,b2,b3,b4,cct,c0,c2]

            # run calculation

            nonlinear.PsCf(bird)
            bird.setPsCf(bs)
            bird.setfullPs()
            resum.Ps(bird)
            projection.AP(bird)
            # projection.Window(bird)
            projection.kdata(bird)

            # collect final result and output with k-scaling
            monopole = interp1d(k_array, bird.fullPs[0], bounds_error=False)
            quadrupole = interp1d(k_array, bird.fullPs[1], bounds_error=False)
            k_M = 0.7
            if ell == 0:
                self.Pk_ell = lambda k: monopole(k) + (rsd_params[3] * 1000 + rsd_params[6] * 1000 * (k / k_M) ** 2)

            elif ell == 2:
                Ivanov_method = True
                if Ivanov_method:
                    self.Pk_ell = lambda k: quadrupole(k) + 4. / 15 * f * (
                                rsd_params[3] * 1000 + rsd_params[6] * 1000 * (k / k_M) ** 2)
                else:
                    self.Pk_ell = quadrupole

            print("Good params")
            return

        except:
            print(traceback.format_exc())
            print("Bad params")
            self.Pk_ell = lambda x: -1e100
        return
