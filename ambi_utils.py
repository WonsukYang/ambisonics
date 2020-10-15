from math import sqrt, pi, factorial
import numpy as np

class Position: 
    '''
    Position in spherical coordinates
    '''
    def __init__(self, phi, nu, radius=1.):
        self.phi, self.nu, self.r = phi, nu, radius
    
    def _radian_to_degree(self, rad):
        return 180 / np.pi * rad 

    def __str__(self):
        return "phi : {}, nu : {}, radius : {}".format(self._radian_to_degree(self.phi),
                                                       self._radian_to_degree(self.nu),
                                                       self.r)

 
def index_to_degree_order(index):
    order = int(sqrt(index))
    index -= order**2
    degree = index - order
    return order, degree


def normalization_factor(order, degree):
    return sqrt((2. - float(degree == 0) * float(factorial(order - abs(degree)) / float(factorial(order + abs(degree))))))
    

# single spherical harmonics functions
def spherical_harmonic_mn(order, degree, phi, nu):
    from scipy.special import lpmv
    norm = normalization_factor(order, degree)
    sph = (-1)**degree * norm * lpmv(abs(degree), order, np.sin(nu)) * (np.cos(abs(degree) * phi) if degree >= 0 else np.sin(abs(degree) * phi))
    return sph


def spherical_harmonics_matrix(positions, max_order):
    num_channels = int((max_order + 1) ** 2)  
    sph_mat = np.zeros((len(positions), num_channels))
    for i, p in enumerate(positions):
        for j in range(num_channels):
            order, degree = index_to_degree_order(j)   
            sph_mat[i][j] = spherical_harmonic_mn(order, degree, p.phi, p.nu)
    return sph_mat