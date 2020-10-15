import numpy as np
from ambi_utils import spherical_harmonics_matrix, Position

def spherical_grid(angular_res):
    '''
    Create a unit spherical grid
    '''
    phi_rg = np.flip(np.arange(-180., 180., angular_res) / 180. * np.pi, 0)
    nu_rg = np.arange(-90., 90., angular_res) / 180. * np.pi 
    phi_mesh, nu_mesh = np.meshgrid(phi_rg, nu_rg)
    return phi_mesh, nu_mesh

   
#### Audio Decoder ####

class AmbiDecoder:
    def __init__(self, sph_grid, ambi_order=1): #, ordering='ACN', normalization='SN3D'):
        self.sph_grid = sph_grid
        self.sph_mat = spherical_harmonics_matrix(sph_grid,
                                                  ambi_order) #,
                                                #   ordering,
                                                #   normalization)
        
    def decode(self, samples):
        assert samples.shape[1] == self.sph_mat.shape[1]

        return np.dot(samples, self.sph_mat.T) # sph_mat.T -> num_channel * number of speakers
                                            # ambi -> sample_size * num_channels 
                                            # returns -> samples_size * number of speakers(i.e. each position
                                            #                                              on the spherical grid )
                                            # Hence, decoder gives the amount of sound coming from certain spherical pos.
                                            # in a given sample. 


#### Spherical Ambisonics Visualizer ####

class SphericalAmbisonicsReader:
    def __init__(self, data, rate=48000, window=0.1, angular_res=2.0):
        self.data = data
        self.phi_mesh, self.nu_mesh = spherical_grid(angular_res)
        self.mesh_p = [Position(phi, nu, 1.) for phi, nu in zip(self.phi_mesh.reshape(-1), self.nu_mesh.reshape(-1))]
        
        # Setup decoder
        self.decoder = AmbiDecoder(self.mesh_p, ambi_order=1)

        # Compute spherical energy averaged over consecutive chunks of "window" secs
        self.window_frames = int(window * rate)
        self.n_frames = data.shape[0] / self.window_frames
        self.cur_frame = -1

    def get_next_frame(self):
        self.cur_frame += 1
        if self.cur_frame >= self.n_frames:
            return None

        # Decode ambisonics on a grid of speakers
        chunk_ambi = self.data[self.cur_frame * self.window_frames:((self.cur_frame + 1) * self.window_frames), :]
        decoded = self.decoder.decode(chunk_ambi)
        
        # Compute RMS at each speaker
        rms = np.sqrt(np.mean(decoded ** 2, 0)).reshape(self.phi_mesh.shape)
        
        return np.flipud(rms)

    def loop_frames(self):
        while True:
            rms = self.get_next_frame()
            if rms is None:
                break
            yield rms