import numpy as np


class RobotTrajectories(object):
    def __init__(self, n_batch, seq_length, n_periods, dt_step=.001, sine_seed=3000, static=False):
        self.n_sines = 5
        self.periods = np.array([.5, 1., .5])[:self.n_sines]
        self.seq_length = seq_length
        self.n_periods = n_periods
        self.n_batch = n_batch
        self.random_state = np.random.RandomState(seed=sine_seed)
        self.seed = sine_seed
        self.static = static
        self.dt_step = dt_step
    

    @property
    def shape(self):
        return (self.n_batch, self.seq_length), (self.n_batch, self.seq_length), (self.n_batch, self.seq_length),\
               (self.n_batch, self.seq_length)

    def generate_data(self, n_samples):
        data = []
        t = np.linspace(0, 1 * 2 * np.pi, self.seq_length // self.n_periods)
        for _ in range(n_samples):
            if self.static:
                self.random_state = np.random.RandomState(seed=self.seed)

        
            periods = self.random_state.rand(self.n_batch, self.n_sines) * .7 + .3
            phase_motor0 = self.random_state.rand(self.n_batch, self.n_sines) * np.pi * 2
            amp0 = self.random_state.rand(self.n_batch, self.n_sines) * 2.5 * 30
            omega0 = np.sin(t[..., None, None] / periods[None, ...] + phase_motor0[None, ...]) * amp0[None, ...]
            omega0 = omega0.sum(-1)
            phi0 = self.dt_step * np.cumsum(omega0, 0)
            phi0_max = np.max(phi0, 0)
            phi0_min = np.min(phi0, 0)
            selector = np.logical_or(phi0_max > np.pi / 2, phi0_min < -np.pi / 2)
            sc = (np.pi / 2) / phi0_max[selector]
            sc2 = (-np.pi / 2) / phi0_min[selector]
            sc[sc < 0.] = 1.
            sc2[sc2 < 0.] = 1.
            sc = np.min((sc, sc2), 0)
            phi0[:, selector] = sc[None, :] * phi0[:, selector]
            omega0[:, selector] = sc[None, :] * omega0[:, selector]
            assert np.all(np.abs(phi0) - 1e-5 <= np.pi / 2)

            phase_motor1 = self.random_state.rand(self.n_batch, self.n_sines) * np.pi * 2
            # amp1 = self.random_state.rand(self.n_batch, self.n_sines) * 1.5
            amp1 = self.random_state.rand(self.n_batch, self.n_sines) * 10.0 
            periods = self.random_state.rand(self.n_batch, self.n_sines) * .7 + .3
            omega1 = np.sin(t[..., None, None] / periods[None, ...] + phase_motor1[None, ...]) * amp1[None, ...]
            omega1 = (omega1 / (omega1.max(0) - omega1.min(0)) * 20.).sum(-1)
            phi1_rel = self.dt_step * np.cumsum(omega1, 0)            
            phi1_max = np.max(phi1_rel, 0)
            phi1_min = np.min(phi1_rel, 0)
            selector = np.logical_or(phi1_max > np.pi / 2, phi1_min < -np.pi / 2)
            sc = (np.pi / 2) / phi1_max[selector]
            sc2 = (-np.pi / 2) / phi1_min[selector]
            sc[sc < 0.] = 1.
            sc2[sc2 < 0.] = 1.
            sc = np.min((sc, sc2), 0)
            phi1_rel[:, selector] = sc[None, :] * phi1_rel[:, selector]
            assert np.all(np.abs(phi1_rel) - 1e-5 <= np.pi / 2)
            omega1[:, selector] = sc[None, :] * omega1[:, selector]
            phi1 = phi0 + phi1_rel + np.pi / 2

            # scale_factor = 100  # Increase the scale factor to increase the range of x and y
            # x = (np.cos(phi0) + np.cos(phi1)).T * scale_factor
            # y = (np.sin(phi0) + np.sin(phi1)).T * scale_factor
            # x = (np.cos(phi0) + np.cos(phi1)).T * .5
            # y = (np.sin(phi0) + np.sin(phi1)).T * .5
         
          
            # x = (np.cos(phi0) + np.cos(phi1)).T * (self.random_state.rand(self.n_batch, self.seq_length) * 100 + 1.0)
            # y = (np.sin(phi0) + np.sin(phi1)).T * (self.random_state.rand(self.n_batch, self.seq_length) * 100 + 1.0)
            # x = np.tile(x[:, :], (1, 2))
            # y = np.tile(y[:, :], (1, 2))
            # x = (np.cos(phi0) + np.cos(phi1)).T * (self.random_state.rand(self.n_batch, self.seq_length) * 10 + 1)
            # y = (np.sin(phi0) + np.sin(phi1)).T * (self.random_state.rand(self.n_batch, self.seq_length) * 10 + 1)


            
            x = (np.cos(phi0) + np.cos(phi1)).T 
            y = (np.sin(phi0) + np.sin(phi1)).T 

            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)

                # Apply the scaling and shift the range to start from 1
            x = 1 + (x - x_min) * (9 / (x_max - x_min))
            y = 1 + (y - y_min) * (9 / (y_max - y_min))

            x = np.tile(x[:, :], (1, 2))
            y = np.tile(y[:, :], (1, 2))
            omega0 = np.tile(omega0.T[:, :], (1, 2))
            omega1 = np.tile(omega1.T[:, :], (1, 2))

            data.append((omega0[:, :self.seq_length], omega1[:, :self.seq_length], x[:, :self.seq_length], y[:, :self.seq_length]))
        return data
    
