import numpy as np
from scipy.interpolate import interp1d
def sampler(cnts, sample_size):
    
    interp_cnts = []
    for c in cnts:
        x,y = c.T

        interp_x = interp1d(np.linspace(0, 1, len(x)), x, kind='cubic')
        interp_y = interp1d(np.linspace(0, 1, len(y)), y, kind='cubic')

        # Interpolate to create new x and y arrays with 'maxN' points
        new_x = interp_x(np.linspace(0, 1, sample_size))
        new_y = interp_y(np.linspace(0, 1, sample_size))

        # Combine the new x and y arrays to form the interpolated contour
        interp_cnt = np.column_stack((new_x, new_y))
        interp_cnts.append(interp_cnt)
        
    return interp_cnts