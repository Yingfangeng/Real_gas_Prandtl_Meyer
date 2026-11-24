import numpy as np
import CoolProp.CoolProp as CP
from real_gas_property import *


def ideal_Prandtl_Meyer_function(T_1, s_1, M_1, theta, tolerance, fluid='CO2'):
    
    theta = np.deg2rad(theta)
    ns = CP.PropsSI('ISENTROPIC_EXPANSION_COEFFICIENT', 'S', s_1, 'T', T_1, fluid)

    nu_1 = (((ns +1)/ (ns-1))**0.5) * np.atan(((ns-1)/(ns+1)* (M_1**2 - 1))**0.5) - np.atan((M_1**2-1)**0.5)

    M_2_lower = 1
    M_2_upper = 5

    max_iteration = 1000
    iteration_count = 0
    residual = 1

    while abs(residual) > tolerance and iteration_count < max_iteration:
        
        M_2 = (M_2_lower + M_2_upper)/2
        print(residual, M_2)
        nu_2 = (((ns +1)/ (ns-1))**0.5) * np.atan(((ns-1)/(ns+1)* (M_2**2 - 1))**0.5) - np.atan((M_2**2-1)**0.5)

        residual = nu_2 - nu_1 - theta

        if residual > 0:
            M_2_upper = M_2
        else:
            M_2_lower = M_2

        iteration_count = iteration_count + 1

    print(M_2)

if __name__ ==  '__main__':

    ideal_Prandtl_Meyer_function(330, 1150, 1.1, 4, 1e-6, fluid='CO2')