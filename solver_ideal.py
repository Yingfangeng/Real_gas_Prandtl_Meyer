import numpy as np
import CoolProp.CoolProp as CP
import pandas as pd
from real_gas_property import *
import os


def ideal_Prandtl_Meyer_function(T_1, s_1, M_1, theta, tolerance, fluid='CO2'):
    
    theta = np.deg2rad(theta)
    ns = CP.PropsSI('ISENTROPIC_EXPANSION_COEFFICIENT', 'S', s_1, 'T', T_1, fluid)

    if fluid == 'air':
        print('Fluid is air')
        ns = 1.4

    nu_1 = (((ns +1)/ (ns-1))**0.5) * np.atan(((ns-1)/(ns+1)* (M_1**2 - 1))**0.5) - np.atan((M_1**2-1)**0.5)

    M_2_lower = 1
    M_2_upper = 5

    max_iteration = 1000
    iteration_count = 0
    residual = 1

    while abs(residual) > tolerance and iteration_count < max_iteration:
        
        M_2 = (M_2_lower + M_2_upper)/2
        print(residual, M_2)
        nu_2 = (((ns +1)/ (ns-1))**0.5) * np.atan((((ns-1)/(ns+1))* (M_2**2 - 1))**0.5) - np.atan((M_2**2-1)**0.5)

        residual = nu_2 - nu_1 - theta

        if residual > 0:
            M_2_upper = M_2
        else:
            M_2_lower = M_2

        iteration_count = iteration_count + 1

    return M_2

if __name__ ==  '__main__':
    

    fluid = 'air'
    
    s_1_list   = np.linspace(1150, 1700, 32)
    T_1_list   = np.linspace(330, 350, 32)
    
    M_1   = 1.1
    theta = 3

    tolerance = 1e-6
    csv_path = f'validation/ideal_{fluid}_validate_M_{M_1}_theta_{theta}.csv'

    for s_1 in s_1_list:
        for T_1 in T_1_list:
            M_2 = ideal_Prandtl_Meyer_function(T_1, s_1, M_1, theta, tolerance, fluid)
            
            row = {}
            row['T_1'] = T_1
            row['s_1'] = s_1
            row['M_1'] = M_1
            row['theta'] = theta
            row['M_2'] = M_2

            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    

