from solver import prandtl_meyer_expansion
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':

    fluid = 'CO2'

    M_1_list   = np.linspace(1.01, 2.2, 10)
    
    theta = 5
    tolerance = 1e-5

    T_1 = 330
    s_1 = 1700
    
    a_1 = CP.PropsSI('A', 'S', s_1, 'T', T_1, fluid)
    h_1 = CP.PropsSI('H', 'S', s_1, 'T', T_1, fluid)
    P_1 = CP.PropsSI('P', 'S', s_1, 'T', T_1, fluid)

    
    csv_path = f'real_gas_PM_dataset/real_gas_PM_result_for_CFD.csv'

    for M_1 in M_1_list:
        result = prandtl_meyer_expansion(T_1, s_1, M_1, theta, tolerance, fluid)

        V_1 = M_1*a_1
        h_0 = h_1 + V_1**2/2
        P_0 = CP.PropsSI('P', 'S', s_1, 'H', h_0, fluid)
        T_0 = CP.PropsSI('T', 'S', s_1, 'H', h_0, fluid)
        P_2 = CP.PropsSI('P', 'S', s_1, 'H', result['h_2'], fluid)
        row = {}
        row['theta'] = theta
        row['T_1'] = T_1
        row['s_1'] = s_1
        row['M_1'] = M_1
        row['P_1'] = P_1
        row['T_0'] = T_0
        row['P_0'] = P_0
        row['P_2'] = P_2
        
        row.update(result)

        pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    
    print(result)