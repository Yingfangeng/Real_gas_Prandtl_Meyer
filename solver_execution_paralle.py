from solver import prandtl_meyer_expansion
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import os
import CoolProp.CoolProp as CP

if __name__ == '__main__':

    start_row = 1       # 1 for start from scratch

    cores = 18

    fluid = 'CO2'
    
    s_1_list   = np.linspace(1150, 1700, 32)
    T_1_list   = np.linspace(330, 350, 32)
    
    M_1   = 1.2
    theta = 2

    tolerance = 1e-5

    csv_path = f'real_gas_PM_dataset/real_gas_PM_result_M_{M_1}_theta_{theta}_time_test.csv'


    with ProcessPoolExecutor(max_workers=cores) as executor:
        tasks = [
            (T_1, s_1, executor.submit(
                prandtl_meyer_expansion, T_1, s_1, M_1, theta, tolerance, fluid))
            for point_number, (T_1, s_1) in enumerate(
                ((T_1, s_1) for T_1 in T_1_list for s_1 in s_1_list), start=1)
            if point_number >= start_row
        ]

        for T_1, s_1, future in tasks:

            result = future.result()

            row = {}
            row['theta'] = theta
            row['M_1'] = M_1
            row['T_1'] = T_1
            row['s_1'] = s_1
            row['Gamma_1'] = CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'T', T_1, 'S', s_1, fluid)
            row['ns_1'] = CP.PropsSI('ISENTROPIC_EXPANSION_COEFFICIENT', 'T', T_1, 'S', s_1, fluid)
            row['rho_1'] = CP.PropsSI('D', 'T', T_1, 'S', s_1, fluid)
            row['P_1'] = CP.PropsSI('P', 'T', T_1, 'S', s_1, fluid)
            row['c_1'] = CP.PropsSI('A', 'T', T_1, 'S', s_1, fluid)
            row['Z_1'] = CP.PropsSI('Z', 'T', T_1, 'S', s_1, fluid)
            row['cp_1'] = CP.PropsSI('CPMASS', 'T', T_1, 'S', s_1, fluid)
            row['cv_1'] = CP.PropsSI('CVMASS', 'T', T_1, 'S', s_1, fluid)
            row['mu_1'] = CP.PropsSI('V', 'T', T_1, 'S', s_1, fluid)

            
            row.update(result)

            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            print(T_1, s_1, 'done')