import numpy as np
import pandas as pd
import pandas as pd
import os
import CoolProp.CoolProp as CP
from real_gas_property import *

def model_validation(s_1, T_1, M_1, theta):

    fluid = 'CO2'
    Gamma_1 = CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'T', T_1, 'S', s_1, fluid)

    M_2 = M_1*(0.00075308991351283118*Gamma_1**2*(M_1 + theta)**2 + 0.8990340825864408)**2 + 0.29712762313654323
    #(M_1 - 1*(-0.15625110459724956))/((Gamma_1**2 + 3.696923639952799)*theta*(-0.005713609523777871) + 1.0932934875532947)
    

    return M_2


if __name__ == '__main__':

    mode = 2

    if mode == 1:
        M_1 = 1.1
        theta = 3
        in_csv_path = f'real_gas_PM_dataset/real_gas_PM_result_M_{M_1}_theta_{theta}.csv'
        out_csv_path = f'validation/model_CO2_validate_M_{M_1}_theta_{theta}.csv'

    elif mode == 2:
        in_csv_path = f'real_gas_PM_dataset/testing.csv'
        out_csv_path = f'validation/model_CO2_validate.csv'
        

    in_df = pd.read_csv(in_csv_path)
    in_df = in_df[in_df['h_2'] != 999]
    in_df = in_df[in_df['Q_2'] == -1]

    result = {}

    for idx, row in in_df.iterrows():
        result['s_1'] = row['s_1']
        result['T_1'] = row['T_1']
        result['M_1'] = row['M_1']
        result['theta'] = row['theta']
        result['M_2'] = row['M_2']
        M_2 = model_validation(row['s_1'], row['T_1'], row['M_1'], row['theta'])
        result['M_2_model'] = M_2
        
        pd.DataFrame([result]).to_csv(out_csv_path, mode='a', header=not os.path.exists(out_csv_path), index=False)