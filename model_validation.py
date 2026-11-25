import numpy as np
import pandas as pd
import pandas as pd
import os
import CoolProp.CoolProp as CP
from real_gas_property import *

def model_validation(s_1, T_1, M_1, theta):

    fluid = 'CO2'
    Gamma_1 = CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'T', T_1, 'S', s_1, fluid)

    M_2 = (M_1 - 1*(-0.15625110459724956))/((Gamma_1**2 + 3.696923639952799)*theta*(-0.005713609523777871) + 1.0932934875532947)
    

    return M_2


if __name__ == '__main__':

    s_1_list   = np.linspace(1150, 1700, 32)
    T_1_list   = np.linspace(330, 350, 32)
    
    M_1   = 1.1
    theta = 3

    csv_path = f'validation/model_CO2_validate_M_{M_1}_theta_{theta}.csv'

    for s_1 in s_1_list:
        for T_1 in T_1_list:
            M_2 = model_validation(s_1, T_1, M_1, theta)
            
            row = {}
            row['T_1'] = T_1
            row['s_1'] = s_1
            row['M_1'] = M_1
            row['theta'] = theta
            row['M_2'] = M_2

            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)