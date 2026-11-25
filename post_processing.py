from pathlib import Path

import CoolProp.CoolProp as CP
import pandas as pd


dataset_path = Path(__file__).parent / 'real_gas_PM_dataset'

for csv_path in sorted(dataset_path.glob('*.csv')):
    data = pd.read_csv(csv_path)
    qualities = []
    ns_1 = []
    Gamma_1 = []
    P_1 = []
    rho_1 = []
    c_1 = []
    Z_1 = []

    

    for index, row in data.iterrows():
        T_1 = row['T_1']
        s_1 = row['s_1']
        ns_1.append(CP.PropsSI('ISENTROPIC_EXPANSION_COEFFICIENT', 'T', T_1, 'S', s_1, 'CO2'))
        Gamma_1.append(CP.PropsSI('FUNDAMENTAL_DERIVATIVE_OF_GAS_DYNAMICS', 'T', T_1, 'S', s_1, 'CO2'))
        P_1.append(CP.PropsSI('P', 'T', T_1, 'S', s_1, 'CO2'))
        rho_1.append(CP.PropsSI('D', 'T', T_1, 'S', s_1, 'CO2'))
        c_1.append(CP.PropsSI('A', 'T', T_1, 'S', s_1, 'CO2'))
        Z_1.append(CP.PropsSI('Z', 'T', T_1, 'S', s_1, 'CO2'))


        if row['h_2'] == 999:
            Q_2 = 999
            
        else:
            Q_2 = CP.PropsSI('Q', 'S', row['s_1'], 'H', row['h_2'], 'CO2')
        qualities.append(Q_2)

    data['Q_2'] = qualities
    data['ns_1'] = ns_1
    data['Gamma_1'] = Gamma_1
    data['P_1'] = P_1
    data['rho_1'] = rho_1
    data['c_1'] = c_1
    data['Z_1'] = Z_1
    data.to_csv(csv_path, index=False)
