import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import CoolProp.CoolProp as CP
from real_gas_property import *
import pandas as pd
import os


def prandtl_meyer_expansion(T_1, s_1, M_1, theta, tolerance, fluid='CO2'):

    h_1 = PropsSI_General('H', 'S', s_1, 'T', T_1, fluid)
    s_0 = s_1
    a_1 = PropsSI_General('A', 'S', s_1, 'T', T_1, fluid)

    V_1 = M_1 * a_1
    h_0 = h_1 + 0.5 * V_1**2

    theta_target = np.deg2rad(theta)


    def integrand(h):

        # Local Mach number along the integral

        a = PropsSI_General('A', 'H', h, 'S', s_0, fluid)
        V = np.sqrt(2 * (h_0 - h))
        M = V / a

        if M < 1:
            raise ValueError("Subsonic state encountered.")
        
        return np.sqrt(M**2 - 1) / (2 * (h_0 - h))


    def calculate_residual(h_2):
        
        result, _ = quad(integrand, h_2, h_1, epsrel=1e-7)
        residual = result - theta_target
        
        
        return residual





    try: 
        step = 0.01 * (h_0 - h_1)
        h_lower = h_1
        
        bracket_found = False

        for _ in range(1000):
            
            h_trial = h_lower - step

            print('h_new', h_trial)
            
            residual = calculate_residual(h_trial)
            print(residual)
            
            if residual >= 0:
                bracket_found = True
                break

            h_lower = h_trial


        if not bracket_found:
            raise RuntimeError("Could not bracket downstream enthalpy.")

        # h_2 = brentq(calculate_residual, h_trial, h_1, xtol = tolereance)


        for _ in range(1000):

            
            h_2 = (h_lower + h_upper) / 2
            print(h_2)
            residual = calculate_residual(h_2)

            print( residual)

            
            if abs(residual) < tolerance:
                break

            if residual > 0:
                h_lower = h_2
            else:
                h_upper = h_2

        else:
            raise RuntimeError("Bisection did not converge.")




        # Downstream properties
        P_2 = CP.PropsSI('P', 'H', h_2, 'S', s_0, fluid)
        T_2 = CP.PropsSI('T', 'H', h_2, 'S', s_0, fluid)
        a_2 = PropsSI_General('A', 'H', h_2, 'S', s_0, fluid)

        V_2 = np.sqrt(2 * (h_0 - h_2))
        M_2 = V_2 / a_2

        result = {}
        result['M_2'] = M_2
        result['h_2'] = h_2

    
    except ValueError as e:

        print('Cannot find a solution for this case', e)

        result = {}
        result['M_2'] = 999
        result['h_2'] = 999

    return result


if __name__ == '__main__':

    csv_path = 'real_gas_PM_result_test.csv'

    fluid = 'CO2'
    
    s_1_list   = np.linspace(1150, 1700, 32)
    T_1_list   = np.linspace(305, 320, 32)
    
    M_1   = 2
    
    theta = 10

    tolerance = 1e-6

    T_1 = 319.51612903225805
    s_1 = 1469.3548387096776

    # for T_1 in T_1_list:
    #     for s_1 in s_1_list:

    result = prandtl_meyer_expansion(T_1, s_1, M_1, theta, tolerance, fluid)

    row = {}
    row['T_1'] = T_1
    row['s_1'] = s_1
    row['M_1'] = M_1
    row['theta'] = theta

    row.update(result)

    # pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    print(T_1, s_1, 'done')