import numpy as np
import CoolProp.CoolProp as CP
from real_gas_property import *
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from scipy.integrate import simpson


def prandtl_meyer_expansion(T_1, s_1, M_1, theta, tolerance, fluid='CO2'):

    h_1 = CP.PropsSI('H', 'S', s_1, 'T', T_1, fluid)
    s_0 = s_1
    a_1 = PropsSI_General('A', 'S', s_1, 'T', T_1, fluid)

    V_1 = M_1 * a_1
    h_0 = h_1 + 0.5 * V_1**2

    theta_target = np.deg2rad(theta)
    

    def integrand(h):

        a = PropsSI_General('A', 'H', h, 'S', s_0, fluid)
        V = np.sqrt(2 * (h_0 - h))
        M = V / a

        if M < 1:
            raise ValueError("Subsonic state encountered.")
        
        return -np.sqrt(M**2 - 1) / (2 * (h_0 - h))



    def calculate_residual(h_2, integration_steps):
        
        # result, _ = quad(integrand, h_1, h_2, epsrel=integration_tolerance)

        h_list = np.linspace(h_1, h_2, integration_steps)
        f_list = []
        
        for h in h_list:
            f_list.append(integrand(h))
        f_list = np.array(f_list)
        result = simpson(f_list, x=h_list)

        residual = result - theta_target
        
        return residual



    try: 
        step = 0.01 * (h_0 - h_1)
        
        h_2_lower = h_1
        h_2_upper = h_1
        
        bracket_found = False

        max_iteration = 50

        iteration_no = 0

        integration_steps = 10000

        while not bracket_found and iteration_no < max_iteration:
            
            # print('h_2 lower bound trial', h_2_lower_trial)

            # print('T', CP.PropsSI('T', 'H', h_2_lower_trial, 'S', s_0, fluid))

            try: 
                h_2_lower_trial = h_2_lower - step
                residual = calculate_residual(h_2_lower_trial, integration_steps)

                print(f'Residual of {iteration_no+1} iteration:', residual)
                            
                if residual >= 0:
                    bracket_found = True
                    print('Lower bound for h_2 found.')
                    h_2_lower = h_2_lower_trial
    
                else:
                    h_2_lower = h_2_lower_trial
                    h_2_upper = h_2_lower_trial
                

            except ValueError:
               
                integration_steps = integration_steps -1
                print('Cannot locate the lower bound of h_2, integration steps', integration_steps)

                continue

            iteration_no = iteration_no + 1
            

        if not bracket_found:
            raise RuntimeError("Could not bracket downstream enthalpy.")


        
        solution_found = False
        iteration_no = 0

        while not solution_found and iteration_no < max_iteration:

            try: 
                h_2 = (h_2_lower + h_2_upper) / 2

                residual = calculate_residual(h_2, integration_steps)

                print(f'Residual of {iteration_no+1} iteration:', residual)
                
                if abs(residual) < tolerance:
                    solution_found = True
                    print('Solution found')

                else: 
                    if residual > 0:
                        h_2_lower = h_2
                    else:
                        h_2_upper = h_2

                
            except ValueError:
                
                integration_steps = integration_steps -1
                print('Value error occured during root finding, integration steps', integration_steps)

                continue

            iteration_no = iteration_no + 1
            
        if not solution_found:
            raise ValueError('Bi-section failed to find the solution')


        a_2 = PropsSI_General('A', 'H', h_2, 'S', s_0, fluid)
        V_2 = np.sqrt(2 * (h_0 - h_2))
        M_2 = V_2 / a_2

    except Exception as e:
        print(f'Error occured for T {T_1} S {s_1}', e)
        M_2 = 999
        h_2 = 999

    result = {}
    result['M_2'] = M_2
    result['h_2'] = h_2
    result['integration_steps'] = integration_steps

    return result


if __name__ == '__main__':

    start_row = 1  # Header only: no points skipped. Use 124 to skip 123 points.

    cores = 18

    csv_path = 'real_gas_PM_result_bisection_parallel.csv'

    fluid = 'CO2'
    
    s_1_list   = np.linspace(1150, 1700, 32)
    T_1_list   = np.linspace(305, 320, 32)
    
    M_1   = 1.5
    
    theta = 5

    tolerance = 1e-5

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
            row['T_1'] = T_1
            row['s_1'] = s_1
            row['M_1'] = M_1
            row['theta'] = theta

            row.update(result)

            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

            print(T_1, s_1, 'done')
