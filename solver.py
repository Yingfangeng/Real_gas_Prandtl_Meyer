import numpy as np
import CoolProp.CoolProp as CP
from real_gas_property import *
import pandas as pd
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

        max_iteration = 100

        iteration_no = 0

        integration_steps = 1000

        while not bracket_found and iteration_no < max_iteration:
            
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
                
                
                

            except ValueError as e:
                print(e)
                print('Cannot locate the lower bound of h_2, 2* the integration tolerance')
                
                integration_steps = integration_steps -1

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

                

            except ValueError as e:
                print(e)
                print('Value error occured, doubling the integration tolerance')
                                
                integration_steps = integration_steps -1

                continue

            iteration_no = iteration_no + 1
            
        if not solution_found:
            raise ValueError('Bi-section failed to find the solution')


        a_2 = PropsSI_General('A', 'H', h_2, 'S', s_0, fluid)
        V_2 = np.sqrt(2 * (h_0 - h_2))
        M_2 = V_2 / a_2
        
        print('T_2', CP.PropsSI('T', 'S', s_1, 'H', h_2, fluid))
    
    except Exception as e:
        print(f'Error occured for T {T_1} S {s_1}', e)
        M_2 = 999
        h_2 = 999

    result = {}
    result['M_2'] = M_2
    result['h_2'] = h_2
    result['integration_tolerance'] = integration_steps

    return result