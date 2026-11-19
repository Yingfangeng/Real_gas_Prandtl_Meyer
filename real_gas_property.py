import numpy as np
from CoolProp.CoolProp import PropsSI

def critical_point_values(fluid):
    P_c   = PropsSI('PCRIT', fluid)
    T_c   = PropsSI('TCRIT', fluid)
    D_c   = PropsSI('D', 'P', P_c, 'T', T_c, fluid)
    S_c   = PropsSI('S', 'P', P_c, 'T', T_c, fluid)
    U_c   = PropsSI('U', 'P', P_c, 'T', T_c, fluid)
    H_c   = PropsSI('H', 'P', P_c, 'T', T_c, fluid)
    c_c = PropsSI('speed_of_sound', 'P', P_c, 'T', T_c, fluid)

    print('================================================')
    print('Critical Pressure        : %.5f Pa'%(P_c))
    print('Critical Temperature     : %.5f K'%(T_c))
    print('Critical Entropy         : %.5f J/kg/K'%(S_c))
    print('Critical Density         : %.5f kg/m^3'%(D_c))
    print('Critical Internal Energy : %.5f J/kg'%(U_c))
    print('Critical Enthalpy        : %.5f J/kg'%(H_c))
    print('Critical Speed of Sound  : %.5f m/s'%(c_c))
    print('================================================')
    return

def point_calculation(Prop_o_name, Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid):
    Units = {"P": "Pa",
             "T": "K",
             "S": "J/kg/K",
             "D": "kg/m^3",
             "H": "J/kg",
             "U": "J/kg",
             "Q": "",
             "A": "m/s",
             "speed_of_sound": "m/s",
             "C": "J/kg/K",
             "L": "W/m/K",
             "conductivity": "W/m/K",
             "fundamental_derivative_of_gas_dynamics": "",
             "isentropic_expansion_coefficient": "",
             "Z": "",
             "V": "Pa s",}

    Names = {"P": "pressure",
             "T": "temperature",
             "S": "entropy",
             "D": "density",
             "H": "enthalpy",
             "U": "internal energy",
             "Q": "vapour quality",
             "A": "speed of sound",
             "speed_of_sound": "speed of sound",
             "C": "mass specific cp",
             "L": "heat conductivity",
             "conductivity": "heat conductivity",
             "fundamental_derivative_of_gas_dynamics": "fundamental_derivative_of_gas_dynamics",
             "isentropic_expansion_coefficient": "isentropic_expansion_coefficient",
             "Z": "compressibility",
             "V": "viscosity",}

    Prop_o_val = PropsSI_General(Prop_o_name, Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid)
    print('The %s (%s) of %s at %s %s = %.4f %s and %s %s = %.4f %s is %.4f %s.'%(Names[Prop_o_name], Prop_o_name, fluid,
                                Names[Prop_a_name], Prop_a_name, Prop_a_val, Units[Prop_a_name],
                                Names[Prop_b_name], Prop_b_name, Prop_b_val, Units[Prop_b_name],
                                Prop_o_val, Units[Prop_o_name]))
    return

def state_calculation_real_full(Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid_name):
    T = PropsSI_General('T', Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid_name)
    S = PropsSI_General('S', Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid_name)
    state = {"T": T,
             "S": S,
             "P": PropsSI_General('P', 'T', T, 'S', S, fluid_name),
             "D": PropsSI_General('D', 'T', T, 'S', S, fluid_name),
             "H": PropsSI_General('H', 'T', T, 'S', S, fluid_name),
             "U": PropsSI_General('U', 'T', T, 'S', S, fluid_name),
             "Q": PropsSI_General('Q', 'T', T, 'S', S, fluid_name),
             "C": PropsSI_General('C', 'T', T, 'S', S, fluid_name), # Cp
             "L": PropsSI_General('L', 'T', T, 'S', S, fluid_name),
             "Z": PropsSI_General('Z', 'T', T, 'S', S, fluid_name),
             "V": PropsSI_General('V', 'T', T, 'S', S, fluid_name),
             "A": PropsSI_General('speed_of_sound', 'T', T, 'S', S, fluid_name),
             "conductivity"                           : PropsSI_General('conductivity', 'T', T, 'S', S, fluid_name),
             "speed_of_sound"                         : PropsSI_General('speed_of_sound', 'T', T, 'S', S, fluid_name),
             "isentropic_expansion_coefficient"       : PropsSI_General('isentropic_expansion_coefficient', 'T', T, 'S', S, fluid_name),
             "fundamental_derivative_of_gas_dynamics" : PropsSI_General('fundamental_derivative_of_gas_dynamics', 'T', T, 'S', S, fluid_name),
             }
    return state

def state_calculation_real(result_name_list, Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid_name):
    T = PropsSI_General('T', Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid_name)
    S = PropsSI_General('S', Prop_a_name, Prop_a_val, Prop_b_name, Prop_b_val, fluid_name)

    state = {}
    for name in result_name_list:
        state[name] = PropsSI_General(name, 'T', T, 'S', S, fluid_name)

    return state

# partial derivative calculation using central scheme
def partial_y_x(y_name, x_name, x, dx, z_name, z, fluid):

    result = (PropsSI(y_name, x_name, x+dx, z_name, z, fluid) - 
              PropsSI(y_name, x_name, x-dx, z_name, z, fluid)) / (2*dx)

    return result

# partial derivatives using coolprop and only goes downwards (use for inside saturation line)
def partial_y_x_lower(y_name, x_name, x, dx, z_name, z, fluid):

    result = (PropsSI(y_name, x_name, x, z_name, z, fluid) - 
                PropsSI(y_name, x_name, x-dx, z_name, z, fluid)) / dx

    return result

def OpenCFD_EOS_validation(D_nondim, T_nondim, u, v, w, fluid = 'CarbonDioxide',
                           T_ref = 310, D_ref = 250, U_ref = 461.69, Re = 635000):

    T = T_ref * T_nondim
    D = D_ref * D_nondim

    P = PropsSI('P', 'T', T, 'D', D, fluid)
    E = PropsSI('U', 'T', T, 'D', D, fluid)
    H = PropsSI('H', 'T', T, 'D', D, fluid)
    c = PropsSI('A', 'T', T, 'D', D, fluid)
    mu = PropsSI('V', 'T', T, 'D', D, fluid)
    k = PropsSI('L', 'T', T, 'D', D, fluid)

    ref_miu = mu # mu should be ref_miu, using mu for LUT approximation

    P_nondim = P/(D_ref*U_ref**2)
    E_nondim = E/U_ref**2
    H_nondim = (H/U_ref**2 + 0.5*(u**2+v**2+w**2)) # this is stagnation enthalpy 
    #          E/U_ref**2 + 0.5*(u**2+v**2+w**2) + P/(D*U_ref**2)
    c_nondim = c/U_ref
    mu_nondim = mu * (1/(Re*ref_miu)) 
    k_nondim = k * T_ref/(Re*U_ref**2*ref_miu)

    print('Non-dimensional P in OpenCFD:', P_nondim)
    print('Non-dimensional E in OpenCFD:', E_nondim)
    print('Non-dimensional H in OpenCFD:', H_nondim)
    print('Non-dimensional c in OpenCFD:', c_nondim)
    print('Non-dimensional mu in OpenCFD:', mu_nondim)
    print('Non-dimensional k in OpenCFD:', k_nondim)

    # P_nondim_cfd = 0.140743
    # E_nondim_cfd = 1.77659135692429504
    # H_nondim_cfd = 2.41638263829852962 # this is stagnation enthalpy
    # c_nondim_cfd = 0.444921603519898401
    # mu_nondim_cfd = 1.57739998485340038e-06
    # k_nondim_cfd = 4.22489983975538053e-06
    
    return

# ===========================================================================================================================
# Fluid property retrivel

def PropsSI_General(propname, input_name1, input1, input_name2, input2, fluid_name = 'CarbonDioxide'):
    if propname == 'fundamental_derivative_of_gas_dynamics':
        T1 = PropsSI_Custom('T', input_name1, input1, input_name2, input2, fluid_name)
        S1 = PropsSI_Custom('S', input_name1, input1, input_name2, input2, fluid_name)
        prop = fundamental_derivative_principle(T1, S1, 0.0001, fluid_name)
        return prop
    elif propname == 'isentropic_expansion_coefficient':
        prop = isentropic_exponents_pv(input_name1, input1, input_name2, input2, fluid_name)
        return prop
    elif propname == 'gamma':
        prop = PropsSI_Custom('CPMASS', input_name1, input1, input_name2, input2, fluid_name) / \
               PropsSI_Custom('CVMASS', input_name1, input1, input_name2, input2, fluid_name)
        return prop
    else:
        try:
            prop = PropsSI(propname, input_name1, input1, input_name2, input2, fluid_name)
            return prop
        except:
            
            T_crit = PropsSI('Tcrit', fluid_name)
            
            T1 = PropsSI_Custom('T', input_name1, input1, input_name2, input2, fluid_name)
            S1 = PropsSI_Custom('S', input_name1, input1, input_name2, input2, fluid_name)

            D1 = PropsSI_Custom('D', 'T', T1, 'S', S1, fluid_name)
            
            # check for temperature lower than critical temperature
            if T1 < T_crit:
                
                # static density inside the wet range (point inside domain)
                D_liquid = PropsSI_Custom('D', 'Q', 0, 'T', T1, fluid_name)
                D_vapour = PropsSI_Custom('D', 'Q', 1, 'T', T1, fluid_name)
                if (D1 >= D_vapour) and (D1 <= D_liquid):

                    # fraction determination
                    if propname == 'D' or propname == 'speed_of_sound' or propname=='A':
                        frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) # volume fraction
                    else:
                        frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) * (D_vapour/D1) # mass fraction

                    # property retrieval: partial derivatives requires special treatment
                    if propname == 'speed_of_sound' or propname=='A':
                        
                        # using partial chi
                        prop = equilibrium_wet_vapour_speed_of_sound(frac, D1, T1, S1, 
                                                                    D_liquid, D_vapour, 
                                                                    fluid_name = fluid_name)

                    else:
                        
                        if '|' in propname:
                            print('Warning: direct partial derivative retrieval in two-phase region may contain error.')
                        
                        # calculation using mass/volume fraction
                        prop_liquid = PropsSI_Custom(propname, 'Q', 0, 'T', T1, fluid_name)
                        prop_vapour = PropsSI_Custom(propname, 'Q', 1, 'T', T1, fluid_name)
                        prop        = (1-frac) * prop_liquid + frac * prop_vapour
                        # print ('Property retrieved using equilibrium mixture assuption for T = %.2f and S = %.2f.'%(T1, S1))
                
                else:

                    # out side saturation line (liquid or gas phase)
                    prop = PropsSI_Custom(propname, 'T', T1, 'S', S1, fluid_name)
                    # print (prop)
            
            else:

                # above critical temperature (above critical point)
                prop = PropsSI_Custom(propname, 'T', T1, 'S', S1, fluid_name)
                # print (prop)
            
            return prop

# PropsSi calculation that overcomes the gap in CoolProp by interpolation
def PropsSI_Custom(propname, input_name1, input1, input_name2, input2, fluid_name = 'CarbonDioxide'):
    try:
        prop = PropsSI(propname, input_name1, input1, input_name2, input2, fluid_name)
    except:
        if np.isnan(input1) or np.isnan(input2) or np.isposinf(input1) or np.isposinf(input2) or np.isneginf(input1) or np.isneginf(input2):
            raise ValueError('Invalid Input value') from None
        elif input_name1=='D' or input_name2=='D':
            # print('Using interpolation for %s = %.2f, %s = %.2f'%(input_name1, input1, input_name2, input2))
            if input_name1=='D':
                prop = PropsSI_D_intp(propname, input1, input_name2, input2, fluid_name, delta = 0.001)
            else:
                prop = PropsSI_D_intp(propname, input2, input_name1, input1, fluid_name, delta = 0.001)
        # elif (input_name1=='P' and input_name2!='T') or (input_name2=='P' and input_name1!='T'):
        #     if input_name1=='P':
        #         prop = PropsSI_P_intp(propname, input1, input_name2, input2, fluid_name, delta = 0.001)
        #     else:
        #         prop = PropsSI_P_intp(propname, input2, input_name1, input1, fluid_name, delta = 0.001)
        else:
            raise ValueError('No calculation available for %s = %.2f, %s = %.2f'%(input_name1, input1, input_name2, input2)) from None
        
    return prop

# stagnation condition from TSM using LUT
def stagnation_condition(T_static, S_static, M, fluid_name = 'CarbonDioxide'):
    
    c  = wet_vapour_properties_SI('speed_of_sound', 'T', T_static, 'S', S_static, fluid_name)
    V  = M * c
    H0 = PropsSI('H', 'T', T_static, 'S', S_static, fluid_name) + 0.5 * V**2
    P0 = PropsSI('P', 'H', H0      , 'S', S_static, fluid_name)
    T0 = PropsSI('T', 'H', H0      , 'S', S_static, fluid_name)
    
    return P0, T0

# stagnation condition from XYM using LUT
def stagnation_condition_SI(props_list, input_name1, input1, input_name2, input2, M, fluid_name = 'CarbonDioxide'):
    
    T_static = PropsSI('T', input_name1, input1, input_name2, input2, fluid_name)
    S_static = PropsSI('S', input_name1, input1, input_name2, input2, fluid_name)

    c  = PropsSI_General('speed_of_sound', 'T', T_static, 'S', S_static, fluid_name)
    V  = M * c
    H0 = PropsSI('H', 'T', T_static, 'S', S_static, fluid_name) + 0.5 * V**2
    
    # returning results
    results = []
    for name in props_list:
        results.append(PropsSI_General(name, 'H', H0, 'S', S_static, fluid_name))

    return results

# stagnation condition from XYM using LUT
def stagnation_condition_state(result_name_list, input_name1, input1, input_name2, input2, M, fluid_name = 'CarbonDioxide'):
    
    T_static = PropsSI('T', input_name1, input1, input_name2, input2, fluid_name)
    S_static = PropsSI('S', input_name1, input1, input_name2, input2, fluid_name)

    c  = PropsSI_General('speed_of_sound', 'T', T_static, 'S', S_static, fluid_name)
    V  = M * c
    H0 = PropsSI('H', 'T', T_static, 'S', S_static, fluid_name) + 0.5 * V**2
    
    # returning results
    state = {}
    for name in result_name_list:
        state[name] = PropsSI_General(name, 'H', H0, 'S', S_static, fluid_name)
        
    return state

# stagnation condition from X0,Y0,M using LUT
def static_condition_state(result_name_list, prop_01_name, prop_01, prop_02_name, prop_02, M, 
                           fluid_name = 'CarbonDioxide', T_step_in = -5, tol=1E-5):

    # Total Condition
    S0      = PropsSI('S', prop_01_name, prop_01, prop_02_name, prop_02, fluid_name)
    H0      = PropsSI('H', prop_01_name, prop_01, prop_02_name, prop_02, fluid_name)
    T0      = PropsSI('T', prop_01_name, prop_01, prop_02_name, prop_02, fluid_name)

    # initial estimation
    H       = H0
    T       = T0

    T_step  = T_step_in
    M_prev  = 0
    res_prev= M_prev - M
    counter = 0
    while np.abs(T_step) > tol:

        T = T + T_step
        H = PropsSI('H', 'T', T, 'S', S0, fluid_name)
        c = PropsSI_General('speed_of_sound', 'T', T, 'S', S0, fluid_name)
        v = (2 * (H0-H))**0.5

        M_new = v / c

        res_new = M_new - M
        if np.sign(res_prev) != np.sign(res_new):
            T_step = - 0.5 * T_step
        res_prev = res_new

        if counter >= 10000:
            print("isentropic_properties_at_given_mach_number Failed to converge.")
            break
        counter = counter + 1

    # # returning results
    # results = []
    # for name in result_name_list:
    #     results.append(PropsSI_General(name, 'H', H, 'S', S0, fluid_name))
    # return results

    state = {}
    for name in result_name_list:
        state[name] = PropsSI_General(name, 'H', H, 'S', S0, fluid_name)
        
    return state

# partial derivative holding constant S, square of speed of sound
def dP_drho(T1, S1, dT, fluid):

        D_lower = PropsSI('D', 'T', T1-dT, 'S', S1, fluid)
        D_upper = PropsSI('D', 'T', T1   , 'S', S1, fluid)

        P_lower = PropsSI('P', 'T', T1-dT, 'S', S1, fluid)
        P_upper = PropsSI('P', 'T', T1   , 'S', S1, fluid)
        
        result  = (P_upper-P_lower) / (D_upper-D_lower)

        return result

# calculating dc/drho on isentrope (constant s)
def dc_drho(T1, S1, dT, fluid):
        
        D_lower = PropsSI('D', 'T', T1-dT, 'S', S1, fluid)
        D_upper = PropsSI('D', 'T', T1   , 'S', S1, fluid)

        c_lower = (dP_drho(T1-dT, S1, dT, fluid))**0.5
        c_upper = (dP_drho(T1   , S1, dT, fluid))**0.5

        # c_lower = PropsSI_General('speed_of_sound', 'T',T1-dT,'S', S1, fluid)
        # c_upper = PropsSI_General('speed_of_sound', 'T',T1,'S', S1, fluid)
        
        result = (c_upper-c_lower) / (D_upper-D_lower)

        return result

# n_s calculation (same result from coolprop) using gradient definition
def isentropic_exponents_pv(x_name1, x1, x_name2, x2, fluid):

    if x_name1 == 'P':
        P = x1
    elif x_name2 == 'P':
        P = x2
    else:
        P = PropsSI('P', x_name1, x1, x_name2, x2, fluid)

    if x_name1 == 'D':
        D = x1
    elif x_name2 == 'D':
        D = x2
    else:
        D = PropsSI('D', x_name1, x1, x_name2, x2, fluid)

    # Using pv^ns = const definition
    # n_s = - PropsSI('d(P)/d(D)|S', x_name1, x1, x_name2, x2, fluid) * (-D**2) / (D*P)

    # Using speed of sound definition
    n_s = wet_vapour_properties_SI('speed_of_sound',x_name1, x1, x_name2, x2, fluid)**2  * D / P

    return n_s

# calculating isentropic expansion coefficient from 1st principle using speed of sound definition
def isentropic_exponents_pv_principle(T1, S1, fluid_name = 'CarbonDioxide'):
    D1 = PropsSI('D', 'T', T1, 'S', S1, fluid_name)
    P1 = PropsSI('P', 'T', T1, 'S', S1, fluid_name)
    ns = wet_vapour_properties_TS(T1, S1, 'speed_of_sound', fluid_name)**2  * D1 / P1
    return ns

# m_s calculation (TP^ms = const assumption)
def isentropic_exponents_Tp(x_name1, x1, x_name2, x2, fluid):

    if x_name1 == 'P':
        P = x1
    elif x_name2 == 'P':
        P = x2
    else:
        P = PropsSI('P', x_name1, x1, x_name2, x2, fluid)

    if x_name1 == 'T':
        T = x1
    elif x_name2 == 'T':
        T = x2
    else:
        T = PropsSI('T', x_name1, x1, x_name2, x2, fluid)

    m_s = (P / T) * PropsSI('d(T)/d(P)|S', x_name1, x1, x_name2, x2, fluid)

    return m_s

# Gamma calculation (same as coolprop)
def fundamental_derivative(x_name1, x1, x_name2, x2, fluid):
    D = PropsSI('D', x_name1, x1, x_name2, x2, fluid)
    c = PropsSI('speed_of_sound', x_name1, x1, x_name2, x2, fluid)
    G = 1 + D*PropsSI('d(speed_of_sound)/d(D)|S', x_name1, x1, x_name2, x2, fluid) / c
    return G

# calculating fundamental derivative from 1st principle
def fundamental_derivative_principle(T1, S1, dT, fluid_name):

    D = PropsSI('D', 'T', T1, 'S', S1, fluid_name)
    c = PropsSI_General('speed_of_sound', 'T',T1,'S', S1, fluid_name)
    G = 1 + D*dc_drho(T1, S1, dT, fluid_name) / c

    return G

def equivalent_ideal_gas_property(prop_1_name, prop_1, prop_2_name, prop_2, fluid_name):
    R_gas = PropsSI('gas_constant', prop_1_name, prop_1, prop_2_name, prop_2, fluid_name) / \
            PropsSI('M'           , prop_1_name, prop_1, prop_2_name, prop_2, fluid_name)
    Gamma = PropsSI_General('isentropic_expansion_coefficient', prop_1_name, prop_1, prop_2_name, prop_2, fluid_name)
    Cp    = R_gas / (1 - 1/Gamma) 
    # Cp    = PropsSI_General('C', prop_1_name, prop_1, prop_2_name, prop_2, fluid_name)
    return R_gas, Gamma, Cp

# ===========================================================================================================================
# Two-phase flows

# retrieves fluid properties with wet vapour region considered
def wet_vapour_properties_TS(T1, S1, propname, fluid_name = 'CarbonDioxide'):
    T_crit = PropsSI('Tcrit', fluid_name)
    D1     = PropsSI('D', 'T', T1, 'S', S1, fluid_name)
    
    # check for temperature lower than critical temperature
    if T1 < T_crit:

        # static density inside the wet range (point inside domain)
        D_liquid = PropsSI('D', 'Q', 0, 'T', T1, fluid_name)
        D_vapour = PropsSI('D', 'Q', 1, 'T', T1, fluid_name)
        if (D1 >= D_vapour) and (D1 <= D_liquid):

            # fraction determination
            if propname == 'D' or propname == 'speed_of_sound':
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) # volume fraction
            else:
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) * (D_vapour/D1) # mass fraction

            # property retrieval: partial derivatives requires special treatment
            if propname == 'speed_of_sound':

                # using partial chi
                prop = equilibrium_wet_vapour_speed_of_sound(frac, D1, T1, S1, 
                                                             D_liquid, D_vapour, 
                                                             fluid_name = fluid_name)

                # using full relaxation model
                # prop = equilibrium_wet_vapour_speed_of_sound_LUND(frac, D1, T1, S1, 
                #                                                   D_liquid, D_vapour, 
                #                                                   equilibrium_model = 3,
                #                                                   fluid_name = fluid_name)

                # direct partial derivative calculation
                # prop = dP_drho(Y[i,j], X[i,j], 0.0000001, fluid)

            else:
                
                if '|' in propname:
                    print('Warning: direct partial derivative retrieval in two-phase region may contain error.')
                
                # calculation using mass/volume fraction
                prop_liquid = PropsSI(propname, 'Q', 0, 'T', T1, fluid_name)
                prop_vapour = PropsSI(propname, 'Q', 1, 'T', T1, fluid_name)
                prop        = (1-frac) * prop_liquid + frac * prop_vapour
                # print ('Property retrieved using equilibrium mixture assuption for T = %.2f and S = %.2f.'%(T1, S1))
        
        else:

            # out side saturation line (liquid or gas phase)
            prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
            # print (prop)
    
    else:

        # above critical temperature (above critical point)
        prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
        # print (prop)
    
    return prop

def wet_vapour_properties_PD(P1, D1, propname, fluid_name = 'CarbonDioxide'):
    T_crit = PropsSI('Tcrit', fluid_name)
    T1     = PropsSI('T', 'P', P1, 'D', D1, fluid_name)
    S1     = PropsSI('S', 'P', P1, 'D', D1, fluid_name)
    
    # check for temperature lower than critical temperature
    if T1 < T_crit:

        # static density inside the wet range (point inside domain)
        D_liquid = PropsSI('D', 'Q', 0, 'T', T1, fluid_name)
        D_vapour = PropsSI('D', 'Q', 1, 'T', T1, fluid_name)
        if (D1 >= D_vapour) and (D1 <= D_liquid):

            # fraction determination
            if propname == 'D' or propname == 'speed_of_sound':
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) # volume fraction
            else:
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) * (D_vapour/D1) # mass fraction

            # property retrieval: partial derivatives requires special treatment
            if propname == 'speed_of_sound':

                # using partial chi
                prop = equilibrium_wet_vapour_speed_of_sound(frac, D1, T1, S1, 
                                                             D_liquid, D_vapour, 
                                                             fluid_name = fluid_name)

                # using full relaxation model
                # prop = equilibrium_wet_vapour_speed_of_sound_LUND(frac, D1, T1, S1, 
                #                                                   D_liquid, D_vapour, 
                #                                                   equilibrium_model = 3,
                #                                                   fluid_name = fluid_name)

                # direct partial derivative calculation
                # prop = dP_drho(Y[i,j], X[i,j], 0.0000001, fluid)

            else:
                
                if '|' in propname:
                    print('Warning: direct partial derivative retrieval in two-phase region may contain error.')
                
                # calculation using mass/volume fraction
                prop_liquid = PropsSI(propname, 'Q', 0, 'T', T1, fluid_name)
                prop_vapour = PropsSI(propname, 'Q', 1, 'T', T1, fluid_name)
                prop        = (1-frac) * prop_liquid + frac * prop_vapour
                # print ('Property retrieved using equilibrium mixture assuption for T = %.2f and S = %.2f.'%(T1, S1))
        
        else:

            # out side saturation line (liquid or gas phase)
            prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
            # print (prop)
    
    else:

        # above critical temperature (above critical point)
        prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
        # print (prop)
    
    return prop

def wet_vapour_properties_DU(D1, U1, propname, fluid_name = 'CarbonDioxide'):
    T_crit = PropsSI('Tcrit', fluid_name)
    # T1     = PropsSI('T', 'P', P1, 'D', D1, fluid_name)
    # S1     = PropsSI('S', 'P', P1, 'D', D1, fluid_name)
    T1     = PropsSI_DU('T', D1, U1, fluid_name, delta=0.001)
    S1     = PropsSI_DU('S', D1, U1, fluid_name, delta=0.001)
    # check for temperature lower than critical temperature
    if T1 < T_crit:

        # static density inside the wet range (point inside domain)
        D_liquid = PropsSI('D', 'Q', 0, 'T', T1, fluid_name)
        D_vapour = PropsSI('D', 'Q', 1, 'T', T1, fluid_name)
        if (D1 >= D_vapour) and (D1 <= D_liquid):

            # fraction determination
            if propname == 'D' or propname == 'speed_of_sound':
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) # volume fraction
            else:
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) * (D_vapour/D1) # mass fraction

            # property retrieval: partial derivatives requires special treatment
            if propname == 'speed_of_sound':

                # using partial chi
                prop = equilibrium_wet_vapour_speed_of_sound(frac, D1, T1, S1, 
                                                             D_liquid, D_vapour, 
                                                             fluid_name = fluid_name)

                # using full relaxation model
                # prop = equilibrium_wet_vapour_speed_of_sound_LUND(frac, D1, T1, S1, 
                #                                                   D_liquid, D_vapour, 
                #                                                   equilibrium_model = 3,
                #                                                   fluid_name = fluid_name)

                # direct partial derivative calculation
                # prop = dP_drho(Y[i,j], X[i,j], 0.0000001, fluid)

            else:
                
                if '|' in propname:
                    print('Warning: direct partial derivative retrieval in two-phase region may contain error.')
                
                # calculation using mass/volume fraction
                prop_liquid = PropsSI(propname, 'Q', 0, 'T', T1, fluid_name)
                prop_vapour = PropsSI(propname, 'Q', 1, 'T', T1, fluid_name)
                prop        = (1-frac) * prop_liquid + frac * prop_vapour
                # print ('Property retrieved using equilibrium mixture assuption for T = %.2f and S = %.2f.'%(T1, S1))
        
        else:

            # out side saturation line (liquid or gas phase)
            prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
            # print (prop)
    
    else:

        # above critical temperature (above critical point)
        prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
        # print (prop)
    
    return prop

def wet_vapour_properties_SI(propname, input_name1, input1, input_name2, input2, fluid_name = 'CarbonDioxide'):
    T_crit = PropsSI('Tcrit', fluid_name)
    
    # exec("T1 = PropsSI('T', '%s', %f, '%s', %f, '%s')"%(input_name1, input1, input_name2, input2, fluid_name))
    # exec("S1 = PropsSI('S', '%s', %f, '%s', %f, '%s')"%(input_name1, input1, input_name2, input2, fluid_name))

    T1 = PropsSI('T', input_name1, input1, input_name2, input2, fluid_name)
    S1 = PropsSI('S', input_name1, input1, input_name2, input2, fluid_name)

    D1 = PropsSI('D', 'T', T1, 'S', S1, fluid_name)
    
    # check for temperature lower than critical temperature
    if T1 < T_crit:

        # static density inside the wet range (point inside domain)
        D_liquid = PropsSI('D', 'Q', 0, 'T', T1, fluid_name)
        D_vapour = PropsSI('D', 'Q', 1, 'T', T1, fluid_name)
        if (D1 >= D_vapour) and (D1 <= D_liquid):

            # fraction determination
            if propname == 'D' or propname == 'speed_of_sound':
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) # volume fraction
            else:
                frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) * (D_vapour/D1) # mass fraction

            # property retrieval: partial derivatives requires special treatment
            if propname == 'speed_of_sound':

                # using partial chi
                prop = equilibrium_wet_vapour_speed_of_sound(frac, D1, T1, S1, 
                                                             D_liquid, D_vapour, 
                                                             fluid_name = fluid_name)

                # using full relaxation model
                # prop = equilibrium_wet_vapour_speed_of_sound_LUND(frac, D1, T1, S1, 
                #                                                   D_liquid, D_vapour, 
                #                                                   equilibrium_model = 3,
                #                                                   fluid_name = fluid_name)

                # direct partial derivative calculation
                # prop = dP_drho(Y[i,j], X[i,j], 0.0000001, fluid)

            else:
                
                if '|' in propname:
                    print('Warning: direct partial derivative retrieval in two-phase region may contain error.')
                
                # calculation using mass/volume fraction
                prop_liquid = PropsSI(propname, 'Q', 0, 'T', T1, fluid_name)
                prop_vapour = PropsSI(propname, 'Q', 1, 'T', T1, fluid_name)
                prop        = (1-frac) * prop_liquid + frac * prop_vapour
                # print ('Property retrieved using equilibrium mixture assuption for T = %.2f and S = %.2f.'%(T1, S1))
        
        else:

            # out side saturation line (liquid or gas phase)
            prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
            # print (prop)
    
    else:

        # above critical temperature (above critical point)
        prop = PropsSI(propname, 'T', T1, 'S', S1, fluid_name)
        # print (prop)
    
    return prop

def vapour_quality_SI(input_name1, input1, input_name2, input2, fluid_name = 'CarbonDioxide'):
    
    T_crit = PropsSI('Tcrit', fluid_name)

    T1 = PropsSI('T', input_name1, input1, input_name2, input2, fluid_name)
    S1 = PropsSI('S', input_name1, input1, input_name2, input2, fluid_name)

    D1     = PropsSI('D', 'T', T1, 'S', S1, fluid_name)
    
    # check for temperature lower than critical temperature
    if T1 < T_crit:

        # static density inside the wet range (point inside domain)
        D_liquid = PropsSI('D', 'Q', 0, 'T', T1, fluid_name)
        D_vapour = PropsSI('D', 'Q', 1, 'T', T1, fluid_name)
        if (D1 > D_vapour) and (D1 < D_liquid):

            frac = ((D1 - D_liquid) / (D_vapour - D_liquid)) * (D_vapour/D1) # mass fraction
        
        elif (D1 >= D_liquid):
            # saturate liquid
            frac = 0
        
        else:
            # saturate vapour
            frac = 1
    
    else:
        frac = 1
    
    return frac

# derivation from partial derivatives with partial chi term
def equilibrium_wet_vapour_speed_of_sound(alpha, D_mixture, T_mixture, S_mixture, D_liquid, D_vapour, fluid_name = 'CarbonDioxide'):
    # works the same as the PTmu equilibrium model introduced by Lund

    X  = alpha * D_vapour / D_mixture    # mass fraction

    # P  = PropsSI('P', 'D', D_mixture, 'S', S_mixture, fluid_name)
    P  = PropsSI('P', 'T', T_mixture, 'S', S_mixture, fluid_name)
    dP = 10.0

    # partial rho_vapour partial p
    D_vapour_upper = PropsSI('D', 'P', P+dP, 'Q', 1, fluid_name)
    D_vapour_lower = PropsSI('D', 'P', P-dP, 'Q', 1, fluid_name)
    partial1       = (D_vapour_upper - D_vapour_lower) / (2*dP)

    # partial rho_liquid partial p
    D_liquid_upper = PropsSI('D', 'P', P+dP, 'Q', 0, fluid_name)
    D_liquid_lower = PropsSI('D', 'P', P-dP, 'Q', 0, fluid_name)
    partial2       = (D_liquid_upper - D_liquid_lower) / (2*dP)

    # partial x partial p
    D_upper     = PropsSI('D', 'P', P+dP, 'S', S_mixture, fluid_name)
    D_lower     = PropsSI('D', 'P', P-dP, 'S', S_mixture, fluid_name)
    alpha_upper = (D_upper - D_liquid_upper) / (D_vapour_upper - D_liquid_upper)
    alpha_lower = (D_lower - D_liquid_lower) / (D_vapour_lower - D_liquid_lower)
    X_upper     = alpha_upper * D_vapour_upper / D_upper
    X_lower     = alpha_lower * D_vapour_lower / D_lower
    partial3    = (X_upper - X_lower) / (2*dP)

    Term1 = (alpha*D_mixture/D_vapour) * partial1
    Term2 = ((1-alpha)*D_mixture/D_liquid) * partial2
    Term3 = (D_mixture**2*(D_vapour-D_liquid) / (D_vapour*D_liquid)) * partial3
    
    temp  = Term1 + Term2 + Term3
    c     = (1/temp)**(0.5)

    return c

# A series of relaxation assumptions in mixture based on Lund's paper
def equilibrium_wet_vapour_speed_of_sound_LUND(alpha, D_mixture, T_mixture, S_mixture, 
                                               D_liquid, D_vapour, 
                                               equilibrium_model = 3, # 1: P, 2: P and T, 3: P, T, mu
                                               fluid_name = 'CarbonDioxide'):
    temp = 0

    # Pressure relaxation
    c_liquid = PropsSI('speed_of_sound', 'Q', 0, 'T', T_mixture, fluid_name)
    c_vapour = PropsSI('speed_of_sound', 'Q', 1, 'T', T_mixture, fluid_name)

    Z_p = alpha/(D_vapour * c_vapour**2) + (1-alpha)/(D_liquid * c_liquid**2)

    temp = temp + Z_p

    if equilibrium_model > 1:
        # Pressure and temperature relaxation

        # TP variation term
        eta_liquid = PropsSI('d(T)/d(P)|S', 'Q', 0, 'T', T_mixture, fluid_name)
        eta_vapour = PropsSI('d(T)/d(P)|S', 'Q', 1, 'T', T_mixture, fluid_name)
        
        # constant pressure specific heat
        c_p_liquid = PropsSI('d(S)/d(T)|P', 'Q', 0, 'T', T_mixture, fluid_name) * T_mixture
        c_p_vapour = PropsSI('d(S)/d(T)|P', 'Q', 1, 'T', T_mixture, fluid_name) * T_mixture
        
        # Big CP
        C_p_liquid = D_liquid * (1-alpha) * c_p_liquid
        C_p_vapour = D_vapour * alpha     * c_p_vapour

        Z_pt = (1/T_mixture) * ((C_p_liquid*C_p_vapour * (eta_liquid - eta_vapour)**2) / (C_p_liquid+C_p_vapour))
        temp = temp+Z_pt

        if equilibrium_model > 2:
            # Pressure, temperature and chemical potential relaxation

            # Enthalpy
            h_liquid = PropsSI('H', 'Q', 0, 'T', T_mixture, fluid_name)
            h_vapour = PropsSI('H', 'Q', 1, 'T', T_mixture, fluid_name)
            
            Z_ptmu = T_mixture * (C_p_liquid+C_p_vapour) * \
                                ((D_liquid-D_vapour) / (D_liquid*D_vapour*(h_liquid-h_vapour)) + \
                                (eta_vapour*C_p_vapour+eta_liquid*C_p_liquid) / (T_mixture*(C_p_liquid+C_p_vapour))  )**2
            temp = temp+Z_ptmu
    
    # temp = D_mixture * (Z_p + Z_pt + Z_ptmu)
    temp = D_mixture * temp
    c    = (1/temp)**(0.5)

    return c

def Gruneisen_parameter_principle(T1, S1, dT, fluid_name):

    D1 = PropsSI('D', 'T', T1, 'S', S1, fluid_name)
    v = 1 / D1

    U_lower = PropsSI('U', 'T', T1-dT, 'D', D1, fluid_name)
    U_upper = PropsSI('U', 'T', T1, 'D', D1, fluid_name)

    P_lower = PropsSI('P', 'T', T1-dT, 'D', D1, fluid_name)
    P_upper = PropsSI('P', 'T', T1, 'D', D1, fluid_name)
    
    dP_dU = (P_upper-P_lower) / (U_upper-U_lower)

    G = v * dP_dU

    return G


# Interpolation to aboid void in CoolProp
# ==========================================================================================================================

# Interpolation on DU flash
def PropsSI_DU(propname, D, U, fluid_name = 'CarbonDioxide', delta = 0.001, delta_max = 0.03):
    try: 
        return PropsSI(propname,'D',D,'U',U,fluid_name)
    except:

        while True:

            try:
                return (PropsSI(propname,'D',D*(1+delta),'U',U,fluid_name) + PropsSI(propname,'D',D*(1-delta),'U',U,fluid_name)) / 2.0
            except:
                delta = delta + 0.001

            if delta >= delta_max:
                raise RuntimeError('Invalid rho-u input: %4f, %.4f'%(D, U)) from None

# Interpolation on DH flash
def PropsSI_DH(propname, D, H, fluid_name = 'CarbonDioxide', delta = 0.001):
    try: 
        return PropsSI(propname,'D',D,'H',H,fluid_name)
    except:

        while True:

            try:
                return (PropsSI(propname,'D',D*(1+delta),'H',H,fluid_name) + PropsSI(propname,'D',D*(1-delta),'H',H,fluid_name)) / 2.0
            except:
                delta = delta + 0.001

            if delta >= 0.03:
                raise RuntimeError('Invalid rho-h input: %4f, %.4f'%(D, H)) from None

# Interpolation on DP flash
def PropsSI_DP(propname, D, P, fluid_name = 'CarbonDioxide', delta = 0.0000001):
    try: 
        return PropsSI(propname,'D',D,'P',P,fluid_name)
    except:

        while True:

            try:
                return (PropsSI(propname,'D',D,'P',P*(1+delta),fluid_name) + PropsSI(propname,'D',D,'P',P*(1-delta),fluid_name)) / 2.0
            except:
                delta = delta + 0.0000001

            if delta >= 0.0000003:
                raise RuntimeError('Invalid rho-P input: %4f, %.4f'%(D, P)) from None

# general case
def PropsSI_D_intp(propname, D, input2_name, input2, fluid_name, delta = 0.001):
    try: 
        return PropsSI(propname,'D',D,input2_name,input2,fluid_name)
    except:

        while True:

            try:
                interpolated_value = PropsSI(propname,'D',D*(1+delta),input2_name,input2,fluid_name) + PropsSI(propname,'D',D*(1-delta),input2_name,input2,fluid_name) / 2.0
                break
            except:
                delta = delta + 0.001

            if delta >= 0.03:
                raise RuntimeError('Invalid rho-%s input: %4f, %.4f'%(input2_name, D, input2)) from None

        print('Using interpolation for %s = %.2f, %s = %.2f with delta = %.5f'%('D', D, input2_name, input2, delta))
        return interpolated_value
    
# general case
def PropsSI_P_intp(propname, P, input2_name, input2, fluid_name, delta = 0.001):
    try: 
        return PropsSI(propname,'P', P,input2_name,input2,fluid_name)
    except:

        while True:

            try:
                return (PropsSI(propname,'P',P*(1+delta),input2_name,input2,fluid_name) + PropsSI(propname,'P',P*(1-delta),input2_name,input2,fluid_name)) / 2.0
            except:
                delta = delta + 0.001

            if delta >= 0.03:
                raise RuntimeError('Invalid P-%s input: %4f, %.4f'%(input2_name, P, input2)) from None