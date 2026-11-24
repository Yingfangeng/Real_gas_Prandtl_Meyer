import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from real_gas_property import *



def plot_downstream_mach_number(path):

    df = pd.read_csv(path)

    failed = df[df['M_2'] == 999]

    M = df.pivot(index='T_1', columns='s_1', values='M_2')
    
    M = M.mask(M == 999)
    M = M.mask(M == 9999)

    S, T = np.meshgrid(M.columns, M.index)

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(S, T, M.to_numpy(), levels=100, cmap='rainbow')

    ax.scatter(failed['s_1'], failed['T_1'], color='black', marker='x',
                s=30, label='Failed convergence')

    fig.colorbar(contour, ax=ax, label=r'$M_2$')
    
    ax.set_xlabel('$s_1$ (J/kg/K)')
    ax.set_ylabel('$T_1$ (K)')
    

    plt.tight_layout()
    plt.show()



def plot_isentropic_mach_number(s_0, T_0, T_1, points):

    fluid = 'CO2'
    T_list = np.linspace(T_0, T_1, points)
    
    P_list = CP.PropsSI('P', 'S', s_0, 'T', T_list, fluid)
    h_list = CP.PropsSI('H', 'S', s_0, 'T', T_list, fluid)

    a_list = []
    gamma_list = []
    for T in T_list:
        gamma_list.append(PropsSI_General('fundamental_derivative_of_gas_dynamics', 'S', s_0, 'T', T, fluid))
        a_list.append(PropsSI_General('A', 'S', s_0, 'T', T, fluid))
    a_list = np.array(a_list)
    gamma_list = np.array(gamma_list)

    # for gamma in gamma_list:
    #     if gamma < 1:
    #         print(gamma)

    h_0 = CP.PropsSI('H', 'S', s_0, 'T', T_0, fluid)
    V_list = (2*(h_0 - h_list))**0.5

    M_list = V_list / a_list

    fig, ax = plt.subplots()

    J = 1 - gamma_list - (M_list)**(-2)
    ax.plot(T_list, M_list, marker = 'o')
    ax.grid(ls = ':')

    fig2, ax2 = plt.subplots()
    ax2.plot(T_list, J, marker = 'o')
    ax2.grid(ls = ':')
    ax2.set_ylim(-5,1)
    plt.show()



def compare_theory_and_CFD(theory_path, cfd_path):

    theory = pd.read_csv(theory_path)
    cfd = pd.read_csv(cfd_path)

    fig, ax = plt.subplots()

    ax.plot(theory['M_1'], theory['M_2'], color = 'b', label = 'theory')
    ax.scatter(cfd['M1'], cfd['M2'], color = 'r', label = 'CFD')

    ax.legend()
    ax.set_xlabel('M1')
    ax.set_ylabel('M2')
    ax.grid(ls = ':')

    plt.show()