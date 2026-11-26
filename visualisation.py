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
    # ax2.set_ylim(-5,1)
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



def compare_ideal_and_real(path):

    df = pd.read_csv(path)

    fig, axes = plt.subplots(1,2, figsize = (15,5))

     # Use the same contour levels for both plots
    M_min = min(df['M_2'].min(), df['M_2_model'].min())
    M_max = max(df['M_2'].max(), df['M_2_model'].max())
    levels = np.linspace(M_min, M_max, 100)

    ax1 = axes[0]
    ax2 = axes[1]

    contour1 = ax1.tricontourf(df['s_1'], df['T_1'], df['M_2_model'], levels=levels, cmap='rainbow')
    ax1.set_title('Ideal Calculation', fontsize=14)
    ax1.set_xlabel(r'$s_1$ (J/kg/K)', fontsize=12)
    ax1.set_ylabel(r'$T_1$ (K)', fontsize=12)
    ax1.tick_params(labelsize=11)

    contour2 = ax2.tricontourf(df['s_1'], df['T_1'], df['M_2'], levels=levels, cmap='rainbow')
    ax2.set_title('Theory', fontsize=14)
    ax2.set_xlabel(r'$s_1$ (J/kg/K)', fontsize=12)
    ax2.set_ylabel(r'$T_1$ (K)', fontsize=12)
    ax2.tick_params(labelsize=11)


    cbar = fig.colorbar(contour1, ax=axes, label=r'$M_2$', pad=0.02)
    cbar.ax.tick_params(labelsize=11)

    plt.show()



def plot_ideal_and_model_error_contour(ideal_path, model_path):

    ideal_df = pd.read_csv(ideal_path)
    model_df = pd.read_csv(model_path)

    
    ideal_error = (np.abs(ideal_df['M_2_model'] - ideal_df['M_2'])/ np.abs(ideal_df['M_2']) * 100)
    model_error = (np.abs(model_df['M_2_model'] - model_df['M_2'])/ np.abs(model_df['M_2']) * 100)

    E_min = 0
    E_max = max(ideal_error.max(), model_error.max())
    levels = np.linspace(E_min, E_max, 100)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    ax1 = axes[0]
    ax2 = axes[1]

    contour1 = ax1.tricontourf(ideal_df['s_1'], ideal_df['T_1'], ideal_error, levels=levels, cmap='rainbow')
    ax1.set_title('Ideal Calculation Error', fontsize=14)
    ax1.set_xlabel(r'$s_1$ (J/kg/K)', fontsize=12)
    ax1.set_ylabel(r'$T_1$ (K)', fontsize=12)
    ax1.tick_params(labelsize=11)

    contour2 = ax2.tricontourf(model_df['s_1'], model_df['T_1'], model_error, levels=levels, cmap='rainbow')
    ax2.set_title('Fitted Model Error', fontsize=14)
    ax2.set_xlabel(r'$s_1$ (J/kg/K)', fontsize=12)
    ax2.set_ylabel(r'$T_1$ (K)', fontsize=12)
    ax2.tick_params(labelsize=11)

    cbar = fig.colorbar(contour1, ax=axes, label='Relative Error (%)', pad=0.02)
    cbar.ax.tick_params(labelsize=11)

    plt.show()



def plot_ideal_and_model_linear(ideal_path, model_path):

    ideal_df = pd.read_csv(ideal_path)
    model_df = pd.read_csv(model_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    ax1 = axes[0]
    ax2 = axes[1]

    ax1.plot(ideal_df['M_2'], ideal_df['M_2'], color = 'r', label = 'Theory')
    ax1.scatter(ideal_df['M_2'], ideal_df['M_2_model'], color = 'b', s = 5, label = 'ideal')

    ax2.plot(model_df['M_2'], model_df['M_2'], color = 'r', label = 'Theory')
    ax2.scatter(model_df['M_2'], model_df['M_2_model'], color = 'b', s = 5)

    ax1.set_xlim(1,4)
    ax1.set_ylim(1,4)
    ax1.set_ylabel('M2 calculated')
    ax1.set_xlabel('M2 theory')
    ax1.set_title('Ideal calculation')
    ax1.grid(ls = ':')


    ax2.set_xlim(1,4)
    ax2.set_ylim(1,4)
    ax2.set_ylabel('M2 calculated')
    ax2.set_xlabel('M2 theory')
    ax2.set_title('Fitted model calculation')
    ax2.grid(ls = ':')

    plt.show()



def plot_sensitivity_study(path):

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df['variable'], df['forest_importance_percent'])
    ax.set_ylabel('Relative importance (%)')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()