import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def plot_downstream_mach_number(path):

    df = pd.read_csv(path)

    failed = df[df['M_2'] == 999]

    M = df.pivot(index='T_1', columns='s_1', values='M_2')
    
    M = M.mask(M == 999)

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