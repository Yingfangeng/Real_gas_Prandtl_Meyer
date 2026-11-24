from solver import prandtl_meyer_expansion
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':

    fluid = 'CO2'
    
    M_1   = 1.1
    theta = 3
    tolerance = 1e-5

    T_1 = 350
    s_1 = 1150

    result = prandtl_meyer_expansion(T_1, s_1, M_1, theta, tolerance, fluid)

    print(result)