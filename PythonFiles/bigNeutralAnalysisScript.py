import pandas as pd
from sqlalchemy import create_engine
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import random
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import gc

db_username_0 = 'simulationuser'
db_password_0 = 'simulations2024'
db_host_0 = 'localhost'
db_port_0 = '5432'
db_name_0 = 'simulations_data'
table_name_0 = 'neutral_ensembles'

connection_string = f"postgresql://{db_username_0}:{db_password_0}@{db_host_0}:{db_port_0}/{db_name_0}"
engine = create_engine(connection_string)
query = f"SELECT * FROM {table_name_0}"
df = pd.read_sql(query, engine)
# df = df.iloc[:10000]

def average_arrays(*arrays):
    if not arrays:
        raise ValueError("No arrays provided for averaging.")
    
    np_arrays = [np.array(arr) for arr in arrays]
    array_lengths = [len(arr) for arr in np_arrays]

    if len(set(array_lengths)) != 1:
        raise ValueError("All input arrays must have the same length.")
    
    average_array = np.mean(np_arrays, axis=0)
    
    return average_array

average_psi_e = average_arrays(*list(df['psi_e']))
average_b_e = average_arrays(*list(df['b_e']))
average_psi_plus = average_arrays(*list(df['psi_plus']))
average_b_plus = average_arrays(*list(df['b_plus']))
average_U = average_arrays(*list(df['u_list']))
average_R = average_arrays(*list(df['r_list']))
average_k_e_psi_e_list = average_arrays(*list(df['k_e_psi_e_list']))
average_k_e_b_e_list = average_arrays(*list(df['k_e_b_e_list']))
average_k_e_psi_plus_list = average_arrays(*list(df['k_e_psi_plus_list']))
average_k_e_b_plus_list = average_arrays(*list(df['k_e_b_plus_list']))
average_heat_flux_psi_e_b_e_list = average_arrays(*list(df['heat_flux_psi_e_b_e_list']))
average_heat_flux_psi_e_b_plus_list = average_arrays(*list(df['heat_flux_psi_e_b_plus_list']))
average_b_e_psi_plus_list = average_arrays(*list(df['b_e_psi_plus_list']))
average_b_e_b_plus_list = average_arrays(*list(df['b_e_b_plus_list']))
average_psi_plus_b_plus_list = average_arrays(*list(df['psi_plus_b_plus_list']))
average_eta = average_arrays(*list(df['eta_list']))

window_size = 5000
dt = .001
epsilon = 0.12394270273516043
N_0_squared = 318.8640217310387

# #<U*k_e_psi_e>
# temp_u_list = df['u_list'].apply(np.array)
# temp_k_e_psi_e_list = df['k_e_psi_e_list'].apply(np.array)
# UK_e_psi_e = np.array([i * j for i, j in zip(temp_u_list, temp_k_e_psi_e_list)])
# average_UK_e_psi_e = average_arrays(*list(UK_e_psi_e))

# #<U*k_e_psi_plus>
# temp_u_list = df['u_list'].apply(np.array)
# temp_k_e_psi_plus_list = df['k_e_psi_plus_list'].apply(np.array)
# UK_e_psi_plus = np.array([i * j for i, j in zip(temp_u_list, temp_k_e_psi_plus_list)])
# average_UK_e_psi_plus = average_arrays(*list(UK_e_psi_plus))

epsilon = 0.12394270273516043
N_0_squared = 318.8640217310387
r_m = 0.1
k = 2 * np.pi * 6
m = 2 * np.pi * 3
m_u = 2 * np.pi * 7
dt = 0.001
total_time = 200
k_e_square = k**2 + m**2
k_plus_square = k**2 + (m + m_u)**2

p = k / k_e_square
q = -k / k_plus_square
r = k * N_0_squared
s = -k * (k_plus_square - m_u**2) / (2*k_e_square)
t = -k * (m_u**2 - k_e_square) / (2 * k_plus_square)
v = k / 2

bigW = np.array([[-1, p, 0, 0], [-r, -1, 0, 0], [0, 0, -1, q], [0, 0, r, -1]])
bigL = np.array([[0, 0, s, 0], [0, 0, 0, v], [t, 0, 0, 0], [0, -v, 0, 0]])
bigQ = np.array([[8/k_e_square, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])

# def create_C_matrix(row):
#     length = len(row['k_e_psi_e_list'])  
#     matrices = []
    
#     for i in range(length):
#         ke_psi_e = row['k_e_psi_e_list'][i]
#         ke_b_e = row['k_e_b_e_list'][i]
#         ke_psi_plus = row['k_e_psi_plus_list'][i]
#         ke_b_plus = row['k_e_b_plus_list'][i]
        
#         heat_flux_psi_e_b_e = row['heat_flux_psi_e_b_e_list'][i]
#         heat_flux_psi_e_b_plus = row['heat_flux_psi_e_b_plus_list'][i]
#         b_e_psi_plus = row['b_e_psi_plus_list'][i]
#         b_e_b_plus = row['b_e_b_plus_list'][i]
        
#         psi_plus_b_plus = row['psi_plus_b_plus_list'][i]
#         psi_e_psi_plus = row['r_list'][i] 

#         C_matrix = np.array([
#             [ke_psi_e, heat_flux_psi_e_b_e, psi_e_psi_plus/(0.25 * k * (k_plus_square - k_e_square)), heat_flux_psi_e_b_plus],
#             [heat_flux_psi_e_b_e, ke_b_e, b_e_psi_plus, b_e_b_plus],
#             [psi_e_psi_plus/(0.25 * k * (k_plus_square - k_e_square)), b_e_psi_plus, ke_psi_plus, psi_plus_b_plus],
#             [heat_flux_psi_e_b_plus, b_e_b_plus, psi_plus_b_plus, ke_b_plus]
#         ])
        
#         matrices.append(C_matrix)

#     return matrices 

# df['C_Matrix'] = df.apply(create_C_matrix, axis = 1)

# C_Mat_Arr = df['C_Matrix']

# del df
# gc.collect()


# #<WC>

# C_Mat_Arr_Multiplied = C_Mat_Arr.apply(lambda matrix_list: np.array([bigW @ matrix for matrix in matrix_list]))

# average_WC = np.mean(C_Mat_Arr_Multiplied.tolist(), axis = 0)

# average_WC_term = np.mean(average_WC, axis=0)

# C_Mat_Arr_Multiplied = C_Mat_Arr.apply(lambda matrix_list: [matrix @ np.transpose(bigW) for matrix in matrix_list])

# average_WC_Transpose = np.mean(C_Mat_Arr_Multiplied.tolist(), axis = 0)

# average_WC_Transpose_term = np.mean(average_WC_Transpose, axis=0)

# #<ULC>
# C_Mat_Arr_ULC_Multiplied = []

# for idx, (matrix_list, u_list) in enumerate(zip(C_Mat_Arr, df['u_list'])):
#     result = []
#     for u, matrix in zip(u_list, matrix_list):
#         uL = u * bigL  # Broadcast scalar u across the matrix L
#         result_matrix = uL @ matrix  # Matrix multiplication
#         result.append(result_matrix)
#     C_Mat_Arr_ULC_Multiplied.append(result)

# average_ULC = np.mean(C_Mat_Arr_ULC_Multiplied, axis = 0)

# average_ULC_term = np.mean(average_ULC, axis=0)

# #<UCL^T>
# C_Mat_Arr_UCLT_Multiplied = []

# L_transposed = bigL.T

# for idx, (matrix_list, u_list) in enumerate(zip(C_Mat_Arr, df['u_list'])):
#     result = []
#     for u, matrix in zip(u_list, matrix_list):
#         uC = u * matrix  # Multiply the scalar u with the matrix C
#         result_matrix = uC @ L_transposed  # Multiply by the transposed matrix L
#         result.append(result_matrix)
#     C_Mat_Arr_UCLT_Multiplied.append(result)

# average_UCLT = np.mean(C_Mat_Arr_UCLT_Multiplied, axis = 0)
# average_UCLT_term = np.mean(average_UCLT, axis=0)

# average_C_Matrix = np.mean(df['C_Matrix'].tolist(), axis = 0)

def std_across_axis(arrays):
    """
    Calculate the standard deviation across axis 1 for a list or array of 2D numpy arrays.
    Returns a single array of standard deviations for each array in the input.
    
    Parameters:
        arrays (array-like): List or array of shape (N, M) or (N, M, ...) where N is the number of arrays.
        
    Returns:
        np.ndarray: Array of standard deviations, shape (N,).
    """
    # arrays = np.asarray(arrays)
    return np.std(arrays, axis=0)

U_k_e_psi_e = np.array([
    np.array(u) * np.array(k)
    for u, k in zip(df['u_list'], df['k_e_psi_e_list'])
])

U_k_e_psi_e_stdev = std_across_axis(U_k_e_psi_e)

U_k_e_psi_e_average = average_arrays(*list(U_k_e_psi_e))

def std_across_axis(arrays):
    return np.std(arrays, axis=0)

U_k_e_psi_e_stdev = std_across_axis(U_k_e_psi_e)
U_k_e_psi_e_stError = U_k_e_psi_e_stdev/100
U_k_e_psi_e_average = average_arrays(*list(U_k_e_psi_e))

average_U_Ke_psi_e = average_U * average_k_e_psi_e_list

std_U = np.std(df['u_list'].tolist(), axis=0)
std_K_e_psi_e = np.std(df['k_e_psi_e_list'].tolist(), axis=0)

std_err_U = std_U / 100
std_err_k_e_psi_e = std_K_e_psi_e / 100

st_err_U_K_e_psi_e = average_U * average_k_e_psi_e_list * np.sqrt((std_err_U / average_U) ** 2 + (std_err_k_e_psi_e / average_k_e_psi_e_list) ** 2)
time_array = np.linspace(-5, 5, int(10/.01))
fig, ax = plt.subplots(figsize=(12, 6))
standard_error_diff = np.sqrt(st_err_U_K_e_psi_e ** 2 + U_k_e_psi_e_stError ** 2)
calc_array = U_k_e_psi_e_average - average_U_Ke_psi_e
# ax.plot(time_array, U_k_e_psi_e_average, label=r'$\langle U \cdot k_{e}\psi_{e} \rangle$', linewidth=2)
ax.plot(time_array, calc_array,
        label=r'$\langle U \rangle \cdot \langle k_{e}\psi_{e} \rangle$', linestyle='-', linewidth=2)

ax.fill_between(time_array, 
                calc_array - standard_error_diff, 
                calc_array + standard_error_diff, 
                alpha=1, label='Standard Error',color='red')

# ax.errorbar(time_array, 
#             U_k_e_psi_e_average - (average_U * average_k_e_psi_e_list), 
#             yerr=standard_error_diff, 
#             label=r'$\langle U \rangle \cdot \langle k_{e}\psi_{e} \rangle$', 
#             linestyle='--', 
#             linewidth=2, 
#             capsize=3)

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title(r'Comparison of $\langle U \cdot k_{e}\psi_{e} \rangle$ vs $\langle U \rangle \cdot \langle k_{e}\psi_{e} \rangle$', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True)

plt.tight_layout()
plt.show()

