
# /*   Cerificate:
# //... This Code was Developed By Mabon Manoj Ninan
# ..... University of Cincinnati
# ......Reseacher Assitant: UCDASEC- Dr. Boyang Wang
# ..... 02/22/2023
# */
import numpy as np
import os, sys
from math import sqrt, isnan
from loadDataUtility import *
from graphGenerationUtilities import *

def data_info(data):
    """
    This function prints the information of the dataset.
    """
    # loading the dataset
    try:
        power_traces, plain_text, key = data['power_trace'], data['plain_text'], data['key']
    except Exception:
        try:
            power_traces, plain_text, key = data['power_trace'], data['plaintext'], data['key']
        except Exception:
            power_traces, plain_text, key = data['trace_mat'], data['textin_mat'], data['key']
        
    print('shape of the power traces: ', power_traces.shape)
    print('shape of the plaintext : ', plain_text.shape)
    key_str = [hex(x) for x in key]
    print('Ground Truth for the key : ', key_str)
def gen_features_and_labels_256_nicv(data, input_target_byte, start_index, end_index):
    """
    This function generates features and labels for the dataset.
    Although similar, this function differs somewhat from the one present in the Step 2.1 notebook.
    It differs from the corresponding function in the TVLA notebook as well.
    """
    # loading the dataset
    try:
        power_traces, plain_text, key = data['power_trace'], data['plain_text'], data['key']
    except Exception:
        try:
            power_traces, plain_text, key = data['power_trace'], data['plain_text'], data['key']
        except Exception:
            power_traces, plain_text, key = data['trace_mat'], data['textin_mat'], data['key']
    
    # Getting the key_byte_value AKA label
    key_byte_value = key[input_target_byte]

    print('generating features and labels for the key byte value: ', key_byte_value)

    labels = [] 
    for i in range(plain_text.shape[0]):
        text_i = plain_text[i]
        # Some plain text values are stored as floats so they must be converted to an int before using bitwise xor
        label = aes_internal(int(text_i[input_target_byte]), key_byte_value) #key[i][input_key_byte]
        labels.append(label)

    labels = np.array(labels)
    if not isinstance(power_traces, np.ndarray):
        power_traces = np.array(power_traces)
    power_traces = power_traces[:, start_index:end_index]

    return power_traces, labels
def load_data_nicv(params):
    """
    This function loads the dataset required.
    """
    print('preparing data ...')
    target_byte = params['target_byte']
    start_idx, end_idx = params["start_idx"], params["end_idx"]
    file_name = params["input_path"]
    
    train_data_whole_pack = np.load(file_name)
    print("access file: {}".format(file_name))
    data_info(train_data_whole_pack)

    print('-'*80)
    print('processing data...')
    power_traces, labels = gen_features_and_labels_256_nicv(train_data_whole_pack,
                                                            target_byte,
                                                            start_idx, end_idx)

    power_traces = power_traces[:params["n"], :]
    labels = labels[:params["n"]]

    print('reshaped power traces: ', power_traces.shape)
    print('shape of the labels: ', labels.shape)

    return power_traces, labels
def calculate_nicv_values(labels_arr, Y_var):
    '''
    This function computes the nicv values (mean, variance, NICV) of the labels_arr
    '''
    Z = [] # A 1D array containing the means of each label (row) is instantiated (AKA Z array).
    for i in range(np.shape(labels_arr)[0]): # Each row (power traces with specific label) is iterated through.
        non_zero_elements = labels_arr[i][labels_arr[i] != 0] # The non-zero elements of the current row are saved.
        if not(len(non_zero_elements)): 
            continue
        else: # Else, the average of the current row's non-zero elements are calculated.
            Z.append(np.average(non_zero_elements))

    
    Z_var = np.var(Z, ddof=1) # The variance of the Z array is calculated.
    if isnan(Z_var/Y_var):
        return 0
    return Z_var/Y_var # NICV is returned

def save_NICV(power_traces, NICV_vals, path_to_save_nicv):
    '''
    This function saves the nicv results to a csv file.
    '''
    # The file name is of the format: "target-byte-x"
    # The thought is that the parent directories will provide the necessary information as to what this file name represents.
    f_name = "target-byte-{}".format(target_byte+1)
    nicv_file_path = path_to_save_nicv + '-{}.csv'.format(f_name)
    
    # Data is an iterator of tuples. These tuples contain the time (incremented by 1) and the corresponding t-value.
    data = zip(range(data_params["start_idx"] + 1, data_params["end_idx"] + 1), NICV_vals)
    nicv_df = pd.DataFrame(list(data))
    nicv_df.to_csv(nicv_file_path, index=False, header=["time", "nicv-value"])
    print("Normalized Inter-Class Variance results sucessfully saved to csv file: {}".format(nicv_file_path))
    return nicv_file_path
def compute_normalized_inter_class_variance(power_traces,data_params, labels, debug=False):
    '''
    This function computes the normalized inter-class variance.
    '''
    NICV_vals = []
    for i in range(np.shape(power_traces)[1]): # Each column (time) of the power_traces array is analyzed.
        curr_power_traces_col = power_traces[:,i]
        var_curr_power_traces_col = np.var(curr_power_traces_col, ddof=1) # The variance of the current column is calculated for NICV.
        labels_arr = np.zeros((256, power_traces.shape[0])) # NOTE: For debugging, replace the "256" with the length of debug key_byte_values (3)
        for j in range(np.shape(curr_power_traces_col)[0]): # Each row of the current power traces column is analyzed.
            labels_arr[labels[j]][j] = curr_power_traces_col[j]
        NICV = calculate_nicv_values(labels_arr, var_curr_power_traces_col)
        NICV_vals.append(NICV)
        
        if debug: # If debug is enabled, additional information will be printed to the screen.
            print("Round {}".format(i+1))
            print("\tThe nicv result is: {}".format(NICV))
    if not(debug):
        print("Saving test vector leakage assessment results to csv file...")
        nicv_save_path = save_NICV(power_traces, NICV_vals, data_params["path_to_save_nicv"])
        return nicv_save_path
    else:
        return None
# author @Mabon Manoj Ninan
# gh: mabonmn