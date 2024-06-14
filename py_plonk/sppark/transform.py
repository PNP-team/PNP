import gmpy2
import numpy as np
import time
import re

# Scale_1=134217728
# Scale_2=16777216
Scale_1=8192
Scale_2=1024
def parse_bigint(s,limbs):
    start = s.find('"(') + 2
    end = s.find(')')
    bigint_str = s[start:end]
    data = gmpy2.mpz(bigint_str,16)
    data_array = []
    for i in range(limbs):
        parse_data = int(data & 0xFFFFFFFFFFFFFFFF)
        data = data >> 64
        data_array.append(parse_data)
    return data_array

def read_pp_data(filename, limbs):

    with open(filename, "r") as file:
        data = file.read()

    lines = data.split('\n')
    current_section = None
    
    powers_of_g = []
    powers_of_gamma_g = []
    h = []
    beta_h = []
    for line in lines:
        if line.endswith(":"):
            current_section = line.rstrip(":")
        elif line.startswith("["):
            values = line.strip("[] ").split()
            if current_section == "powers_of_g":
                x_str = values[1]
                y_str = values[2]
                G1_point = [parse_bigint(x_str,limbs),parse_bigint(y_str,limbs)]
                powers_of_g.append(G1_point)
            elif current_section == "powers_of_gamma_g":
                x_str = values[1]
                y_str = values[2]
                G1_point = [parse_bigint(x_str,limbs),parse_bigint(y_str,limbs)]
                powers_of_gamma_g.append(G1_point)
            elif current_section == "h":
                h.append([parse_bigint(values[1],limbs),parse_bigint(values[2],limbs),
                          parse_bigint(values[3],limbs),parse_bigint(values[4],limbs)])
                
            elif current_section == "beta_h":
                beta_h.append([parse_bigint(values[1],limbs),parse_bigint(values[2],limbs),
                               parse_bigint(values[3],limbs),parse_bigint(values[4],limbs)])
 
    pp_dict = {"powers_of_g":np.array(powers_of_g, dtype=np.uint64),
               "powers_of_gamma_g":np.array(powers_of_gamma_g, dtype=np.uint64),
               "h":np.array(h, dtype=np.uint64),
               "beta_h":np.array(beta_h, dtype=np.uint64)}
    # Save the dictionary to a binary file using numpy.savez
    np.savez('pp-3.npz', **pp_dict)
        



def read_pk_data(filename, limbs):
    start = time.time()
    data = {
        "n": 0,
        "arithmetic": {
            "q_m": {
                "coeffs": 
                    np.zeros([], dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
                
            },
            "q_l": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_r": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_o": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_4": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
                
            },
            "q_c": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_hl": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_hr": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_h4": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            },
            "q_arith": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)    
            }
        },
        "range_selector": {
            "coeffs": 
                np.array([],dtype = np.uint64)
            ,
            "evals": 
                np.zeros((Scale_1,4), dtype = np.uint64)
        },
        "logic_selector": {
            "coeffs": 
                np.array([],dtype = np.uint64)
            ,
            "evals": 
                np.zeros((Scale_1,4), dtype = np.uint64)   
        },
        "lookup": {
            "q_lookup": {
                "coeffs": 
                    np.array([],dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
            },
            "table1": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)     
            },
            "table2": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
            },
            "table3": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)  
            },
            "table4": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64) 
            }
        },
        "fixed_group_add_selector": {
            "coeffs": 
                np.array([],dtype = np.uint64)
            ,
            "evals": 
                np.zeros((Scale_1,4), dtype = np.uint64)
        },
        "variable_group_add_selector": {
            "coeffs": 
                np.array([],dtype = np.uint64)
            ,
            "evals": 
                np.zeros((Scale_1,4), dtype = np.uint64)   
        },
        "permutation": {
            "left_sigma": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64) 
            },
            "right_sigma": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
            },
            "out_sigma": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
            },
            "fourth_sigma": {
                "coeffs": 
                    np.zeros((Scale_2,4), dtype = np.uint64)
                ,
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
            }
        },

        "linear_evaluations": {
                "evals": 
                    np.zeros((Scale_1,4), dtype = np.uint64)
        },

        "v_h_coset_8n": {
            "evals": 
                np.zeros((Scale_1,4), dtype = np.uint64)
        }
    }

    with open(filename, "r") as file:
        lines = file.readlines()

    n = int(lines[0].split(':')[1].strip())
    data["n"]=n
    current_key = None
    pattern = r'\(.*?\)'

    for line in lines[1:]:
        line = line.strip()
        if line.startswith("arithmetic:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("range_selector:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("logic_selector:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("lookup:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("fixed_group_add_selector:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("variable_group_add_selector:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("permutation:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("linear_evaluations:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("v_h_coset_8n:"):
            i = 0
            current_key = line[:-1]
            subkey=None
        elif line.startswith("q_m"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_l"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_r"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_o"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_4"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_c"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_hl"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_hr"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_h4"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_arith"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="range_selector" and subkey==None:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="range_selector" and subkey:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="logic_selector" and subkey==None:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="logic_selector" and subkey:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("q_lookup:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("table1:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("table2:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("table3:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("table4:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("left_sigma:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("right_sigma:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("out_sigma:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.startswith("fourth_sigma:"):
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="fixed_group_add_selector" and subkey==None:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="fixed_group_add_selector" and subkey:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="variable_group_add_selector" and subkey==None:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="variable_group_add_selector" and subkey:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="linear_evaluations" and subkey==None:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key=="v_h_coset_8n" and subkey==None:
            i = 0
            subkey=line[:-1]
            subsubkey=None
        elif line.endswith(":") and current_key and subkey and subsubkey==None:
            i = 0
            subsubkey=line[:-1]
        elif line.endswith(":") and current_key and subkey and subsubkey:
            i = 0
            subsubkey=line[:-1]
        elif not(line.endswith(":")):
            matches = re.findall(pattern, line)
            element_str = matches[0].strip('()')
            extracted_list = np.array([int(x) for x in element_str.split(',')], dtype = np.uint64)
            if subkey==None and subsubkey==None:
                for j in range(limbs):
                    data[current_key][i][j] = extracted_list[j]
            elif subkey and subsubkey==None:
                for j in range(limbs):
                    data[current_key][subkey][i][j] = extracted_list[j]
            elif line.startswith("-") and subsubkey:
                subsubkey="coeffs"
                for j in range(limbs):
                    data[current_key][subkey][subsubkey][i][j] = extracted_list[j]
            elif subsubkey:
                subsubkey="evals"
                for j in range(limbs):
                    data[current_key][subkey][subsubkey][i][j] = extracted_list[j]
            i += 1
    
    # Save the dictionary to a binary file using numpy.savez
    np.savez('pk-17.npz', **data)
    elapse = time.time() - start
    print("耗时：",elapse)

def read_cs_data(filename, limbs):
    with open(filename,"r")as file:
        lines = file.readlines()
    n = int(lines[0].split(':')[1].strip())
    intended_pi_pos=lines[1].split(':')[1].strip()
    zero_var=int(lines[2].split(':')[1].strip())
    pattern = r'\[([\d\s,]+)\]'
    match = re.search(pattern, lines[-1])
    pi = match.group(1)
    public_inputs_list = [int(x) for x in pi.split(',')]
    data = {}
    data["n"]=n
    data["intended_pi_pos"]=eval(intended_pi_pos)
    data["public_inputs"] = np.array(public_inputs_list, dtype = np.uint64)
    data["zero_var"]=zero_var
    current_key = None

    # 解析文件数据
    for line in lines[3:-2]:
        line = line.strip()
        if line.endswith(":"):
            current_key=line.rstrip(":")
            data[current_key]=[]
        elif line.startswith("Fp"):
            value = np.array(parse_bigint(line,limbs), dtype = np.uint64)
            data[current_key].append(value)
        else:
            data[current_key].append(int(line))
    np.savez('cs-17.npz', **data)

import numpy as np

def read_scalar_data(filename):
    big_list = []

    with open(filename, 'r') as file:
        for line in file:
            start_index = line.find("([")
            end_index = line.find("])")

            if start_index != -1 and end_index != -1:

                list_str = line[start_index + 2:end_index]
                values = [int(x) for x in list_str.split(',')]
                big_list.append(values)


    output_filename = filename.split('.')[0] + '_scalar-3.npy'

    np.save(output_filename, np.array(big_list, dtype = np.uint64))
    
# import os
# print(os.getcwd())
# read_pp_data("params.txt",6)
# read_cs_data("cs.txt",4)
# read_scalar_data("py_plonk/w_l_scalar.txt")
# read_scalar_data("py_plonk/w_r_scalar.txt")
# read_scalar_data("py_plonk/w_o_scalar.txt")
# read_scalar_data("py_plonk/w_4_scalar.txt")

# read_pk_data("py_plonk/pk.txt", 4)
# read_pp_data("params.txt", 6)
# read_cs_data("cs.txt", 4)
# read_scalar_data("w_l_scalar.txt")
# read_scalar_data("w_r_scalar.txt")
# read_scalar_data("w_o_scalar.txt")
# read_scalar_data("w_4_scalar.txt")
# read_pk_data("pk.txt", 4)
