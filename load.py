import gmpy2
import torch
import re
from typing import List
from dataclasses import dataclass

def from_list_gmpy(input:list):
    out = []
    for i in range(len(input)):
        output = 0 
        for j in reversed(input[i]):
            output = output<<64
            output = output | j
        out.append(output)
    return out

def from_gmpy_tensor(input:list,limbs,dtype):
    for i in range(len(input)):
        output = []
        for j in range(limbs):
            output.append( int(input[i] & 0xFFFFFFFFFFFFFFFF) )
            input[i] = input[i] >> 64
        input[i] = output
    return torch.tensor(input, dtype = dtype)



def parse_bigint(s, limbs):
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

def read_pp_data(filename):
    # 打开文本文件以读取数据
    with open(filename, "r") as file:
        data = file.read()

    powers_of_g = []

    lines = data.split('\n')
    current_section = None

    for line in lines:
        if line.endswith(":"):
            current_section = line.rstrip(":")
        elif line.startswith("["):
            values = line.strip("[] ").split()
            if current_section == "powers_of_g":
                x_str = values[1]
                y_str = values[2]
                powers_of_g.append(parse_bigint(x_str,6))
                powers_of_g.append(parse_bigint(y_str,6))
                
    return powers_of_g

def read_scalar_data(filename):
    with open(filename, "r") as file:
        content = file.read()

    # 使用正则表达式匹配包含列表的字符串
    pattern = r'\[.*?\]'
    matches = re.findall(pattern, content)

    big_list = []

    # 如果第一个元素不符合正则表达式，手动添加到大列表
    if content.startswith('Fp256'):
        str_list=content.split('Fp256(BigInteger256(')[1].split(')')[0]
        element_str = str_list.strip('[]')
        elements = [int(e) for e in element_str.split(',')]
        big_list.append(elements)

    matches.pop(0)

    # 遍历匹配的字符串列表并处理
    for match in matches:
        # 去除字符串中的方括号，并按逗号分割成元素列表
        elements_str = match.strip('[]')
        elements = [int(e) for e in elements_str.split(',')]

        big_list.append(elements)
    return big_list