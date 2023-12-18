import math

def list_to_uint64(num_list):
    # 将列表中的4个uint64数字拼接成1个大数
    big_num = (num_list[3] << 192) | (num_list[2] << 128) | (num_list[1] << 64) | num_list[0]
    
    # 将大数拆分成8个uint32组成的列表
    uint32_list = [((big_num >> (i * 32)) & 0xFFFFFFFF) for i in range(7, -1, -1)]
    hex_results = ["{:#010x}".format(num) for num in uint32_list]
    hex_results.reverse()
    return hex_results

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x

def mod_inverse(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"{a} 在模 {m} 下没有逆元，因为它们不互质。")
    else:
        return x % m
    
def compute(num_list ,mod_list):
    # 将列表中的4个uint64数字拼接成1个大数
    big_num = (num_list[3] << 192) | (num_list[2] << 128) | (num_list[1] << 64) | num_list[0] 
    mod = (mod_list[7] << 224) | (mod_list[6] << 192) | (mod_list[5] << 160) | (mod_list[4] << 128) | \
              (mod_list[3] << 96) | (mod_list[2] << 64) | (mod_list[1] << 32) | mod_list[0] 
    # 将大数拆分成8个uint32组成的列表

    square_inv = mod_inverse(big_num, mod)
    square_inv = (square_inv * (2 ** 256)) % mod
    square = (big_num * (2 ** 256)) % mod
    uint32_list1 = [((square >> (i * 32)) & 0xFFFFFFFF) for i in range(7, -1, -1)]
    uint32_list2 = [((square_inv >> (i * 32)) & 0xFFFFFFFF) for i in range(7, -1, -1)]
    hex_results1 = ["{:#010x}".format(num) for num in uint32_list1]
    hex_results2 = ["{:#010x}".format(num) for num in uint32_list2]
    hex_results1.reverse()
    hex_results2.reverse()
    return hex_results1,hex_results2
# 输入一个长度为4的列表
def frombase(org, mod_list):
    mod = (mod_list[7] << 224) | (mod_list[6] << 192) | (mod_list[5] << 160) | (mod_list[4] << 128) | \
              (mod_list[3] << 96) | (mod_list[2] << 64) | (mod_list[1] << 32) | mod_list[0]
    
    org_inv = mod_inverse(org, mod)
    org_inv = (org_inv * (2**256)) % mod
    uint32_list1 = [((org_inv >> (i * 32)) & 0xFFFFFFFF) for i in range(7, -1, -1)]
    hex_results1 = ["{:#010x}".format(num) for num in uint32_list1]
    hex_results1.reverse()
    return hex_results1

input_list = [0x5b2b3e9cfffffffd, 0x992c350be3420567, 0xffffffffffffffff, 0x3fffffffffffffff]
mod_list = [0x00000001, 0x8c46eb21, 0x0994a8dd, 0x224698fc, 0x00000000, 0x00000000, 0x00000000, 0x40000000]
# 调用函数得到结果
k=5
result_list1 = frombase(2**k,mod_list)
#
# 打印结果
print("输入列表:", input_list)
print("拼接成大数:", hex((input_list[3] << 192) | (input_list[2] << 128) | (input_list[1] << 64) | input_list[0]))
print([num for num in result_list1])