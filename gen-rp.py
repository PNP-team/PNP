import gmpy2

def into_repr(input, limbs):
    output = []
    for i in range(limbs):
        x = input & 0xffffffff
        input = input >> 32
        output.append(x)
    return output

def into_repr_64(input, limbs):
    output = []
    for i in range(limbs//2):
        x = input & 0xffffffffffffffff
        input = input >> 64
        output.append(x)
    return output

def from_repr(input,limbs):
    output = 0
    for i in range(limbs):
        output = output << 32
        output = output|input[limbs-i-1]
    return output

def square(input,inv,mod):
    output = (input*input*inv)%mod
    return output

def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = extended_gcd(b % a, a)
        return (g, x - (b // a) * y, y)

def mod_inverse(a, m):
    g, x, y = extended_gcd(a, m)
    if g != 1:
        raise Exception('mod inverse does not exist!')
    else:
        return x % m
def closest_power_of_two(n):
    power = 32
    while power < n:
        power *= 2
    return power

if __name__ == "__main__":
    actualbits = 59
    bits = 64
    limbs = (bits+31)//32
    S = 18
    modulus = 1152921504606584833
    t = 4398046511103
    group_gen = 10
    root_of_unity = int(gmpy2.powmod(gmpy2.mpz(group_gen),gmpy2.mpz(t),gmpy2.mpz(modulus)))
    root_of_unity = root_of_unity *(2**bits) % modulus
    modulus_repr = into_repr_64(modulus,limbs)
    modulus_repr = [f"0x{element:016x}" for element in modulus_repr]
    R = (2**bits) % modulus
    R2 = (2**(bits*2)) % modulus
    R_INV = mod_inverse(R,modulus)%modulus
    R_repr = into_repr_64(R,limbs)
    R_repr = [f"0x{element:016x}" for element in R_repr]
    R2_repr = into_repr_64(R2,limbs)
    R2_repr = [f"0x{element:016x}" for element in R2_repr]
    N_INV = (-mod_inverse(modulus,2**bits))%(2**bits)
    N_INV_repr = into_repr_64(N_INV,limbs)
    N_INV_repr = [f"0x{element:016x}" for element in N_INV_repr]
    group_gen_r = (group_gen * (2**bits)) % modulus
    group_gen_repr = into_repr(group_gen_r,limbs)
    group_gen_repr = [f"0x{element:08x}" for element in group_gen_repr]
    group_gen_inv = mod_inverse(group_gen,modulus)
    group_gen_inv_r = (group_gen_inv * (2**bits)) % modulus
    group_gen_inv_repr = into_repr(group_gen_inv_r,limbs)
    group_gen_inv_repr = [f"0x{element:08x}" for element in group_gen_inv_repr]
    closest_power = closest_power_of_two(bits)
    Rx = (modulus<<(bits- actualbits))
    Rx_repr = into_repr_64(Rx,limbs)
    Rx_repr = [f"0x{element:016x}" for element in Rx_repr]
    
    print("modulus: ",modulus_repr)
    print("===========================\n")

    print("R: ",R_repr)
    print("===========================\n")

    print("m0: ",N_INV_repr)
    print("===========================\n")

    print("R2: ",R2_repr)
    print("===========================\n")

    print("Rx: ",Rx_repr)
    print("===========================\n")

    print("group_gen: ",group_gen_repr)
    print("===========================\n")

    print("group_gen_inv: ",group_gen_inv_repr)
    print("===========================\n")

    rp_repr = into_repr(root_of_unity,limbs)
    rp = [[] for _ in range(S+1)]
    rp [-1] = [f"0x{element:08x}" for element in rp_repr]
    

    root_of_unity_inv = mod_inverse(root_of_unity,modulus)
    root_of_unity_inv_r = (root_of_unity_inv*(2**(bits*2)))%modulus
    rp_inv_repr = into_repr(root_of_unity_inv_r,limbs)
    rp_inv = [[] for _ in range(S+1)]
    rp_inv [-1] = [f"0x{element:08x}" for element in rp_inv_repr]

    rp_gen = root_of_unity
    for i in range(S):
        rp_gen = square(rp_gen, R_INV, modulus)
        rp_gen_repr = into_repr(rp_gen,limbs)
        rp[S-i-1] = [f"0x{element:08x}" for element in rp_gen_repr]
        rp_inv_gen = mod_inverse(rp_gen,modulus)
        rp_inv_gen_r = (rp_inv_gen*(2**(bits*2)))%modulus
        rp_inv_gen_repr = into_repr(rp_inv_gen_r,limbs)
        rp_inv[S-i-1] = [f"0x{element:08x}" for element in rp_inv_gen_repr]
    R_repr = into_repr(R,limbs)
    R_repr = [f"0x{element:08x}" for element in R_repr]

    for i in range(S+1):
        print(rp[i])
    print("===========================\n")
    for i in range(S+1):
        print(rp_inv[i])
    print("===========================\n")

    domain_size_inv = [[] for _ in range(S+1)]
    for i in range(S+1):
        size = 2**i 
        size_r = (size * R) % modulus
        size_inv = mod_inverse(size_r,modulus)
        size_inv = (size_inv * (2**(bits*2)))% modulus
        size_inv_repr = into_repr(size_inv,limbs)
        domain_size_inv[i] = [f"0x{element:08x}" for element in size_inv_repr]
    for i in range(S+1):
        print(domain_size_inv[i])
    
