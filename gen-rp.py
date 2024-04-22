import gmpy2

def into_repr(input, limbs):
    output = []
    for i in range(limbs):
        x = input & 0xffffffff
        input = input >> 32
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

if __name__ == "__main__":
    bits = 768
    limbs = bits//32
    S = 30
    modulus = 41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888458477323173057491593855069696241854796396165721416325350064441470418137846398469611935719059908164220784476160001
    root_of_unity = 5431548564651772770863376209190533321743766006080874345421017090576169920304713950094628043692772801995471539849411522704471393987882883355624697206026582300050878644000631322086989454860102191886653186986980927065212650747291
    root_of_unity = root_of_unity *(2**bits) % modulus
    modulus_repr = into_repr(modulus,limbs)
    R = (2**bits) % modulus
    R_INV = mod_inverse(R,modulus)%modulus
    
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
