import copy
from .bls12_381 import fr,fq
from .structure import AffinePointG1
from .jacobian import ProjectivePointG1
import math
import random
import torch
import torch.nn.functional as F
import time
import torch.nn as nn
import torch


def extend_tensor(input:torch.tensor,size):
    res = torch.zeros(size, 4, dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in range(len(res)):
        res[i] = input
    return res.to('cuda')


def calculate_execution_time(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()  # 记录函数开始执行的时间
            result = func(*args, **kwargs)
            end_time = time.time()    # 记录函数执行结束的时间
            execution_time = end_time - start_time  # 计算函数执行时间
            print(f"func {func.__name__} consumed: {execution_time} s")
            return result
        return wrapper

def neg(self):
    a=torch.tensor([18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    if F.trace_equal(self,torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')):
        return self
    else:
        res= F.sub_mod(a,self)
        return res
    
def neg_extend(self,size):
    res=torch.zeros(size,4,dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    ### 把是零的位置记下来然后先整体做sub再重新赋值0
    one=torch.tensor([18446744069414584321, 6034159408538082302, 3691218898639771653, 8353516859464449352],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    extend_one=extend_tensor(one,size)
    record=[]
    for i in range(len(self)):
        if F.trace_equal(self[i],torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')):
            record.append(i)
    res=F.sub_mod(extend_one,self)
    #
    for i in record:
        res[i]=self[i]

    return res 

def neg_fq(self):
    # a is fq modualr
    a=torch.tensor([13402431016077863595, 2210141511517208575, 7435674573564081700, 7239337960414712511, 5412103778470702295, 1873798617647539866],dtype=torch.BLS12_381_Fq_G1_Mont)
    if F.trace_equal(self,torch.tensor([0,0,0,0,0,0],dtype=torch.BLS12_381_Fq_G1_Mont)):
        return self
    else:
        res= F.sub_mod(a,self)
        return res

def from_list_tensor(input:list, dtype=torch.BLS12_381_Fr_G1_Mont):
    base_input=[]
    for i in range(len(input)):
        # print(input[i].value)
        base_input.append(input[i].value)
    output = torch.tensor(base_input,dtype = dtype,device='cpu')
    return output

def from_tensor_list(input:torch.Tensor):
    output = input.tolist()
    # print("output 值为",output)
    for i in range(len(input)):
        output[i]=fr.Fr(value=output[i])
    return output

def tensor_to_int(input:torch.Tensor):
    list = input.tolist()
    output = 0 
    for i in reversed(list):
        output = output << 64
        output = output | i
    return output

# def from_list_gmpy(input:list):
#     for i in range(len(input)):
#         output = 0 
#         for j in reversed(input[i].value):
#             output = output<<64
#             output = output | j
#         input[i] =  fr.Fr(value = gmpy2.mpz(output))

def from_gmpy_list(input:list): ##inplace
    if(len(input)==0):
        input=torch.tensor([],dtype=torch.uint64)
    for i in range(len(input)):
        output = []
        for j in range(fr.Fr.Limbs):
            output.append( int(input[i].value & 0xFFFFFFFFFFFFFFFF) )
            input[i].value = input[i].value >> 64
        input[i] = fr.Fr(value = output)

def from_gmpy_list_1(input:fr.Fr):
    output = []
    input_copy=copy.deepcopy(input)
    for j in range(fr.Fr.Limbs):
        output.append( int(input_copy.value & 0xFFFFFFFFFFFFFFFF) )
        input_copy.value = input_copy.value >> 64
    # input=fr.Fr(value=output)
    return output

def challenge_to_tensor(input:fr.Fr):
    output = []
    input_copy=copy.deepcopy(input)
    for j in range(fr.Fr.Limbs):
        output.append( int(input_copy.value & 0xFFFFFFFFFFFFFFFF) )
        input_copy.value = input_copy.value >> 64
    output=torch.tensor(output,dtype=torch.BLS12_381_Fr_G1_Mont)
    return output

def domain_trans_tensor(domain_ele):
        a=copy.deepcopy(domain_ele)
        temp=from_gmpy_list_1(a)
        xx = torch.tensor(temp, dtype=torch.BLS12_381_Fr_G1_Mont)
        # domain_ele.value=xx
        return xx

def extend_tensor(input:torch.Tensor,size):
    if size == 0:
        return input
    res = torch.zeros(size, 4, dtype=torch.BLS12_381_Fr_G1_Mont)
    for i in range(len(res)):
        res[i] = input
    return res.to('cuda')

def pow(self, exp):
    res = self.clone()
    for i in range(exp-1):
        res = F.mul_mod(res, self)
    return res

def pow_single(self, exp):
    res = fr.Fr.one()
    if self.is_cuda:
        res = res.to("cuda")
    for i in range(exp):
        res = F.mul_mod(res, self)
    return res

# def pow_2(self,exp):
#     res = ONE
#     for i in range(63,-1,-1):
#         #modsquare
#         res = F.mul_mod(res,res)
#         if ((exp >> i) & 1) == 1:
#             # res = res.mul(self)
#             res = F.mul_mod(res,self)
#     return res

def reverse_bits(operand, bit_count):
    acc = 0
    for i in range(bit_count):
        acc = (acc << 1) | ((operand >> i) & 1)
    return acc

def derange(xi, log_len):
    for idx in range(1, len(xi) - 1):
        ridx = reverse_bits(idx, log_len)
        if idx < ridx:
            xi[idx], xi[ridx] = xi[ridx], xi[idx]
    return xi

# def precompute_twiddles(domain:Radix2EvaluationDomain, root:fr.Fr):
#     log_size = int(math.log2(domain.size))
#     # powers = [root.zero()] * (1 << (log_size - 1))
#     # powers[0] = root.one()
#     powers=torch.zeros(1 << (log_size - 1),4,dtype=torch.BLS12_381_Fr_G1_Mont)
#     powers[0]=torch.tensor(from_gmpy_list_1(root.one()),dtype=torch.BLS12_381_Fr_G1_Mont)

#     for idx in range(1, len(powers)):
#         # powers[idx] = powers[idx - 1].mul(root.value)
#         powers[idx]=F.mul_mod(powers[idx - 1],root.value)
#     return powers

# def precompute_twiddles(domain:Radix2EvaluationDomain, root:fr.Fr):
#     log_size = int(math.log2(domain.size))
#     powers = [root.zero()] * (1 << (log_size - 1))
#     powers[0] = root.one()
#     for idx in range(1, len(powers)):
#         powers[idx] = powers[idx - 1].mul(root)
#     return powers

# def  operator(domain:Radix2EvaluationDomain, xi:torch.tensor, root:fr.Fr):
#     log_size = int(math.log2(domain.size))
#     xi=from_tensor_list(xi)
#     from_list_gmpy(xi)
#     xi = derange(xi,log_size)
#     twiddles=precompute_twiddles(domain,root)
#     chunk = 2
#     twiddle_chunk = domain.size // 2

#     twiddles=from_tensor_list(twiddles)
#     from_list_gmpy(twiddles)
#     for i in range(log_size):
#         for j in range(0, domain.size, chunk):
#             t = xi[j + chunk // 2]       # Copy Right[0]
#             xi[j + chunk // 2] = xi[j]   # Right[0] = Left[0]
#             xi[j] = xi[j].add(t)
#             xi[j + chunk // 2] = xi[j + chunk // 2].sub(t)
#             for m in range(chunk // 2 - 1):
#                 twiddle = twiddles[(m + 1) * twiddle_chunk]
#                 t1 = xi[j + chunk // 2 + m + 1]
#                 t1 = t1.mul(twiddle)
#                 xi[j + chunk // 2 + m + 1] = xi[j + m + 1]
#                 xi[j + m + 1] = xi[j + m + 1].add(t1)  # a + b * w
#                 # a - b * w
#                 xi[j + chunk // 2 + m + 1] = xi[j + chunk // 2 + m + 1].sub(t1)  
#         chunk *= 2  # Merge up
#         twiddle_chunk //= 2

#     # for i in range(log_size):
#     #     for j in range(0, domain.size, chunk):
#     #         t = xi[j + chunk // 2]       # Copy Right[0]
#     #         xi[j + chunk // 2] = xi[j]   # Right[0] = Left[0]
#     #         xi[j] = F.add_mod(xi[j].clone(), t.clone())
#     #         xi[j + chunk // 2] = F.sub_mod(xi[j + chunk // 2].clone(), t.clone())
#     #         for m in range(chunk // 2 - 1):
#     #             twiddle = twiddles[(m + 1) * twiddle_chunk]
#     #             t1 = xi[j + chunk // 2 + m + 1]
#     #             t1 = F.mul_mod(t1.clone(), twiddle.clone())
#     #             xi[j + chunk // 2 + m + 1] = xi[j + m + 1]
#     #             xi[j + m + 1] = F.add_mod(xi[j + m + 1].clone(), t1.clone())
#     #             xi[j + chunk // 2 + m + 1] = F.sub_mod(xi[j + chunk // 2 + m + 1].clone(), t1.clone())
#     #     chunk *= 2  # Merge up
#     #     twiddle_chunk //= 2
#     from_gmpy_list(xi)
#     xi=from_list_tensor(xi)
#     return xi

# def operator(domain, xi:list[fr.Fr], root:fr.Fr):
#     log_size = int(math.log2(domain.size))
#     xi = derange(xi,log_size)
#     twiddles=precompute_twiddles(domain,root)
#     chunk = 2
#     twiddle_chunk = domain.size // 2
#     for i in range(log_size):
#         for j in range(0, domain.size, chunk):
#             t = xi[j + chunk // 2]       # Copy Right[0]
#             xi[j + chunk // 2] = xi[j]   # Right[0] = Left[0]
#             xi[j] = xi[j].add(t)
#             xi[j + chunk // 2] = xi[j + chunk // 2].sub(t)
#             for m in range(chunk // 2 - 1):
#                 twiddle = twiddles[(m + 1) * twiddle_chunk]
#                 t1 = xi[j + chunk // 2 + m + 1]
#                 t1 = t1.mul(twiddle)
#                 xi[j + chunk // 2 + m + 1] = xi[j + m + 1]
#                 xi[j + m + 1] = xi[j + m + 1].add(t1)  # a + b * w
#                 # a - b * w
#                 xi[j + chunk // 2 + m + 1] = xi[j + chunk // 2 + m + 1].sub(t1)  
#         chunk *= 2  # Merge up
#         twiddle_chunk //= 2
#     return xi


# def resize(self, target_len, padding):
#     res = self[:]
#     if len(self) < target_len:
#         num_to_pad = target_len - len(self)
#         res.extend([padding for _ in range(num_to_pad)])
#     return res

def resize_gpu(self: torch.Tensor, target_len):
    res = self.clone()
    if res.size(0) < target_len:
        padding = torch.zeros((target_len - res.size(0), fr.Fr.Limbs), dtype = fr.Fr.Dtype).to("cuda")
        res = torch.cat((res, padding))
    return res

def resize_cpu(self: torch.Tensor, target_len):
    res = self.clone()
    if res.size(0) < target_len:
        padding = torch.zeros((target_len - res.size(0), fr.Fr.Limbs), dtype = fr.Fr.Dtype)
        res = torch.cat((res, padding))
    return res

# def resize(self, target_len, padding):
#     res = copy.deepcopy(self)
#     if len(self) < target_len:
#         num_to_pad = target_len - len(self)
#         res.extend([padding for _ in range(num_to_pad)])
#     return res


def distribute_powers(coeffs, g):
    g_field = fr.Fr(value = g)
    one = g_field.one()
    distribute_powers_and_mul_by_const(coeffs, g_field, one)

# Multiply the `i`-th element of `coeffs` with `c*g^i`.
def distribute_powers_and_mul_by_const(coeffs, g, c):
    pow = c
    for i in range(len(coeffs)):
        coeffs[i] = coeffs[i].mul(pow)
        pow = pow.mul(g)
def degree(poly):
    if len(poly)==0:
        return 0
    else:
        assert not poly[-1]==0
        return len(poly) - 1

def convert_to_bigints(p: torch.Tensor):
    if p.size(0) == 0:
        return torch.tensor([],dtype = fr.Fr.Base_type)
    else:
        res = F.to_base(p)
        return res
    
# def convert_to_bigints(p: list[fr.Fr]):
#     coeffs = [fr.Fr(value = s.into_repr()) for s in p]
#     return coeffs


# def skip_leading_zeros_and_convert_to_bigints(p: list[fr.Fr]):
#     num_leading_zeros = 0
#     while num_leading_zeros < len(p) and p[num_leading_zeros] == 0:
#         num_leading_zeros += 1

#     coeffs = convert_to_bigints(p[num_leading_zeros:])
#     return num_leading_zeros, coeffs

def skip_leading_zeros_and_convert_to_bigints(p: torch.Tensor):
    # p = p.to("cpu")
    # num_leading_zeros = 0
    # zero = fr.Fr.zero()
    # while num_leading_zeros < p.size(0) and torch.equal(p[num_leading_zeros], zero):
    #     num_leading_zeros += 1
    # coeffs = convert_to_bigints(p[num_leading_zeros:])  
    return 0, convert_to_bigints(p.to('cpu'))
    return num_leading_zeros, coeffs

def distribute_powers_new(coeffs, g):

    g=torch.tensor([64424509425, 1721329240476523535, 18418692815241631664, 3824455624000121028],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    one = torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    return distribute_powers_and_mul_by_const_new(coeffs, g, one)

# Multiply the `i`-th element of `coeffs` with `c*g^i`.
def distribute_powers_and_mul_by_const_new(coeffs, g, c):
    pow = c
    for i in range(len(coeffs)):
        coeffs[i] = F.mul_mod(coeffs[i],pow)
        pow= F.mul_mod(pow,g)
    return coeffs

def INTT(size, evals: torch.Tensor):
    inttclass = nn.Intt(fr.Fr.TWO_ADICITY, fr.Fr.Dtype)
    evals_resize = resize_gpu(evals, size)
    res = inttclass.forward(evals_resize)
    return res

def NTT(size, evals:torch.Tensor):
    nttclass = nn.Ntt(fr.Fr.TWO_ADICITY, fr.Fr.Dtype)
    evals_resize=resize_gpu(evals,size)
    res= nttclass.forward(evals_resize)
    return res

def coset_NTT(size, coeffs:torch.Tensor):
    ntt_coset_class = nn.Ntt_coset(fr.Fr.TWO_ADICITY, fr.Fr.Dtype)
    resize_coeffs= resize_gpu(coeffs, size)
    evals = ntt_coset_class.forward(resize_coeffs)
    return evals


def coset_INTT(size, evals:torch.tensor):
    ntt_class = nn.Intt_coset(fr.Fr.TWO_ADICITY, fr.Fr.Dtype)
    resize_evals= resize_gpu(evals, size)
    coeffs = ntt_class.forward(resize_evals)
    return coeffs


# Compute a NTT over a coset of the domain, modifying the input vector in place.
# def coset_NTT(coeffs:torch.tensor, domain):

#     coeffs=from_tensor_list(coeffs)
#     from_list_gmpy(coeffs)

#     modified_coeffs = coeffs[:]
#     distribute_powers(modified_coeffs, fr.Fr.GENERATOR)
#     from_gmpy_list(modified_coeffs)
#     modified_coeffs=from_list_tensor(modified_coeffs)
#     evals = NTT(domain,modified_coeffs)
#     return evals

# Compute a INTT over a coset of the domain, modifying the input vector in place.
# def coset_INTT(evals:torch.tensor, domain):
#     evals=from_tensor_list(evals)
#     from_list_gmpy(evals)
#     #add zero to resize
#     zero = fr.Fr.zero()
#     resize_evals = resize(evals,domain.size,zero)
#     evals = operator(domain,resize_evals,domain.group_gen_inv)
#     distribute_powers_and_mul_by_const(evals, domain.generator_inv,domain.size_inv)
#     from_gmpy_list(evals)
#     evals=from_list_tensor(evals)
#     return evals

def from_coeff_vec(poly:torch.Tensor):
    return poly
    poly1 = poly.to("cpu")
    zero = fr.Fr.zero()
    counter = 0
    while counter < poly1.size(0) and torch.equal(poly1[-counter-1], zero):
        counter += 1
    return poly[:poly.size(0)-counter]


def poly_add_poly(self: torch.tensor, other: torch.tensor):    #input tensor output tensor
    if self.size(0) == 0:
        res = other[:]
        return res
    if other.size(0) == 0:
        res =self[:]
        return res
    elif len(self) >= len(other):
        result = self.clone()
        F.add_mod(result[:len(other)],other,True)
        result = from_coeff_vec(result)
        return result.to("cuda")
    else:
        result = other.clone()
        F.add_mod(result[:len(self)],self,True)
        result = from_coeff_vec(result)
        return result.to("cuda")

def poly_mul_const(poly:torch.Tensor, elem:torch.Tensor):  
    if poly.size(0) == 0 :
        return poly
    else:
        result = F.mul_mod_scalar(poly,elem)
        return result

# def divide_with_q_and_r(self: torch.Tensor, divisor: torch.Tensor):
#     if self.size(0) == 0:
#         return self
#     elif self.size(0) < divisor.size(0):
#         zero = fr.Fr.zero().value
#         return zero
#     else:
#         one = fr.Fr.one().value
#         quotient = torch.zeros(self.size(0) - divisor.size(0) + 1, fr.Fr.Limbs, dtype = fr.Fr.Dtype)
#         remainder = self

#         divisor_leading = divisor[-1].clone()
#         divisor_leading_inv = F.div_mod(one, divisor_leading)
#         while remainder.size(0) != 0 and remainder.size(0) >= divisor.size(0):
#             remainder_leading = remainder[-1]
#             cur_q_coeff = F.mul_mod(remainder_leading, divisor_leading_inv)
#             cur_q_degree = remainder.size(0) - divisor.size(0)
#             quotient[cur_q_degree] = cur_q_coeff
#             for i, div_coeff in enumerate(divisor):
#                 temp = F.mul_mod(cur_q_coeff, div_coeff)
#                 remainder[cur_q_degree + i] = F.sub_mod(remainder[cur_q_degree + i], temp)

#             i = 0
#             while F.trace_equal(remainder[-1 - i], fr.Fr.zero().value):
#                 i = i+1
#             remainder = remainder[:remainder.size(0) - i]
#         res_quotient = from_coeff_vec(quotient)
#         return res_quotient, remainder
            
# def poly_div_poly(self: torch.Tensor, divisor: torch.Tensor):
#         res, remainder = divide_with_q_and_r(self, divisor)
#         return res

def rand_poly(d):
    random.seed(42)
    random_coeffs = [fr.Fr.make_tensor(random.random) for _ in range(d + 1)]
    return from_coeff_vec(random_coeffs)


# Evaluates `self` at the given `point` in `Self::Point`.
# def evaluate(self, point: fr.Fr):
#     zero =torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)
#     if point.is_cuda:
#         point=point.to('cpu')

#     if len(self) == 0:
#         return zero.to('cuda')
#     elif F.trace_equal(point,zero):
#         return self[0]
#     return horner_evaluate(self, point.to('cuda'))

# Horner's method for polynomial evaluation
# def horner_evaluate(poly_coeffs: list, point: fr.Fr):

#     result =torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
#     repoly_coeffs=copy.deepcopy(poly_coeffs)
#     repoly_coeffs=repoly_coeffs.to('cpu')
#     repoly_coeffs=reversed(repoly_coeffs)
#     repoly_coeffs=repoly_coeffs.to('cuda')
#     for coeff in repoly_coeffs:
#         result = F.mul_mod(result, point)
#         result = F.add_mod(result, coeff)
#     return result

def poly_add_poly_mul_const(self:torch.Tensor, f: torch.Tensor, other: torch.Tensor):
    if self.size(0) == 0 and other.size(0) == 0:
        return torch.tensor([], dtype = self.dtype, device = self.device)

    if self.size(0) == 0:
        other = other.to('cuda')
        self = other.clone()
        self = F.mul_mod_scalar(self, f)
        return self
    elif other.size(0) == 0:
        self = self.to('cuda')
        return self
    elif self.size(0) >= other.size(0):
        self = self.to('cuda')
        other = other.to('cuda')
    else:
        self = self.to('cuda')
        other = other.to('cuda')
        self = resize_gpu(self, other.size(0))
    temp = F.mul_mod_scalar(other, f)
    F.add_mod(self[:other.size(0)], temp, True)

    res = from_coeff_vec(self)
    
    return res


# Given a vector of field elements {v_i}, compute the vector {coeff * v_i^(-1)}
def batch_inversion_and_mul(v, coeff):
    v_inv = F.inv_mod(v)
    res = F.mul_mod_scalar(v_inv, coeff)
    return res

# Given a vector of field elements {v_i}, compute the vector {v_i^(-1)}
def batch_inversion(v):
    one = fr.Fr.one()
    res = batch_inversion_and_mul(v, one.to("cuda"))
    return res
# The first lagrange polynomial has the expression:
# L_0(X) = mul_from_1_to_(n-1) [(X - omega^i) / (1 - omega^i)]
#
# with `omega` being the generator of the domain (the `n`th root of unity).
#
# We use two equalities:
#   1. `mul_from_2_to_(n-1) [1 / (1 - omega^i)] = 1 / n`
#   2. `mul_from_2_to_(n-1) [(X - omega^i)] = (X^n - 1) / (X - 1)`
# to obtain the expression:
# L_0(X) = (X^n - 1) / n * (X - 1)
def compute_first_lagrange_evaluation(size, z_h_eval, z_challenge):
    # single scalar OP on CPU
    one = fr.Fr.one()
    n_fr = fr.Fr.make_tensor(size)
    z_challenge_sub_one = F.sub_mod(z_challenge, one)
    denom = F.mul_mod(n_fr, z_challenge_sub_one)
    denom_in = F.div_mod(one, denom)
    res = F.mul_mod(z_h_eval, denom_in)
    return res  

# def MSM(bases:list[AffinePointG1], scalars:list[fr.Fr], params):
#     size = min(len(bases), len(scalars))
#     fq_elem = bases[0].x
#     scalars = scalars[:size]
#     bases = bases[:size]
#     scalars_and_bases_iter = [(s, b) for s, b in zip(scalars, bases) if not s==0]

#     c = 3 if size < 32 else ln_without_floats(size) + 2
#     num_bits = fr.Fr.MODULUS_BITS
#     fr_one = params.one()
#     fr_one = fr_one.into_repr()

#     zero:ProjectivePointG1 = ProjectivePointG1.zero(fq_elem)
#     window_starts = list(range(0, num_bits, c))

#     window_sums = []
#     for w_start in window_starts:
#         res = zero
#         buckets = [zero for _ in range((1 << c) - 1)]

#         for org_scalar, org_base in scalars_and_bases_iter:
#             scalar = copy.copy(org_scalar)
#             base = copy.copy(org_base)
#             if scalar.value == fr_one:
#                 if w_start == 0:
#                     res = res.add_assign_mixed(base)
#             else:
#                 # We right-shift by w_start, thus getting rid of the lower bits
#                 scalar.value >>= w_start
#                 # We mod the remaining bits by 2^{window size}, thus taking `c` bits.
#                 scalar.value %= (1 << c)
#                 if scalar.value != 0:
#                     buckets[scalar.value - 1] = buckets[scalar.value - 1].add_assign_mixed(base)

#         running_sum:ProjectivePointG1 = ProjectivePointG1.zero(fq_elem)
#         for b in reversed(buckets):
#             running_sum = running_sum.add_assign(b)
#             res = res.add_assign(running_sum)

#         window_sums.append(res)

#     lowest = window_sums[0]

#     total:ProjectivePointG1 = zero
#     for sum_i in reversed(window_sums[1:]):
#         total = total.add_assign(sum_i)
#         for _ in range(c):
#             total = total.double()
#     total = lowest.add_assign(total)
#     return total

def MSM_new(bases,scalar): #bases POINT scalar SCALAR
    min_size_1 = min(bases.size(0)//2, scalar.size(0))
    
    if min_size_1==0:### empty msm return zero_point
        res =[[],[],[]]
        res[2] = fq.zero()
        res[1] = fq.one()
        res[0] = fq.one()
        commitment=ProjectivePointG1(res[0],res[1],res[2])
        return commitment
    else:
        base = bases.clone()
        base = base[:min_size_1].view(-1, 6) # dim2 to 1
        base = base.to('cuda')
        scalar = scalar.to('cuda')
        commitment = F.multi_scalar_mult(base, scalar)
        commitment = ProjectivePointG1(commitment[0],commitment[1],commitment[2])
        return commitment