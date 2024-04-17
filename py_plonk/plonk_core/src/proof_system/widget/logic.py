from dataclasses import dataclass
from .....bls12_381 import fr
from .....plonk_core.src.proof_system.mod import CustomEvaluations
from .....plonk_core.src.proof_system.widget.mod import WitnessValues,delta
from .arithmetic import poly_mul_const
from .....arithmetic import from_gmpy_list_1 ,extend_tensor
import torch.nn.functional as F
import torch
@dataclass
class LogicValues:
    # Left wire value in the next position
    a_next_val: fr.Fr
    # Right wire value in the next position
    b_next_val: fr.Fr
    # Fourth wire value in the next position
    d_next_val: fr.Fr
    # Constant selector value
    q_c_val: fr.Fr

    @staticmethod
    def from_evaluations(custom_evals:CustomEvaluations):
        a_next_val = custom_evals.get("a_next_eval")
        b_next_val = custom_evals.get("b_next_eval")
        d_next_val = custom_evals.get("d_next_eval")
        q_c_val = custom_evals.get("q_c_eval")
        return LogicValues(a_next_val,b_next_val,d_next_val,q_c_val)
    
class LogicGate:
    @staticmethod
    def constraints(separation_challenge:fr.Fr, wit_vals:WitnessValues, custom_vals:LogicValues):
        four = fr.Fr.from_repr(4)
        four=torch.tensor(from_gmpy_list_1(four),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
        kappa= F.mul_mod(separation_challenge,separation_challenge)
        kappa_sq =F.mul_mod(kappa,kappa)
        kappa_cu= F.mul_mod(kappa_sq,kappa)
        kappa_qu = F.mul_mod(kappa_cu,kappa)

        a_1 = F.mul_mod(four, wit_vals.a_val)
        a = F.sub_mod(custom_vals.a_next_val, a_1)
        c_0 = delta(a,0)

        b_1 = F.mul_mod(four, wit_vals.b_val)
        b = F.sub_mod(custom_vals.b_next_val, b_1)
        c_1 = delta(b,0)

        d_1 = F.mul_mod(four, wit_vals.d_val)
        d = F.sub_mod(custom_vals.d_next_val, d_1)
        c_2 = delta(d,0)

        w = wit_vals.c_val
        w_1 = F.mul_mod(a, b)
        w_2 = F.sub_mod(w, w_1)
        c_3 = F.mul_mod(w_2, kappa_cu)

        c_4_1 = delta_xor_and(a, b, w, d, custom_vals.q_c_val,0)
        c_4 = F.mul_mod(c_4_1, kappa_qu)

        mid1 = F.add_mod(c_0, c_1)
        mid2 = F.add_mod(mid1, c_2)
        mid3 = F.add_mod(mid2, c_3)
        mid4 = F.add_mod(mid3, c_4)
        res = F.mul_mod(mid4, separation_challenge)

        return res

    
    @staticmethod
    def quotient_term(four ,selector: fr.Fr, separation_challenge: fr.Fr, 
                      wit_vals: WitnessValues, custom_vals:LogicValues,size):
        separation_challenge=extend_tensor(separation_challenge,size)
        kappa= F.mul_mod(separation_challenge,separation_challenge)
        kappa_sq =F.mul_mod(kappa,kappa)
        kappa_cu= F.mul_mod(kappa_sq,kappa)
        kappa_qu = F.mul_mod(kappa_cu,kappa)

        a_1 = F.mul_mod(four, wit_vals.a_val)
        a = F.sub_mod(custom_vals.a_next_val, a_1)
        c_0 = delta(a,size)

        b_1 = F.mul_mod(four, wit_vals.b_val)
        b = F.sub_mod(custom_vals.b_next_val, b_1)
        c_1 = delta(b,size)

        d_1 = F.mul_mod(four, wit_vals.d_val)
        d = F.sub_mod(custom_vals.d_next_val, d_1)
        c_2 = delta(d,size)

        w = wit_vals.c_val
        w_1 = F.mul_mod(a, b)
        w_2 = F.sub_mod(w, w_1)
        c_3 = F.mul_mod(w_2, kappa_cu)

        c_4_1 = delta_xor_and(a, b, w, d, custom_vals.q_c_val ,size)
        c_4 = F.mul_mod(c_4_1, kappa_qu)

        mid1 = F.add_mod(c_0, c_1)
        mid2 = F.add_mod(mid1, c_2)
        mid3 = F.add_mod(mid2, c_3)
        mid4 = F.add_mod(mid3, c_4)
        temp = F.mul_mod(mid4, separation_challenge)

        res= F.mul_mod(selector,temp)
        return res
    
    @staticmethod
    def linearisation_term(selector_poly, separation_challenge, wit_vals, custom_vals):
        temp = LogicGate.constraints(separation_challenge, wit_vals, custom_vals)
        res = poly_mul_const(selector_poly,temp)
        return res

# The identity we want to check is `q_logic * A = 0` where:
# A = B + E
# B = q_c * [9c - 3(a+b)]
# E = 3(a+b+c) - 2F
# F = w[w(4w - 18(a+b) + 81) + 18(a^2 + b^2) - 81(a+b) + 83]
def delta_xor_and(a: fr.Fr, b: fr.Fr, w: fr.Fr, c: fr.Fr, q_c: fr.Fr,size):
    nine = fr.Fr.from_repr(9)
    two = fr.Fr.from_repr(2)
    three = fr.Fr.from_repr(3)
    four = fr.Fr.from_repr(4)
    eighteen = fr.Fr.from_repr(18)
    eighty_one = fr.Fr.from_repr(81)
    eighty_three = fr.Fr.from_repr(83)


    nine = torch.tensor(from_gmpy_list_1(nine), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    two = torch.tensor(from_gmpy_list_1(two), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    three = torch.tensor(from_gmpy_list_1(three), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    four = torch.tensor(from_gmpy_list_1(four), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    eighteen = torch.tensor(from_gmpy_list_1(eighteen), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    eighty_one = torch.tensor(from_gmpy_list_1(eighty_one), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    eighty_three = torch.tensor(from_gmpy_list_1(eighty_three), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')

    if size ==0 :
        pass

    else :
        nine = extend_tensor(nine, size)
        two = extend_tensor(two, size)
        three = extend_tensor(three, size)
        four = extend_tensor(four, size)
        eighteen = extend_tensor(eighteen, size)
        eighty_one = extend_tensor(eighty_one, size)
        eighty_three = extend_tensor(eighty_three, size)


    f_1_1 = F.mul_mod(four, w)
    f_1_2_1 = F.add_mod(a, b)
    f_1_2 = F.mul_mod(eighteen, f_1_2_1)
    f_1 = F.sub_mod(f_1_1, f_1_2)
    f_1 = F.add_mod(f_1, eighty_one)
    f_1 = F.mul_mod(f_1, w)

    f_2_1_1 = F.mul_mod(a,a)
    f_2_1_2 = F.mul_mod(a,a)
    f_2_1 = F.add_mod(f_2_1_1, f_2_1_2)
    f_2 = F.mul_mod(eighteen, f_2_1)

    f_3_1 = F.add_mod(a, b)
    f_3 = F.mul_mod(eighty_one, f_3_1)

    f = F.add_mod(f_1, f_2)
    f = F.sub_mod(f, f_3)
    f = F.add_mod(f, eighty_three)
    f = F.mul_mod(w, f)

    e_1_1 = F.add_mod(f_3_1, c)
    e_1 = F.mul_mod(three, e_1_1)
    e_2 = F.mul_mod(two, f)
    e = F.sub_mod(e_1, e_2)

    b_1_1 = F.mul_mod(nine, c)
    b_1_2 = F.mul_mod(three, f_3_1)
    b_1 = F.sub_mod(b_1_1, b_1_2)
    b = F.mul_mod(q_c, b_1)

    res = F.add_mod(b, e)
    return res


