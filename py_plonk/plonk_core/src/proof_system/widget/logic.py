from dataclasses import dataclass
from .....bls12_381 import fr
from .....plonk_core.src.proof_system.widget.mod import WitnessValues,delta
from .arithmetic import poly_mul_const
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
    def from_evaluations(custom_evals):
        a_next_val = custom_evals["a_next_eval"]
        b_next_val = custom_evals["b_next_eval"]
        d_next_val = custom_evals["d_next_eval"]
        q_c_val = custom_evals["q_c_eval"]
        return LogicValues(a_next_val,b_next_val,d_next_val,q_c_val)
    
class LogicGate:
    @staticmethod
    def constraints(separation_challenge:fr.Fr, wit_vals:WitnessValues, custom_vals:LogicValues):

        four = fr.make_tensor(4)
        four = four.to("cuda")
        kappa = F.mul_mod(separation_challenge,separation_challenge)
        kappa_sq =F.mul_mod(kappa,kappa)
        kappa_cu = F.mul_mod(kappa_sq,kappa)
        kappa_qu = F.mul_mod(kappa_cu,kappa)

        a_1 = F.mul_mod(four, wit_vals.a_val)
        a = F.sub_mod(custom_vals.a_next_val, a_1)
        c_0 = delta(a)

        b_1 = F.mul_mod(four, wit_vals.b_val)
        b = F.sub_mod(custom_vals.b_next_val, b_1)
        c_1 = delta(b)

        d_1 = F.mul_mod(four, wit_vals.d_val)
        d = F.sub_mod(custom_vals.d_next_val, d_1)
        c_2 = delta(d)

        w = wit_vals.c_val
        w_1 = F.mul_mod(a, b)
        w_2 = F.sub_mod(w, w_1)
        c_3 = F.mul_mod(w_2, kappa_cu)

        c_4_1 = delta_xor_and(a, b, w, d, custom_vals.q_c_val)
        c_4 = F.mul_mod(c_4_1, kappa_qu)

        mid1 = F.add_mod(c_0, c_1)
        mid2 = F.add_mod(mid1, c_2)
        mid3 = F.add_mod(mid2, c_3)
        mid4 = F.add_mod(mid3, c_4)
        res = F.mul_mod(mid4, separation_challenge)

        return res

    
    @staticmethod
    def quotient_term(selector: torch.Tensor, separation_challenge: torch.Tensor, 
                      wit_vals: WitnessValues, custom_vals:LogicValues):
        four = fr.make_tensor(4)
        four = four.to('cuda')

        # single scalar OP on CPU
        kappa = F.mul_mod(separation_challenge, separation_challenge)
        kappa_sq = F.mul_mod(kappa, kappa)
        kappa_cu = F.mul_mod(kappa_sq, kappa)
        kappa_qu = F.mul_mod(kappa_cu, kappa)

        a_1 = F.mul_mod_scalar(wit_vals.a_val, four)
        a = F.sub_mod(custom_vals.a_next_val, a_1)
        c_0 = delta(a)

        b_1 = F.mul_mod_scalar(wit_vals.b_val, four)
        b = F.sub_mod(custom_vals.b_next_val, b_1)
        c_1 = delta(b)

        d_1 = F.mul_mod_scalar(wit_vals.d_val, four)
        d = F.sub_mod(custom_vals.d_next_val, d_1)
        c_2 = delta(d)

        w = wit_vals.c_val
        w_1 = F.mul_mod(a, b)
        w_2 = F.sub_mod(w, w_1)
        c_3 = F.mul_mod_scalar(w_2, kappa_cu.to("cuda"))

        c_4_1 = delta_xor_and(a, b, w, d, custom_vals.q_c_val)
        c_4 = F.mul_mod_scalar(c_4_1, kappa_qu.to("cuda"))

        mid1 = F.add_mod(c_0, c_1)
        mid2 = F.add_mod(mid1, c_2)
        mid3 = F.add_mod(mid2, c_3)
        mid4 = F.add_mod(mid3, c_4)
        temp = F.mul_mod_scalar(mid4, separation_challenge.to("cuda"))

        res= F.mul_mod(selector, temp)
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
def delta_xor_and(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, c: torch.Tensor, q_c: torch.Tensor):

    nine = fr.make_tensor(9).to("cuda")
    two = fr.make_tensor(2).to("cuda")
    three = fr.make_tensor(3).to("cuda")
    four = fr.make_tensor(4).to("cuda")
    eighteen = fr.make_tensor(18).to("cuda")
    eighty_one = fr.make_tensor(81).to("cuda")
    eighty_three = fr.make_tensor(83).to("cuda")


    f_1_1 = F.mul_mod_scalar(w, four)
    f_1_2_1 = F.add_mod(a, b)
    f_1_2 = F.mul_mod_scalar(f_1_2_1, eighteen)
    f_1 = F.sub_mod(f_1_1, f_1_2)
    f_1 = F.add_mod_scalar(f_1, eighty_one)
    f_1 = F.mul_mod(f_1, w)

    f_2_1_1 = F.mul_mod(a,a)
    f_2_1_2 = F.mul_mod(a,a)
    f_2_1 = F.add_mod(f_2_1_1, f_2_1_2)
    f_2 = F.mul_mod_scalar(f_2_1, eighteen)

    f_3_1 = F.add_mod(a, b)
    f_3 = F.mul_mod_scalar(f_3_1, eighty_one)

    f = F.add_mod(f_1, f_2)
    f = F.sub_mod(f, f_3)
    f = F.add_mod_scalar(f, eighty_three)
    f = F.mul_mod(w, f)

    e_1_1 = F.add_mod(f_3_1, c)
    e_1 = F.mul_mod_scalar(e_1_1, three)
    e_2 = F.mul_mod_scalar(f, two)
    e = F.sub_mod(e_1, e_2)

    b_1_1 = F.mul_mod_scalar(c, nine)
    b_1_2 = F.mul_mod_scalar(f_3_1, three)
    b_1 = F.sub_mod(b_1_1, b_1_2)
    b = F.mul_mod(q_c, b_1)

    res = F.add_mod(b, e)
    return res


