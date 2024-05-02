from .....bls12_381 import fr
from .....plonk_core.src.proof_system.mod import CustomEvaluations
from .....plonk_core.src.proof_system.widget.mod import WitnessValues,delta
from .arithmetic import poly_mul_const
from .....arithmetic import from_gmpy_list_1,extend_tensor
import torch.nn.functional as F
import torch


class RangeValues:
    def __init__(self, d_next_val:fr.Fr):
        self.d_next_val = d_next_val
        
    @staticmethod
    def from_evaluations(custom_vals:CustomEvaluations):
        d_next_val = custom_vals.get("d_next_eval")
        return RangeValues(d_next_val)

class RangeGate:
    @staticmethod
    def constraints(four,separation_challenge:fr.Fr, wit_vals:WitnessValues, custom_vals:RangeValues):
       
     
        kappa = F.mul_mod(separation_challenge,separation_challenge)
        kappa_sq = F.mul_mod(kappa,kappa)
        kappa_cu = F.mul_mod(kappa_sq,kappa)

        b_1_1 = F.mul_mod(four, wit_vals.d_val)
        f_b1= F.sub_mod(wit_vals.c_val,b_1_1)
        b_1 = delta(f_b1,0)

        b_2_1 = F.mul_mod(four, wit_vals.c_val)
        b_2_2 = F.sub_mod(wit_vals.b_val,b_2_1)
        f_b2 = delta(b_2_2,0)
        b_2 = F.mul_mod(f_b2, kappa)

        b_3_1 = F.mul_mod(four, wit_vals.b_val)
        b_3_2 = F.sub_mod(wit_vals.a_val,b_3_1)
        f_b3 = delta(b_3_2,0)
        b_3 = F.mul_mod(f_b3, kappa_sq)

        b_4_1 = F.mul_mod(four, wit_vals.a_val)
        b_4_2 = F.sub_mod(custom_vals.d_next_val,b_4_1)
        f_b4 = delta(b_4_2,0)
        b_4 = F.mul_mod(f_b4, kappa_cu)

        mid1 = F.add_mod(b_1, b_2)
        mid2 = F.add_mod(mid1, b_3)
        mid3 = F.add_mod(mid2, b_4)
        res = F.mul_mod(mid3, separation_challenge)

        return res
    
    @staticmethod
    def quotient_term(four,selector, separation_challenge, wit_vals, custom_vals,size):
        
       
        separation_challenge = extend_tensor(separation_challenge,size)
        kappa = F.mul_mod(separation_challenge,separation_challenge)
       
        kappa_sq = F.mul_mod(kappa,kappa)
        kappa_cu = F.mul_mod(kappa_sq,kappa)

        

        b_1_1 = F.mul_mod(four, wit_vals.d_val)
        f_b1= F.sub_mod(wit_vals.c_val,b_1_1)
        b_1 = delta(f_b1,size)

        b_2_1 = F.mul_mod(four, wit_vals.c_val)
        b_2_2 = F.sub_mod(wit_vals.b_val,b_2_1)
        f_b2 = delta(b_2_2,size)
        b_2 = F.mul_mod(f_b2, kappa)

        b_3_1 = F.mul_mod(four, wit_vals.b_val)
        b_3_2 = F.sub_mod(wit_vals.a_val,b_3_1)
        f_b3 = delta(b_3_2,size)
        b_3 = F.mul_mod(f_b3, kappa_sq)

        b_4_1 = F.mul_mod(four, wit_vals.a_val)
        b_4_2 = F.sub_mod(custom_vals.d_next_val,b_4_1)
        f_b4 = delta(b_4_2,size)
        b_4 = F.mul_mod(f_b4, kappa_cu)

        mid1 = F.add_mod(b_1, b_2)
        mid2 = F.add_mod(mid1, b_3)
        mid3 = F.add_mod(mid2, b_4)
        temp = F.mul_mod(mid3, separation_challenge)

        res = F.mul_mod(selector,temp)
        return res
    
    @staticmethod
    def linearisation_term(four,selector_poly, separation_challenge, wit_vals, custom_vals):
        temp = RangeGate.constraints(four,separation_challenge, wit_vals, custom_vals)
        res = poly_mul_const(selector_poly,temp)
        return res