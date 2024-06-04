from .....bls12_381 import fr
from .mod import WitnessValues,delta
import torch.nn.functional as F


def _constraints(separation_challenge, wit_vals:WitnessValues, custom_vals):
    four = fr.make_tensor(4)
    four = four.to("cuda")
    kappa = F.mul_mod(separation_challenge, separation_challenge)
    kappa_sq = F.mul_mod(kappa, kappa)
    kappa_cu = F.mul_mod(kappa_sq, kappa)

    b_1_1 = F.mul_mod_scalar(wit_vals.d_val, four)
    f_b1 = F.sub_mod(wit_vals.c_val, b_1_1)
    b_1 = delta(f_b1)

    b_2_1 = F.mul_mod(four, wit_vals.c_val)
    b_2_2 = F.sub_mod(wit_vals.b_val,b_2_1)
    f_b2 = delta(b_2_2)
    b_2 = F.mul_mod(f_b2, kappa)

    b_3_1 = F.mul_mod(four, wit_vals.b_val)
    b_3_2 = F.sub_mod(wit_vals.a_val,b_3_1)
    f_b3 = delta(b_3_2)
    b_3 = F.mul_mod(f_b3, kappa_sq)

    b_4_1 = F.mul_mod(four, wit_vals.a_val)
    b_4_2 = F.sub_mod(custom_vals["d_next_eval"],b_4_1)
    f_b4 = delta(b_4_2)
    b_4 = F.mul_mod(f_b4, kappa_cu)

    mid1 = F.add_mod(b_1, b_2)
    mid2 = F.add_mod(mid1, b_3)
    mid3 = F.add_mod(mid2, b_4)
    res = F.mul_mod(mid3, separation_challenge)

    return res
    
def quotient_term(selector, separation_challenge: fr.Fr, wit_vals, custom_vals):

    four = fr.make_tensor(4)
    four = four.to('cuda')

    #single scalar OP on CPU
    kappa = F.mul_mod(separation_challenge,separation_challenge) 
    kappa_sq = F.mul_mod(kappa,kappa)
    kappa_cu = F.mul_mod(kappa_sq,kappa)

    b_1_1 = F.mul_mod_scalar(wit_vals.d_val,four)
    f_b1 = F.sub_mod(wit_vals.c_val,b_1_1)
    b_1 = delta(f_b1)

    b_2_1 = F.mul_mod_scalar(wit_vals.c_val,four)
    b_2_2 = F.sub_mod(wit_vals.b_val,b_2_1)
    f_b2 = delta(b_2_2)
    b_2 = F.mul_mod_scalar(f_b2, kappa.to("cuda"))

    b_3_1 = F.mul_mod_scalar(wit_vals.b_val,four)
    b_3_2 = F.sub_mod(wit_vals.a_val,b_3_1)
    f_b3 = delta(b_3_2)
    b_3 = F.mul_mod_scalar(f_b3,kappa_sq.to("cuda"))

    b_4_1 = F.mul_mod_scalar(wit_vals.a_val,four)
    b_4_2 = F.sub_mod(custom_vals["d_next_eval"],b_4_1)
    f_b4 = delta(b_4_2)
    b_4 = F.mul_mod_scalar(f_b4,kappa_cu.to("cuda"))

    mid1 = F.add_mod(b_1, b_2)
    mid2 = F.add_mod(mid1, b_3)
    mid3 = F.add_mod(mid2, b_4)
    temp = F.mul_mod_scalar(mid3, separation_challenge.to("cuda"))

    res = F.mul_mod(selector,temp)
    return res
    
def linearisation_term(selector_poly, separation_challenge, wit_vals, custom_vals):
    temp = _constraints(separation_challenge, wit_vals, custom_vals)
    if selector_poly.size(0) == 0:
        res = selector_poly.clone()
    else:
        res = F.mul_mod_scalar(selector_poly, temp)
    return res