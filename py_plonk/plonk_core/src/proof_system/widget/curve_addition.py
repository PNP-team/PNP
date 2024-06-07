from dataclasses import dataclass
from .....bls12_381 import fr
from .....plonk_core.src.proof_system.widget.mod import WitnessValues
from .....bls12_381 import edwards as P
import torch
import torch.nn.functional as F
@dataclass
class CAValues:
    # Left wire value in the next position
    a_next_val: fr.Fr
    # Right wire value in the next position
    b_next_val: fr.Fr
    # Fourth wire value in the next position
    d_next_val: fr.Fr

    @staticmethod
    def from_evaluations(custom_evals):
        a_next_val = custom_evals["a_next_eval"]
        b_next_val = custom_evals["b_next_eval"]
        d_next_val = custom_evals["d_next_eval"]

        return CAValues(a_next_val,b_next_val,d_next_val)
    
class CAGate:
    @staticmethod
    def constraints(separation_challenge, wit_vals: WitnessValues, custom_vals: CAValues):
        x_1 = wit_vals.a_val
        x_3 = custom_vals.a_next_val
        y_1 = wit_vals.b_val
        y_3 = custom_vals.b_next_val
        x_2 = wit_vals.c_val
        y_2 = wit_vals.d_val
        x1_y2 = custom_vals.d_next_val

        kappa = F.mul_mod(separation_challenge,separation_challenge)

        # Check that `x1 * y2` is correct
        x1y2 = F.mul_mod(x_1, y_2)
        xy_consistency = F.sub_mod(x1y2, x1_y2)

        y1_x2 = F.mul_mod(y_1, x_2)
        y1_y2 = F.mul_mod(y_1, y_2)
        x1_x2 = F.mul_mod(x_1, x_2)

        # Check that `x_3` is correct
        x3_lhs = F.add_mod(x1_y2, y1_x2)
        x_3_D = F.mul_mod(x_3, P.COEFF_D())
        x_3_D_x1_y2 = F.mul_mod(x_3_D, x1_y2)
        x_3_D_x1_y2_y1_x2 = F.mul_mod(x_3_D_x1_y2, y1_x2)
        x3_rhs = F.add_mod(x_3, x_3_D_x1_y2_y1_x2)
        x3_l_sub_r = F.sub_mod(x3_lhs, x3_rhs)
        x3_consistency = F.mul_mod(x3_l_sub_r, kappa)

        # Check that `y_3` is correct
        x1_x2_A = F.mul_mod(P.COEFF_A(), x1_x2)
        y3_lhs = F.sub_mod(y1_y2, x1_x2_A)
        y_3_D = F.mul_mod(y_3, P.COEFF_D())
        y_3_D_x1_y2 = F.mul_mod(y_3_D, x1_y2)
        y_3_D_x1_y2_y1_x2 = F.mul_mod(y_3_D_x1_y2, y1_x2)
        y3_rhs = F.sub_mod(y_3, y_3_D_x1_y2_y1_x2)
        y3_l_sub_r = F.sub_mod(y3_lhs, y3_rhs)
        kappa2 = F.mul_mod(kappa,kappa)
        y3_consistency = F.mul_mod(y3_l_sub_r, kappa2)

        mid1 = F.add_mod(xy_consistency, x3_consistency)
        mid2 = F.add_mod(mid1, y3_consistency)
        result = F.mul_mod(mid2, separation_challenge)
        return result



    @staticmethod
    def quotient_term(selector: torch.Tensor, separation_challenge: torch.Tensor, 
                      wit_vals: WitnessValues, custom_vals:CAValues):
        x_1 = wit_vals.a_val
        x_3 = custom_vals.a_next_val
        y_1 = wit_vals.b_val
        y_3 = custom_vals.b_next_val
        x_2 = wit_vals.c_val
        y_2 = wit_vals.d_val
        x1_y2 = custom_vals.d_next_val

        #single scalar OP on CPU
        kappa = F.mul_mod(separation_challenge, separation_challenge)
        kappasq = F.mul_mod(kappa, kappa)

        # Check that `x1 * y2` is correct
        x1y2 = F.mul_mod(x_1, y_2)
        xy_consistency = F.sub_mod(x1y2, x1_y2)

        y1_x2 = F.mul_mod(y_1, x_2)
        y1_y2 = F.mul_mod(y_1, y_2)
        x1_x2 = F.mul_mod(x_1, x_2)

        # Check that `x_3` is correct
        x3_lhs = F.add_mod(x1_y2, y1_x2)
        x_3_D = F.mul_mod_scalar(x_3, P.COEFF_D().to("cuda"))
        x_3_D_x1_y2 = F.mul_mod(x_3_D, x1_y2)
        x_3_D_x1_y2_y1_x2 = F.mul_mod(x_3_D_x1_y2, y1_x2)
        x3_rhs = F.add_mod(x_3, x_3_D_x1_y2_y1_x2)
        x3_l_sub_r = F.sub_mod(x3_lhs, x3_rhs)
        x3_consistency = F.mul_mod_scalar(x3_l_sub_r, kappa.to("cuda"))

        # Check that `y_3` is correct
        x1_x2_A = F.mul_mod_scalar(x1_x2, P.COEFF_A().to("cuda"))
        y3_lhs = F.sub_mod(y1_y2, x1_x2_A)
        y_3_D = F.mul_mod_scalar(y_3, P.COEFF_D().to("cuda"))
        y_3_D_x1_y2 = F.mul_mod(y_3_D, x1_y2)
        y_3_D_x1_y2_y1_x2 = F.mul_mod(y_3_D_x1_y2, y1_x2)
        y3_rhs = F.sub_mod(y_3, y_3_D_x1_y2_y1_x2)
        y3_l_sub_r = F.sub_mod(y3_lhs, y3_rhs)
        y3_consistency = F.mul_mod_scalar(y3_l_sub_r, kappasq.to("cuda"))

        mid1 = F.add_mod(xy_consistency, x3_consistency)
        mid2 = F.add_mod(mid1, y3_consistency)
        temp = F.mul_mod_scalar(mid2, separation_challenge.to("cuda"))

        res = F.mul_mod(selector, temp)
        return res
    
    @staticmethod
    def linearisation_term(selector_poly, separation_challenge, wit_vals, custom_vals):
        temp = CAGate.constraints(separation_challenge, wit_vals, custom_vals)
        res = F.mul_mod_scalar(selector_poly,temp)
        return res
