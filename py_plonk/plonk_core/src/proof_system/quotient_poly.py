from ....domain import Radix2EvaluationDomain
import gmpy2
import torch
import copy
import torch.nn.functional as F
from ....bls12_381 import fr
from ....arithmetic import INTT,coset_NTT,coset_INTT,from_coeff_vec,coset_NTT_new
from ....plonk_core.src.proof_system.widget.mod import WitnessValues
from ....plonk_core.src.proof_system.widget.range import RangeGate,RangeValues
from ....plonk_core.src.proof_system.widget.logic import LogicGate,LogicValues
from ....plonk_core.src.proof_system.widget.fixed_base_scalar_mul import FBSMGate,FBSMValues
from ....plonk_core.src.proof_system.widget.curve_addition import CAGate,CAValues
from ....plonk_core.src.proof_system.mod import CustomEvaluations
from ....arithmetic import INTT,from_coeff_vec,resize,\
                        from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list,from_gmpy_list_1,domian_trans_tensor,calculate_execution_time,coset_NTT_new,coset_INTT_new,extend_tensor
import torch.nn as nn
from  .widget.arithmetic import compute_quotient_i
from ..proof_system.permutation import permutation_compute_quotient
from  ..proof_system.widget.lookup import compute_lookup_quotient_term
import numpy as np
import time
# Computes the first lagrange polynomial with the given `scale` over `domain`.
def compute_first_lagrange_poly_scaled(domain: Radix2EvaluationDomain,scale: fr.Fr):
    # x_evals = [fr.Fr.zero() for _ in range(domain.size)]
    inttclass = nn.Intt(domain.size, torch.BLS12_381_Fr_G1_Mont)
    x_evals=torch.zeros(domain.size,4,dtype=torch.BLS12_381_Fr_G1_Mont)
    x_evals[0] = scale

    x_coeffs = inttclass.forward(x_evals.to('cuda'))
    result_poly = from_coeff_vec(x_coeffs)
    return result_poly

def compute_gate_constraint_satisfiability(domain, 
    range_challenge, logic_challenge, fixed_base_challenge,
    var_base_challenge, prover_key, wl_eval_8n, wr_eval_8n, 
    wo_eval_8n, w4_eval_8n, pi_poly):

    #get Fr
    params = fr.Fr(gmpy2.mpz(0))
    domain_8n = Radix2EvaluationDomain.new(8 * domain.size,params)
    
    # domian_trans_tensor(domain_8n.group_gen_inv)
    # domian_trans_tensor(domain_8n.size_inv)
    # domian_trans_tensor(domain_8n.group_gen)
    pi_poly=pi_poly.to('cuda')
    pi_eval_8n = coset_NTT_new(domain_8n,pi_poly)

    gate_contributions = []

    
    def convert_to_tensors(data):
        for key, value in data.items():
            if isinstance(value, dict):
                convert_to_tensors(value)  # Recursively apply conversion
            elif isinstance(value, np.ndarray):
                if np.array_equal(value,np.array(4575657222473777152,dtype=np.uint64)):
                    value=[]
                data[key] = torch.tensor(value, dtype=torch.BLS12_381_Fr_G1_Mont)  # Convert numpy array to tensor
    
    prover_key_arithmetic=prover_key["arithmetic"].tolist()
    convert_to_tensors(prover_key_arithmetic)
    for key in ['q_l', 'q_r', 'q_c', 'q_m', 'q_o', 'q_4', 'q_hl', 'q_hr', 'q_h4', 'q_arith']:
        prover_key_arithmetic[key]['evals'] = prover_key_arithmetic[key]['evals'].to('cuda')

    prover_key_range_selector=prover_key["range_selector"].tolist()
    convert_to_tensors(prover_key_range_selector)
    prover_key_range_selector['evals']=prover_key_range_selector['evals'].to('cuda')

    prover_key_logic_selector=prover_key["logic_selector"].tolist()
    convert_to_tensors(prover_key_logic_selector)
    prover_key_logic_selector['evals']=prover_key_logic_selector['evals'].to('cuda')

    prover_key_fixed_group_add_selector=prover_key["fixed_group_add_selector"].tolist()
    convert_to_tensors(prover_key_fixed_group_add_selector)
    prover_key_fixed_group_add_selector['evals']=prover_key_fixed_group_add_selector['evals'].to('cuda')

    prover_key_variable_group_add_selector=prover_key["variable_group_add_selector"].tolist()
    convert_to_tensors(prover_key_variable_group_add_selector)
    prover_key_variable_group_add_selector['evals']=prover_key_variable_group_add_selector['evals'].to('cuda')


    timings = {
    'compute_quotient_i': 0,
    'RangeGate_quotient_term': 0,
    'LogicGate_quotient_term': 0,
    'FBSMGate_quotient_term': 0,
    'CAGate_quotient_term': 0
    }
    size=domain_8n.size
    four = fr.Fr.from_repr(4)
    four= torch.tensor(from_gmpy_list_1(four),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    four =extend_tensor(four,domain_8n.size)
    wit_vals = WitnessValues(
        a_val=wl_eval_8n[:8192],
        b_val=wr_eval_8n[:8192],
        c_val=wo_eval_8n[:8192],
        d_val=w4_eval_8n[:8192]
    )
    
    custom_vals = CustomEvaluations(
        vals=[
            ("a_next_eval", wl_eval_8n[8:]),
            ("b_next_eval", wr_eval_8n[8:]),
            ("d_next_eval", w4_eval_8n[8:]),
            ("q_l_eval", copy.deepcopy(prover_key_arithmetic['q_l']['evals'])),
            ("q_r_eval", copy.deepcopy(prover_key_arithmetic['q_r']['evals'])),
            ("q_c_eval", copy.deepcopy(prover_key_arithmetic["q_c"]['evals'])),
            # Possibly unnecessary but included nonetheless...
            ("q_hl_eval", copy.deepcopy(prover_key_arithmetic['q_hl']['evals'])),
            ("q_hr_eval", copy.deepcopy(prover_key_arithmetic['q_hr']['evals'])),
            ("q_h4_eval", copy.deepcopy(prover_key_arithmetic['q_h4']['evals']))
        ]
    )
    start = time.time()
    arithmetic = compute_quotient_i(prover_key_arithmetic, wit_vals)
    timings['compute_quotient_i'] += time.time() - start

    # 计时开始 RangeGate.quotient_term
    start = time.time()
    range_term = RangeGate.quotient_term(
        four,
        prover_key_range_selector['evals'],
        range_challenge,
        wit_vals,
        RangeValues.from_evaluations(custom_vals),
        size
    )
    timings['RangeGate_quotient_term'] += time.time() - start

    # 计时开始 LogicGate.quotient_term
    start = time.time()
    logic_term = LogicGate.quotient_term(
        four,
        prover_key_logic_selector['evals'],
        logic_challenge,
        wit_vals,
        LogicValues.from_evaluations(custom_vals),
        size
    )
    timings['LogicGate_quotient_term'] += time.time() - start

    # 计时开始 FBSMGate.quotient_term
    start = time.time()
    fixed_base_scalar_mul_term = FBSMGate.quotient_term(
        prover_key_fixed_group_add_selector['evals'],
        fixed_base_challenge,
        wit_vals,
        FBSMValues.from_evaluations(custom_vals),
        size

    )
    timings['FBSMGate_quotient_term'] += time.time() - start

    # 计时开始 CAGate.quotient_term
    start = time.time()
    curve_addition_term = CAGate.quotient_term(
        prover_key_variable_group_add_selector['evals'],
        var_base_challenge,
        wit_vals,
        CAValues.from_evaluations(custom_vals),
        size
    )
    timings['CAGate_quotient_term'] += time.time() - start
    mid1 = F.add_mod(arithmetic ,pi_eval_8n[:8192])
    mid2 = F.add_mod(mid1, range_term)
    mid3 = F.add_mod(mid2, logic_term)
    mid4 = F.add_mod(mid3, fixed_base_scalar_mul_term)
    gate_contributions = F.add_mod(mid4, curve_addition_term)
    # gate_contributions.append(gate_i)


    for function, total_time in timings.items():
        print(f"Total time for {function}: {total_time:.6f} seconds")
    return gate_contributions

@calculate_execution_time
def compute_permutation_checks(
    domain:Radix2EvaluationDomain,
    prover_key,
    wl_eval_8n: list[fr.Fr], wr_eval_8n: list[fr.Fr],
    wo_eval_8n: list[fr.Fr], w4_eval_8n: list[fr.Fr],
    z_eval_8n: list[fr.Fr], alpha: fr.Fr, beta: fr.Fr, gamma: fr.Fr):

    #get Fr
    params = fr.Fr(gmpy2.mpz(0))
    #get NTT domain
    domain_8n:Radix2EvaluationDomain = Radix2EvaluationDomain.new(8 * domain.size,params)
    domian_trans_tensor(domain_8n.group_gen_inv)
    domian_trans_tensor(domain_8n.size_inv)
    domian_trans_tensor(domain_8n.group_gen)
    
    # Calculate l1_poly_alpha and l1_alpha_sq_evals
    alpha=torch.tensor(from_gmpy_list_1(alpha),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    gamma=torch.tensor(from_gmpy_list_1(gamma),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    beta=torch.tensor(from_gmpy_list_1(beta),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')

    alpha2= F.mul_mod(alpha,alpha)
    l1_poly_alpha = compute_first_lagrange_poly_scaled(domain, alpha2)
    l1_alpha_sq_evals = coset_NTT_new(domain_8n,l1_poly_alpha.to('cuda'))

    # Initialize result list
    result = []

    pk_permutation = prover_key['permutation'].tolist()
    # keys = ["left_sigma", "right_sigma", "out_sigma", "fourth_sigma"]
    # for key in keys:
    #     pk_permutation[key]= pk_permutation[key].tolist()

    pk_linear_evaluations=prover_key["linear_evaluations"].tolist()

    pk_linear_evaluations_evals = torch.tensor(pk_linear_evaluations['evals'], dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    pk_left_sigma_evals = torch.tensor(pk_permutation["left_sigma"]['evals'], dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    pk_right_sigma_evals = torch.tensor(pk_permutation["right_sigma"]['evals'], dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    pk_out_sigma_evals = torch.tensor(pk_permutation["out_sigma"]['evals'], dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    pk_fourth_sigma_evals = torch.tensor(pk_permutation["fourth_sigma"]['evals'], dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    
    # Calculate permutation contribution for each index
    
    quotient = permutation_compute_quotient(
        pk_linear_evaluations_evals,
        pk_left_sigma_evals,
        pk_right_sigma_evals,
        pk_out_sigma_evals,
        pk_fourth_sigma_evals,
        wl_eval_8n[:domain_8n.size],
        wr_eval_8n[:domain_8n.size],
        wo_eval_8n[:domain_8n.size],
        w4_eval_8n[:domain_8n.size],
        z_eval_8n[:domain_8n.size],
        z_eval_8n[8:],
        alpha,
        l1_alpha_sq_evals[:domain_8n.size],
        beta,
        gamma
    )
    

    return quotient

@calculate_execution_time
def compute_quotient_poly(domain: Radix2EvaluationDomain, 
            prover_key, 
            z_poly, z2_poly, 
            w_l_poly, w_r_poly, w_o_poly, w_4_poly, 
            public_inputs_poly, 
            f_poly, table_poly, h1_poly, h2_poly, 
            alpha: fr.Fr, beta, gamma, delta, epsilon, zeta, 
            range_challenge, logic_challenge, 
            fixed_base_challenge, var_base_challenge, 
            lookup_challenge):
    start_time = time.time()
    #get Fr
    params = fr.Fr(gmpy2.mpz(0))
    #get NTT domain
    domain_8n = Radix2EvaluationDomain.new(8 * domain.size,params)
    # domian_trans_tensor(domain_8n.group_gen_inv)
    # domian_trans_tensor(domain_8n.size_inv)
    # domian_trans_tensor(domain_8n.group_gen)
    l1_poly = compute_first_lagrange_poly_scaled(domain, torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)) ########输出为Tensor
    
    a=l1_poly.tolist()
    with open('list_file.txt', 'w') as f:
        for row in a:
            row_str = ' '.join(map(str, row)) + '\n'
            f.write(row_str)

    l1_eval_8n = coset_NTT_new(domain_8n,l1_poly.to('cuda'))
    z_eval_8n = coset_NTT_new(domain_8n,z_poly.to('cuda'))
    
    z_eval_8n = torch.cat((z_eval_8n, z_eval_8n[:8]), dim=0)
    
    wl_eval_8n = coset_NTT_new(domain_8n,w_l_poly.to('cuda'))
    wl_eval_8n = torch.cat((wl_eval_8n, wl_eval_8n[:8]), dim=0)

    wr_eval_8n = coset_NTT_new(domain_8n,w_r_poly.to('cuda'))
    wr_eval_8n = torch.cat((wr_eval_8n, wr_eval_8n[:8]), dim=0)

    wo_eval_8n = coset_NTT_new(domain_8n,w_o_poly.to('cuda'))

    w4_eval_8n = coset_NTT_new(domain_8n,w_4_poly.to('cuda'))
    w4_eval_8n = torch.cat((w4_eval_8n, w4_eval_8n[:8]), dim=0)

    z2_eval_8n = coset_NTT_new(domain_8n,z2_poly.to('cuda'))
    z2_eval_8n = torch.cat((z2_eval_8n, z2_eval_8n[:8]), dim=0)

    f_eval_8n =coset_NTT_new(domain_8n,f_poly.to('cuda'))

    table_eval_8n = coset_NTT_new(domain_8n,table_poly.to('cuda'))
    table_eval_8n = torch.cat((table_eval_8n, table_eval_8n[:8]), dim=0)

    h1_eval_8n = coset_NTT_new(domain_8n,h1_poly.to('cuda'))
    h1_eval_8n = torch.cat((h1_eval_8n, h1_eval_8n[:8]), dim=0)

    h2_eval_8n = coset_NTT_new(domain_8n,h2_poly.to('cuda'))

    end_time = time.time()
    print(f"1_Operation took {end_time - start_time} seconds")
    range_challenge=torch.tensor(from_gmpy_list_1(range_challenge),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    logic_challenge = torch.tensor(from_gmpy_list_1(logic_challenge), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    fixed_base_challenge = torch.tensor(from_gmpy_list_1(fixed_base_challenge), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    var_base_challenge = torch.tensor(from_gmpy_list_1(var_base_challenge), dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    lookup_challenge =torch.tensor(from_gmpy_list_1(lookup_challenge),dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')

    gate_constraints = compute_gate_constraint_satisfiability(
        domain,
        range_challenge,logic_challenge,
        fixed_base_challenge,var_base_challenge,
        prover_key,
        wl_eval_8n,wr_eval_8n,wo_eval_8n,w4_eval_8n,
        public_inputs_poly,
    )
    gate_constraints_time = time.time()
    print(f"gate_constraints took {gate_constraints_time - start_time} seconds")
    permutation = compute_permutation_checks(
        domain,
        prover_key,
        wl_eval_8n,wr_eval_8n,wo_eval_8n,w4_eval_8n,z_eval_8n,
        alpha,beta,gamma,
    )
    permutation_time = time.time()
    print(f"permutation took {permutation_time - start_time} seconds")

    pk_lookup=prover_key['lookup'].tolist()
    pk_lookup_qlookup_evals=torch.tensor(pk_lookup['q_lookup']['evals'],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    

    lookup = compute_lookup_quotient_term(
        domain,
        wl_eval_8n,
        wr_eval_8n,
        wo_eval_8n,
        w4_eval_8n,
        f_eval_8n,
        table_eval_8n,
        h1_eval_8n,
        h2_eval_8n,
        z2_eval_8n,
        l1_eval_8n,
        delta,
        epsilon,
        zeta,
        lookup_challenge,
        pk_lookup_qlookup_evals
    )
    lookup_time = time.time()
    print(f"permutation took {lookup_time - start_time} seconds")
    quotient = torch.empty(domain_8n.size,4,dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    prover_key_v_h_coset_8n=prover_key["v_h_coset_8n"].tolist()
    prover_key_v_h_coset_8n_evals=torch.tensor(prover_key_v_h_coset_8n['evals'],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    #TODO 
    # for i in range(domain_8n.size):
    #     numerator = F.add_mod(gate_constraints[i],permutation[i])
    #     numerator = F.add_mod(numerator,lookup[i])
    #     denominator=F.div_mod(torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda'),prover_key_v_h_coset_8n_evals[i])
    #     res =F.mul_mod(numerator,denominator)
    #     quotient[i]=res
    one=torch.tensor([8589934590, 6378425256633387010, 11064306276430008309, 1739710354780652911],dtype=torch.BLS12_381_Fr_G1_Mont)
    extend_one=extend_tensor(one,domain_8n.size)
    numerator = F.add_mod(gate_constraints,permutation)
    numerator = F.add_mod(numerator,lookup)
    denominator = F.div_mod(extend_one,prover_key_v_h_coset_8n_evals)
    res =F.mul_mod(numerator,denominator)
    quotient_poly = coset_INTT_new(res,domain_8n)
    hx = from_coeff_vec(quotient_poly)

    return hx