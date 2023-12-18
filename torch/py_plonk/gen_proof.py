import gmpy2
import copy
import itertools
import torch
import torch.nn as nn
from .domain import newdomain,element
from .transcript import transcript
from .composer import StandardComposer
from .transcript import transcript
from .plonk_core.lookup import multiset
from .plonk_core.src.permutation import mod 
from .plonk_core.src.proof_system.prover_key import Prover_Key
from .plonk_core.src.proof_system.pi import into_dense_poly
from .plonk_core.src.proof_system import quotient_poly
from .plonk_core.src.proof_system import linearisation_poly
from .arithmetic import INTT,from_coeff_vec,resize,is_zero_poly,\
                        from_gmpy_list,from_list_gmpy,from_list_tensor,from_tensor_list
from .load import read_scalar_data
from .KZG import kzg10
from .KZG.kzg10 import commit
from .bls12_381 import fr


def gen_proof(pp, pk: Prover_Key, cs: StandardComposer, transcript: transcript.Transcript):
    #init Fr params (FFTfield), the value is arbitrary
    Fr=fr.Fr(value = 0)
    #get FFT domaim
    domain = newdomain(cs.circuit_bound())
    n = domain["size"]
    transcript.append_pi(b"pi")
    powers = [pp.powers_of_g, pp.powers_of_gamma_g]
    dtype = torch.BLS12_381_Fr_G1_Mont

    #1. Compute witness Polynomials
    w_l_scalar=read_scalar_data("torch/py_plonk/w_l_scalar.txt")
    w_r_scalar=read_scalar_data("torch/py_plonk/w_r_scalar.txt")
    w_o_scalar=read_scalar_data("torch/py_plonk/w_o_scalar.txt")
    w_4_scalar=read_scalar_data("torch/py_plonk/w_4_scalar.txt")

    INTTclass = nn.NTT_zkp(True, 0, 32, dtype)

    w_l_evals = copy.deepcopy(w_l_scalar)
    w_r_evals = copy.deepcopy(w_r_scalar)
    w_o_evals = copy.deepcopy(w_o_scalar)
    w_4_evals = copy.deepcopy(w_4_scalar)

    from_gmpy_list(w_l_evals)
    from_gmpy_list(w_r_evals)
    from_gmpy_list(w_o_evals)
    from_gmpy_list(w_4_evals)

    w_l_input = from_list_tensor(w_l_evals, dtype)
    w_r_input = from_list_tensor(w_r_evals, dtype)
    w_o_input = from_list_tensor(w_o_evals, dtype)
    w_4_input = from_list_tensor(w_4_evals, dtype)

    w_l_input_gpu = w_l_input.to("cuda")
    w_r_input_gpu = w_r_input.to("cuda")
    w_o_input_gpu = w_o_input.to("cuda")
    w_4_input_gpu = w_4_input.to("cuda")
    #假设数据已经存在gpu上，如果想要只输入data，需要把ntt/intt单独写成class
    w_l_output = INTTclass.forward(w_l_input_gpu, True, False)
    w_r_output = INTTclass.forward(w_r_input_gpu, True, False)
    w_o_output = INTTclass.forward(w_o_input_gpu, True, False)
    w_4_output = INTTclass.forward(w_4_input_gpu, True, False)

    w_l_coeffs = w_l_output.to("cpu")
    w_r_coeffs = w_r_output.to("cpu")
    w_o_coeffs = w_o_output.to("cpu")
    w_4_coeffs = w_4_output.to("cpu")

    w_l_poly = from_tensor_list(w_l_coeffs)
    w_r_poly = from_tensor_list(w_r_coeffs)
    w_o_poly = from_tensor_list(w_o_coeffs)
    w_4_poly = from_tensor_list(w_4_coeffs)

    #w_commits_tensor = torch.tensor([w_l_poly,w_r_poly,w_o_poly,w_4_poly], dtype = dtype)

    from_list_gmpy(w_l_poly)
    from_list_gmpy(w_r_poly)
    from_list_gmpy(w_o_poly)
    from_list_gmpy(w_4_poly)

    # w_l_poly = from_coeff_vec(INTT(w_l_scalar))
    # w_r_poly = from_coeff_vec(INTT(w_r_scalar))
    # w_o_poly = from_coeff_vec(INTT(w_o_scalar))
    # w_4_poly = from_coeff_vec(INTT(w_4_scalar))

    w_polys = [w_l_poly,w_r_poly,w_o_poly,w_4_poly]
    #假设做完INTT后做MSM，数据直接传入，MSM的参数为2个tensor，scalar 和 point
    w_l_commits, w_l_rands = commit(powers,w_l_poly)
    w_r_commits, w_r_rands = commit(powers,w_r_poly)
    w_o_commits, w_o_rands = commit(powers,w_o_poly)
    w_4_commits, w_4_rands = commit(powers,w_4_poly)

    w_commits = [w_l_commits,
                 w_r_commits,
                 w_o_commits,
                 w_4_commits]
    
    w_rands = [w_l_rands,
               w_r_rands,
               w_o_rands,
               w_4_rands]
    
    transcript.append(b"w_l",w_l_commits)
    transcript.append(b"w_r",w_r_commits)
    transcript.append(b"w_o",w_o_commits)
    transcript.append(b"w_4",w_4_commits)
    #2. Derive lookup polynomials

    # Generate table compression factor
    zeta = transcript.challenge_scalar(b"zeta",Fr)
    transcript.append(b"zeta",zeta)

    # Compress lookup table into vector of single elements
    t_multiset = [pk.lookup.table_1,pk.lookup.table_2,
                  pk.lookup.table_3,pk.lookup.table_4]
    
    compressed_t_multiset = multiset.compress(t_multiset, zeta)

    #Compute table poly
    flag_t = is_zero_poly(compressed_t_multiset)
    if flag_t:
        compressed_t_poly = copy.deepcopy(compressed_t_multiset)
    else:
        compressed_t_poly = INTT(compressed_t_multiset)
    table_poly = from_coeff_vec(compressed_t_poly)

    # Compute query table f
    # When q_lookup[i] is zero the wire value is replaced with a dummy
    # value currently set as the first row of the public table
    # If q_lookup[i] is one the wire values are preserved
    # This ensures the ith element of the compressed query table
    # is an element of the compressed lookup table even when
    # q_lookup[i] is 0 so the lookup check will pass

    q_lookup_pad = [fr.Fr(value=gmpy2.mpz(0)) for _ in  range((n - len(cs.q_lookup)))]
    padded_q_lookup = cs.q_lookup + q_lookup_pad

    f_scalars = [[],[],[],[]]
    for q_lookup, w_l, w_r, w_o, w_4 in zip(padded_q_lookup, w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar):
        if q_lookup.value == 0:
            f_scalars[0].append(compressed_t_multiset[0])
            for key in range(1,4):
                    f_scalars[key].append(fr.Fr(gmpy2.mpz(0)))  
        else:
            f_scalars[0].append(w_l)
            f_scalars[1].append(w_r)
            f_scalars[2].append(w_o)
            f_scalars[3].append(w_4)

    # Compress all wires into a single vector
    compressed_f_multiset = multiset.compress(f_scalars, zeta)

    # Compute query poly
    flag_f = is_zero_poly(compressed_f_multiset)
    if flag_f:
        compressed_f_poly = copy.deepcopy(compressed_f_multiset)
    else:
        compressed_f_poly = INTT(compressed_f_multiset)
    f_poly = from_coeff_vec(compressed_f_poly)

    # Commit to query polynomial
    f_poly_commit, _ = commit(powers,f_poly)
    transcript.append(b"f",f_poly_commit)

    # Compute s, as the sorted and concatenated version of f and t
    h_1, h_2 = multiset.combine_split(compressed_t_multiset, compressed_f_multiset)

    # Compute h polys
    flag_h_1 = is_zero_poly(h_1)
    flag_h_2 = is_zero_poly(h_2)
    if flag_h_1:
        h_1_temp = copy.deepcopy(h_1)
    else:
        h_1_temp = INTT(h_1)
    if flag_h_2:
        h_2_temp = copy.deepcopy(h_2)
    else:
        h_2_temp = INTT(h_2)

    h_1_poly = from_coeff_vec(h_1_temp)
    h_2_poly = from_coeff_vec(h_2_temp)

    # Commit to h polys
    h_1_poly_commit,_ = commit(powers,h_1_poly)
    h_2_poly_commit,_ = commit(powers,h_2_poly)

    # Add h polynomials to transcript
    transcript.append(b"h1", h_1_poly_commit)
    transcript.append(b"h2", h_2_poly_commit)

    # 3. Compute permutation polynomial

    # Compute permutation challenge `beta`.
    beta = transcript.challenge_scalar(b"beta",Fr)
    transcript.append(b"beta", beta)
    # Compute permutation challenge `gamma`.
    gamma = transcript.challenge_scalar(b"gamma",Fr)
    transcript.append(b"gamma", gamma)
    # Compute permutation challenge `delta`.
    delta = transcript.challenge_scalar(b"delta",Fr)
    transcript.append(b"delta", delta)
    # Compute permutation challenge `epsilon`.
    epsilon = transcript.challenge_scalar(b"epsilon",Fr)
    transcript.append(b"epsilon", epsilon)

    # Challenges must be different
    assert beta.value != gamma.value, "challenges must be different"
    assert beta.value != delta.value, "challenges must be different"
    assert beta.value != epsilon.value, "challenges must be different"
    assert gamma.value != delta.value, "challenges must be different"
    assert gamma.value != epsilon.value, "challenges must be different"
    assert delta.value != epsilon.value, "challenges must be different"
    
    z_poly = mod.compute_permutation_poly(domain,
        (w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar),
        beta,
        gamma,
        (
            pk.permutation.left_sigma[0],
            pk.permutation.right_sigma[0],
            pk.permutation.out_sigma[0],
            pk.permutation.fourth_sigma[0]
        ))
    # Commit to permutation polynomial.
    z_poly_commit,_ = commit(powers,z_poly)

    # Add permutation polynomial commitment to transcript.
    transcript.append(b"z", z_poly_commit)
    
    # Compute mega permutation polynomial.
    # Compute lookup permutation poly
    z_2_poly = mod.compute_lookup_permutation_poly(
        domain,
        compressed_f_multiset,
        compressed_t_multiset,
        h_1,
        h_2,
        delta,
        epsilon
    )

    # Commit to lookup permutation polynomial.
    z_2_poly_commit,_ = commit(powers,z_2_poly)

    # 3. Compute public inputs polynomial
    pi_poly = into_dense_poly(cs.public_inputs,cs.intended_pi_pos,n)

    # 4. Compute quotient polynomial

    # Compute quotient challenge `alpha`, and gate-specific separation challenges.
    alpha = transcript.challenge_scalar(b"alpha",Fr)
    transcript.append(b"alpha", alpha)

    range_sep_challenge = transcript.challenge_scalar(b"range separation challenge",Fr)
    transcript.append(b"range seperation challenge", range_sep_challenge)

    logic_sep_challenge = transcript.challenge_scalar(b"logic separation challenge",Fr)
    transcript.append(b"logic seperation challenge", logic_sep_challenge)

    fixed_base_sep_challenge = transcript.challenge_scalar(b"fixed base separation challenge",Fr)
    transcript.append(b"fixed base separation challenge", fixed_base_sep_challenge)

    var_base_sep_challenge = transcript.challenge_scalar(b"variable base separation challenge",Fr)
    transcript.append(b"variable base separation challenge", var_base_sep_challenge)

    lookup_sep_challenge = transcript.challenge_scalar(b"lookup separation challenge",Fr)
    transcript.append(b"lookup separation challenge", lookup_sep_challenge)

    t_poly = quotient_poly.compute(
        domain,pk,
        z_poly,z_2_poly,
        w_l_poly,w_r_poly,w_o_poly,w_4_poly,
        pi_poly,
        f_poly,table_poly,h_1_poly,h_2_poly,
        alpha,beta,gamma,delta,epsilon,zeta,
        range_sep_challenge,logic_sep_challenge,
        fixed_base_sep_challenge,
        var_base_sep_challenge,
        lookup_sep_challenge)

    t_i_poly = split_tx_poly(n, t_poly)

    t_1_commits, _ = commit(powers,t_i_poly[0])
    t_2_commits, _ = commit(powers,t_i_poly[1])
    t_3_commits, _ = commit(powers,t_i_poly[2])
    t_4_commits, _ = commit(powers,t_i_poly[3])
    t_5_commits, _ = commit(powers,t_i_poly[4])
    t_6_commits, _ = commit(powers,t_i_poly[5])
    t_7_commits, _ = commit(powers,t_i_poly[6])
    t_8_commits, _ = commit(powers,t_i_poly[7])

    # Add quotient polynomial commitments to transcript
    transcript.append(b"t_1", t_1_commits)
    transcript.append(b"t_2", t_2_commits)
    transcript.append(b"t_3", t_3_commits)
    transcript.append(b"t_4", t_4_commits)
    transcript.append(b"t_5", t_5_commits)
    transcript.append(b"t_6", t_6_commits)
    transcript.append(b"t_7", t_7_commits)
    transcript.append(b"t_8", t_8_commits)

    # 4. Compute linearisation polynomial

    # Compute evaluation challenge `z`.
    z_challenge = transcript.challenge_scalar(b"z", Fr)
    transcript.append(b"z", z_challenge)

    lin_poly, evaluations = linearisation_poly.compute(
            domain,
            pk,
            alpha,beta,gamma,delta,epsilon,zeta,
            range_sep_challenge,
            logic_sep_challenge,
            fixed_base_sep_challenge,
            var_base_sep_challenge,
            lookup_sep_challenge,
            z_challenge,
            w_l_poly,w_r_poly,w_o_poly,w_4_poly,
            t_i_poly[0],
            t_i_poly[1],
            t_i_poly[2],
            t_i_poly[3],
            t_i_poly[4],
            t_i_poly[5],
            t_i_poly[6],
            t_i_poly[7],
            z_poly,
            z_2_poly,
            f_poly,
            h_1_poly,
            h_2_poly,
            table_poly)
    
    # Add evaluations to transcript.
    # First wire evals
    transcript.append(b"a_eval", evaluations.wire_evals.a_eval)
    transcript.append(b"b_eval", evaluations.wire_evals.b_eval)
    transcript.append(b"c_eval", evaluations.wire_evals.c_eval)
    transcript.append(b"d_eval", evaluations.wire_evals.d_eval)

    # Second permutation evals
    transcript.append(b"left_sig_eval", evaluations.perm_evals.left_sigma_eval)
    transcript.append(b"right_sig_eval",evaluations.perm_evals.right_sigma_eval)
    transcript.append(b"out_sig_eval", evaluations.perm_evals.out_sigma_eval)
    transcript.append(b"perm_eval", evaluations.perm_evals.permutation_eval)

    # Third lookup evals
    transcript.append(b"f_eval", evaluations.lookup_evals.f_eval)
    transcript.append(b"q_lookup_eval", evaluations.lookup_evals.q_lookup_eval)
    transcript.append(b"lookup_perm_eval",evaluations.lookup_evals.z2_next_eval)
    transcript.append(b"h_1_eval", evaluations.lookup_evals.h1_eval)
    transcript.append(b"h_1_next_eval", evaluations.lookup_evals.h1_next_eval)
    transcript.append(b"h_2_eval", evaluations.lookup_evals.h2_eval)

    # Fourth, all evals needed for custom gates
    for label, eval in evaluations.custom_evals.vals:
        static_label = label.encode('utf-8')
        transcript.append(static_label, eval)

    # 5. Compute Openings using KZG10
    #
    # We merge the quotient polynomial using the `z_challenge` so the SRS
    # is linear in the circuit size `n`

    # Compute aggregate witness to polynomials evaluated at the evaluation
    # challenge `z`
    aw_challenge = transcript.challenge_scalar(b"aggregate_witness", Fr)

    # XXX: The quotient polynomials is used here and then in the
    # opening poly. It is being left in for now but it may not
    # be necessary. Warrants further investigation.
    # Ditto with the out_sigma poly.
    
    aw_polys = [lin_poly,
                pk.permutation.left_sigma[0],
                pk.permutation.right_sigma[0],
                pk.permutation.out_sigma[0],
                f_poly,
                h_2_poly,
                table_poly]
    
    lin_commits, lin_rands = commit(powers,lin_poly)
    left_sigma_commits, left_sigma_rands = commit(powers,pk.permutation.left_sigma[0])
    right_sigma_commits, right_sigma_rands = commit(powers,pk.permutation.right_sigma[0])
    out_sigma_commits, out_sigma_rands = commit(powers,pk.permutation.out_sigma[0])
    f_poly_sigma_commits, f_poly_rands = commit(powers,f_poly)
    h_2_poly_sigma_commits, h_2_poly_rands = commit(powers,h_2_poly)
    table_poly_commits, table_poly_rands = commit(powers,table_poly)

    aw_commits = [lin_commits,
                  left_sigma_commits,
                  right_sigma_commits,
                  out_sigma_commits,
                  f_poly_sigma_commits,
                  h_2_poly_sigma_commits,
                  table_poly_commits]
    
    aw_rands = [lin_rands,
                left_sigma_rands,
                right_sigma_rands,
                out_sigma_rands,
                f_poly_rands,
                h_2_poly_rands,
                table_poly_rands]
    
    aw_opening = kzg10.open(
        powers,
        itertools.chain(aw_polys, w_polys),
        itertools.chain(aw_commits, w_commits),
        z_challenge,
        aw_challenge,
        itertools.chain(aw_rands, w_rands),
        None
    )

    saw_challenge = transcript.challenge_scalar(b"aggregate_witness", Fr)
    
    saw_polys = [z_poly,
                w_l_poly,
                w_r_poly,
                w_4_poly,
                h_1_poly,
                z_2_poly,
                table_poly]
    
    z_poly_commits, z_poly_rands = commit(powers,z_poly)
    w_l_poly_commits, w_l_poly_rands = commit(powers,w_l_poly)
    w_r_poly_commits, w_r_poly_rands = commit(powers,w_r_poly)
    w_4_poly_commits, w_4_poly_rands = commit(powers,w_4_poly)
    h_1_poly_commits, h_1_poly_rands = commit(powers,h_1_poly)
    z_2_poly_commits, z_2_poly_rands = commit(powers,z_2_poly)
    table_poly_commits, table_poly_rands = commit(powers,table_poly)

    saw_commits=[z_poly_commits,
                 w_l_poly_commits,
                 w_r_poly_commits,
                 w_4_poly_commits,
                 h_1_poly_commits,
                 z_2_poly_commits,
                 table_poly_commits]

    saw_rands = [z_poly_rands,
                 w_l_poly_rands,
                 w_r_poly_rands,
                 w_4_poly_rands,
                 h_1_poly_rands,
                 z_2_poly_rands,
                 table_poly_rands]
    

    saw_opening = kzg10.open(
        powers,
        saw_polys,
        saw_commits,
        z_challenge.mul(element(domain, 1)),
        saw_challenge,
        saw_rands,
        None
    )

    Proof = kzg10.Proof(
            a_comm = w_l_commits,
            b_comm = w_r_commits,
            c_comm = w_o_commits,
            d_comm = w_4_commits,
            z_comm = z_poly_commits,
            f_comm = f_poly_commit,
            h_1_comm = h_1_poly_commit,
            h_2_comm = h_2_poly_commit,
            z_2_comm = z_2_poly_commit,
            t_1_comm = t_1_commits,
            t_2_comm = t_2_commits,
            t_3_comm = t_3_commits,
            t_4_comm = t_4_commits,
            t_5_comm = t_5_commits,
            t_6_comm = t_6_commits,
            t_7_comm = t_7_commits,
            t_8_comm = t_8_commits,
            aw_opening = aw_opening,
            saw_opening = saw_opening,
            evaluations = evaluations)
    return Proof


def split_tx_poly(n, t_x):
    buf:list = t_x[:]
    buf = resize(buf, n << 3, fr.Fr.zero())
    return [
        from_coeff_vec(buf[0:n]),
        from_coeff_vec(buf[n:2 * n]),
        from_coeff_vec(buf[2 * n:3 * n]),
        from_coeff_vec(buf[3 * n:4 * n]),
        from_coeff_vec(buf[4 * n:5 * n]),
        from_coeff_vec(buf[5 * n:6 * n]),
        from_coeff_vec(buf[6 * n:7 * n]),
        from_coeff_vec(buf[7 * n:])
    ]


