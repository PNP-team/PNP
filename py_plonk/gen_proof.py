import itertools
from .domain import Radix2EvaluationDomain
from .transcript import transcript
from .composer import StandardComposer
from .transcript import transcript
from .bls12_381 import fr
from .plonk_core.lookup import multiset
from .plonk_core.src.permutation import mod 
from .plonk_core.src.proof_system.prover_key import Prover_Key
from .plonk_core.src.proof_system.pi import into_dense_poly
from .plonk_core.src.proof_system import quotient_poly
from .plonk_core.src.proof_system import linearisation_poly
import numpy as np
from .arithmetic import from_coeff_vec,resize_cpu,\
                        from_list_gmpy_1,INTT
from .KZG import kzg10
import torch
import time
date_set2=["../../data/MERKLE-HEIGHT-9/pp-9.npz","../../data/MERKLE-HEIGHT-9/pk-9.npz","../../data/MERKLE-HEIGHT-9/cs-9.npz","../../data/MERKLE-HEIGHT-9/w_l_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_r_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_o_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_4_scalar-9.npy"]

def split_tx_poly(n,t_x):
    t_x = resize_cpu(t_x, n<<3)
    return [
        from_coeff_vec(t_x[0:n]),
        from_coeff_vec(t_x[n:2 * n]),
        from_coeff_vec(t_x[2 * n:3 * n]),
        from_coeff_vec(t_x[3 * n:4 * n]),
        from_coeff_vec(t_x[4 * n:5 * n]),
        from_coeff_vec(t_x[5 * n:6 * n]),
        from_coeff_vec(t_x[6 * n:7 * n]),
        from_coeff_vec(t_x[7 * n:])
    ]


class gen_proof:
    def __init__ (self):
        pass
       
    def __call__(self,pp, pk: Prover_Key, cs: StandardComposer, transcript: transcript.Transcript):
        #get FFT domaim
        domain=Radix2EvaluationDomain.new(cs.circuit_bound())
        n=domain.size
        transcript.append_pi(b"pi", fr.Fr(torch.tensor(cs.public_inputs,dtype=torch.BLS12_381_Fr_G1_Mont)), int(cs.intended_pi_pos))

        #1. Compute witness Polynomials
        w_l_scalar=torch.tensor(np.load(date_set2[3],allow_pickle=True),dtype=torch.BLS12_381_Fr_G1_Mont)
        w_r_scalar=torch.tensor(np.load(date_set2[4],allow_pickle=True),dtype=torch.BLS12_381_Fr_G1_Mont)
        w_o_scalar=torch.tensor(np.load(date_set2[5],allow_pickle=True),dtype=torch.BLS12_381_Fr_G1_Mont)
        w_4_scalar=torch.tensor(np.load(date_set2[6],allow_pickle=True),dtype=torch.BLS12_381_Fr_G1_Mont)
        w_l_scalar = w_l_scalar.to('cuda')
        w_r_scalar = w_r_scalar.to('cuda')
        w_o_scalar = w_o_scalar.to('cuda')
        w_4_scalar = w_4_scalar.to('cuda')

        w_l_scalar_intt=INTT(n,w_l_scalar)
        w_r_scalar_intt=INTT(n,w_r_scalar)
        w_o_scalar_intt=INTT(n,w_o_scalar)
        w_4_scalar_intt=INTT(n,w_4_scalar)
      
        w_l_poly = from_coeff_vec(w_l_scalar_intt) 
        w_r_poly = from_coeff_vec(w_r_scalar_intt)
        w_o_poly = from_coeff_vec(w_o_scalar_intt)
        w_4_poly = from_coeff_vec(w_4_scalar_intt)


        w_polys = [kzg10.LabeledPoly.new(label="w_l_poly",hiding_bound=None,poly=w_l_poly),kzg10.LabeledPoly.new(label="w_r_poly",hiding_bound=None,poly=w_r_poly),
                   kzg10.LabeledPoly.new(label="w_o_poly",hiding_bound=None,poly=w_o_poly),kzg10.LabeledPoly.new(label="w_4_poly",hiding_bound=None,poly=w_4_poly)]

        
        w_commits, w_rands = kzg10.commit_poly_new(pp,w_polys) 

        # w_c_1=transtompz(w_commits[0].commitment.value)
        # w_c_2=transtompz(w_commits[1].commitment.value)
        # w_c_3=transtompz(w_commits[2].commitment.value)
        # w_c_4=transtompz(w_commits[3].commitment.value)
        transcript.append(b"w_l",w_commits[0].commitment.value)
        transcript.append(b"w_r",w_commits[1].commitment.value)
        transcript.append(b"w_o",w_commits[2].commitment.value)
        transcript.append(b"w_4",w_commits[3].commitment.value)
        
        #2. Derive lookup polynomials

        # Generate table compression factor
        zeta = transcript.challenge_scalar(b"zeta")
        transcript.append(b"zeta",zeta)

        pk_lookup=pk["lookup"].tolist()
        pk_lookup_table1=torch.tensor(pk_lookup["table1"]['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        pk_lookup_table2=torch.tensor(pk_lookup["table2"]['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        pk_lookup_table3=torch.tensor(pk_lookup["table3"]['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        pk_lookup_table4=torch.tensor(pk_lookup["table4"]['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        # Compress lookup table into vector of single elements
        concatenated_lookup=torch.stack([
        pk_lookup_table1,
        pk_lookup_table2,
        pk_lookup_table3,
        pk_lookup_table4], dim=0)

        t_multiset = multiset.MultiSet(concatenated_lookup)

        
        compressed_t_multiset = t_multiset.compress(zeta)     
        # Compute table poly

        compressed_t_poly =INTT(n,compressed_t_multiset.elements)
        table_poly = from_coeff_vec(compressed_t_poly)

        
        # Compute query table f
        # When q_lookup[i] is zero the wire value is replaced with a dummy
        # value currently set as the first row of the public table
        # If q_lookup[i] is one the wire values are preserved
        # This ensures the ith element of the compressed query table
        # is an element of the compressed lookup table even when
        # q_lookup[i] is 0 so the lookup check will pass
        q_lookup_pad = [0 for _ in range(n-len(cs.q_lookup))]
        padded_q_lookup = list(cs.q_lookup) + q_lookup_pad

        f_scalars = multiset.MultiSet([[],[],[],[]])
        f_scalars.elements[0]=torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        f_scalars.elements[1]=torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        f_scalars.elements[2]=torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        f_scalars.elements[3]=torch.zeros(n,4,dtype=torch.BLS12_381_Fr_G1_Mont)
        index=0
        for q_lookup, w_l, w_r, w_o, w_4 in zip(padded_q_lookup, w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar):
            if np.array_equal(q_lookup, np.zeros(4, dtype = np.uint64)):
                f_scalars.elements[0][index]=compressed_t_multiset.elements[0]
                for key in range(1,4):
                        f_scalars.elements[key][index]=torch.tensor([0,0,0,0],dtype=torch.BLS12_381_Fr_G1_Mont)
            else:
                f_scalars.elements[0][index]=w_l
                f_scalars.elements[1][index]=w_r
                f_scalars.elements[2][index]=w_o
                f_scalars.elements[3][index]=w_4

            index=index+1

        # Compress all wires into a single vector
        concatenated_f_scalars=torch.stack([
        f_scalars.elements[0],
        f_scalars.elements[1],
        f_scalars.elements[2],
        f_scalars.elements[3]], dim=0)
        f_scalars=multiset.MultiSet(concatenated_f_scalars)
        compressed_f_multiset = f_scalars.compress(zeta)

        # Compute query poly
        compressed_f_poly = INTT(n,compressed_f_multiset.elements)
        f_poly= from_coeff_vec(compressed_f_poly)
        f_polys = [kzg10.LabeledPoly.new(label="f_poly",hiding_bound=None,poly=f_poly)]

        # Commit to query polynomial
        f_poly_commit, _ = kzg10.commit_poly_new(pp,f_polys)
        transcript.append(b"f",f_poly_commit[0].commitment.value)

        # Compute s, as the sorted and concatenated version of f and t 
        # work on cpu
        h_1, h_2 = compressed_t_multiset.combine_split(compressed_f_multiset)

        # Compute h polys
        h_1=h_1.to('cuda')
        h_2=h_2.to('cuda')
        h_1_temp = INTT(n,h_1)
        h_2_temp = INTT(n,h_2)
        h_1_poly = from_coeff_vec(h_1_temp)  
        h_2_poly = from_coeff_vec(h_2_temp)  
        
        # Commit to h polys
        h_1_polys = [kzg10.LabeledPoly.new(label="h_1_poly",hiding_bound=None,poly=h_1_poly)]
        h_2_polys = [kzg10.LabeledPoly.new(label="h_2_poly",hiding_bound=None,poly=h_2_poly)]
        h_1_poly_commit,_ = kzg10.commit_poly_new(pp,h_1_polys)
        h_2_poly_commit,_ = kzg10.commit_poly_new(pp,h_2_polys)

        # Add h polynomials to transcript
        transcript.append(b"h1", h_1_poly_commit[0].commitment.value)
        transcript.append(b"h2", h_2_poly_commit[0].commitment.value)

        # 3. Compute permutation polynomial

        # Compute permutation challenge `beta`.
        beta = transcript.challenge_scalar(b"beta")
        transcript.append(b"beta", beta)
    
        # Compute permutation challenge `gamma`.
        gamma = transcript.challenge_scalar(b"gamma")
        transcript.append(b"gamma", gamma)
        
        # Compute permutation challenge `delta`.
        delta = transcript.challenge_scalar(b"delta")
        transcript.append(b"delta", delta)
        
        # Compute permutation challenge `epsilon`.
        epsilon = transcript.challenge_scalar(b"epsilon")
        transcript.append(b"epsilon", epsilon)
    

        # Challenges must be different
        assert torch.equal(beta.value, gamma.value) == False, "challenges must be different"
        assert torch.equal(beta.value, delta.value) == False, "challenges must be different"
        assert torch.equal(beta.value, epsilon.value) == False, "challenges must be different"
        assert torch.equal(gamma.value, delta.value) == False, "challenges must be different"
        assert torch.equal(gamma.value, epsilon.value) == False, "challenges must be different"
        assert torch.equal(delta.value, epsilon.value) == False, "challenges must be different"
        
        pk_permutation=pk["permutation"].tolist()
        z_poly = mod.compute_permutation_poly(domain,
            (w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar),
            beta.value,
            gamma.value,
            [
                torch.tensor(pk_permutation['left_sigma']['coeffs'], dtype = fr.Fr.Dtype),
                torch.tensor(pk_permutation['right_sigma']['coeffs'], dtype = fr.Fr.Dtype),
                torch.tensor(pk_permutation['out_sigma']['coeffs'], dtype = fr.Fr.Dtype),
                torch.tensor(pk_permutation['fourth_sigma']['coeffs'], dtype = fr.Fr.Dtype)
            ]
            )
        # Commit to permutation polynomial.

        z_polys = [kzg10.LabeledPoly.new(label="z_poly",hiding_bound=None,poly=z_poly)]
        z_poly_commit,_ = kzg10.commit_poly_new(pp,z_polys)

        # Add permutation polynomial commitment to transcript.
        transcript.append(b"z", z_poly_commit[0].commitment.value)
        
        # Compute mega permutation polynomial.
        # Compute lookup permutation poly
        z_2_poly = mod.compute_lookup_permutation_poly(
            n,
            compressed_f_multiset.elements,
            compressed_t_multiset.elements,
            h_1,
            h_2,
            delta.value,
            epsilon.value
        )
        # Commit to lookup permutation polynomial.
        z_2_polys = [kzg10.LabeledPoly.new(label="z_2_poly",hiding_bound=None,poly=z_2_poly)]
        z_2_poly_commit,_ = kzg10.commit_poly_new(pp,z_2_polys)

        # 3. Compute public inputs polynomial
        cs_public_inputs = torch.tensor(cs.public_inputs, dtype = fr.Fr.Dtype)
        pi_poly = into_dense_poly(cs_public_inputs,int(cs.intended_pi_pos),n)


        # 4. Compute quotient polynomial
        # Compute quotient challenge `alpha`, and gate-specific separation challenges.
        alpha = transcript.challenge_scalar(b"alpha")
        transcript.append(b"alpha", alpha)
        
        range_sep_challenge = transcript.challenge_scalar(b"range separation challenge")
        transcript.append(b"range seperation challenge", range_sep_challenge)

        logic_sep_challenge = transcript.challenge_scalar(b"logic separation challenge")
        transcript.append(b"logic seperation challenge", logic_sep_challenge)

        fixed_base_sep_challenge = transcript.challenge_scalar(b"fixed base separation challenge")
        transcript.append(b"fixed base separation challenge", fixed_base_sep_challenge)

        var_base_sep_challenge = transcript.challenge_scalar(b"variable base separation challenge")
        transcript.append(b"variable base separation challenge", var_base_sep_challenge)

        lookup_sep_challenge = transcript.challenge_scalar(b"lookup separation challenge")
        transcript.append(b"lookup separation challenge", lookup_sep_challenge)

        
        t_poly = quotient_poly.compute_quotient_poly(
            n,pk,
            z_poly, z_2_poly,
            w_l_poly, w_r_poly, w_o_poly, w_4_poly,
            pi_poly,
            f_poly, table_poly, h_1_poly, h_2_poly,
            alpha.value, beta.value, gamma.value, delta.value, epsilon.value, zeta.value,
            range_sep_challenge.value, logic_sep_challenge.value,
            fixed_base_sep_challenge.value,
            var_base_sep_challenge.value,
            lookup_sep_challenge.value)
        
        t_i_poly = split_tx_poly(n, t_poly)

        t_i_polys = [kzg10.LabeledPoly.new(label="t_i_polys[0]",hiding_bound=None,poly=t_i_poly[0]),
                    kzg10.LabeledPoly.new(label="t_i_polys[1]",hiding_bound=None,poly=t_i_poly[1]),
                    kzg10.LabeledPoly.new(label="t_i_polys[2]",hiding_bound=None,poly=t_i_poly[2]),
                    kzg10.LabeledPoly.new(label="t_i_polys[3]",hiding_bound=None,poly=t_i_poly[3]),
                    kzg10.LabeledPoly.new(label="t_i_polys[4]",hiding_bound=None,poly=t_i_poly[4]),
                    kzg10.LabeledPoly.new(label="t_i_polys[5]",hiding_bound=None,poly=t_i_poly[5]),
                    kzg10.LabeledPoly.new(label="t_i_polys[6]",hiding_bound=None,poly=t_i_poly[6]),
                    kzg10.LabeledPoly.new(label="t_i_polys[7]",hiding_bound=None,poly=t_i_poly[7])]
        
        t_commits, _ = kzg10.commit_poly_new(pp,t_i_polys)
        
        # Add quotient polynomial commitments to transcript
        for i in range(0, 8):
            transcript.append(f"t_{i+1}".encode(), t_commits[i].commitment.value)

        # 4. Compute linearisation polynomial
        # Compute evaluation challenge `z`.
        z_challenge = transcript.challenge_scalar(b"z")
        transcript.append(b"z", z_challenge)
        lin_poly, evaluations = linearisation_poly.compute_linearisation_poly(
                domain,
                pk,
                alpha.value, beta.value, gamma.value, delta.value, epsilon.value, zeta.value,
                range_sep_challenge.value,
                logic_sep_challenge.value,
                fixed_base_sep_challenge.value,
                var_base_sep_challenge.value,
                lookup_sep_challenge.value,
                z_challenge.value,
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

        # for work on cpu append still uses gmp
        evaluations.wire_evals.a_eval = evaluations.wire_evals.a_eval.tolist()
        evaluations.wire_evals.b_eval = evaluations.wire_evals.b_eval.tolist()
        evaluations.wire_evals.c_eval = evaluations.wire_evals.c_eval.tolist()
        evaluations.wire_evals.d_eval = evaluations.wire_evals.d_eval.tolist()
        evaluations.perm_evals.left_sigma_eval = evaluations.perm_evals.left_sigma_eval.tolist()
        evaluations.perm_evals.right_sigma_eval = evaluations.perm_evals.right_sigma_eval.tolist()
        evaluations.perm_evals.out_sigma_eval = evaluations.perm_evals.out_sigma_eval.tolist()
        evaluations.perm_evals.permutation_eval = evaluations.perm_evals.permutation_eval.tolist()
        evaluations.lookup_evals.f_eval = evaluations.lookup_evals.f_eval.tolist()
        evaluations.lookup_evals.q_lookup_eval = evaluations.lookup_evals.q_lookup_eval.tolist()
        evaluations.lookup_evals.z2_next_eval = evaluations.lookup_evals.z2_next_eval.tolist()
        evaluations.lookup_evals.h1_eval = evaluations.lookup_evals.h1_eval.tolist()
        evaluations.lookup_evals.h1_next_eval = evaluations.lookup_evals.h1_next_eval.tolist()
        evaluations.lookup_evals.h2_eval = evaluations.lookup_evals.h2_eval.tolist()

        wea=from_list_gmpy_1(evaluations.wire_evals.a_eval)
        web=from_list_gmpy_1(evaluations.wire_evals.b_eval)
        wec=from_list_gmpy_1(evaluations.wire_evals.c_eval)
        wed=from_list_gmpy_1(evaluations.wire_evals.d_eval)
        lse=from_list_gmpy_1(evaluations.perm_evals.left_sigma_eval)
        rse=from_list_gmpy_1(evaluations.perm_evals.right_sigma_eval)
        ose=from_list_gmpy_1(evaluations.perm_evals.out_sigma_eval)
        ppe=from_list_gmpy_1(evaluations.perm_evals.permutation_eval)
        lef=from_list_gmpy_1(evaluations.lookup_evals.f_eval)
        leqe=from_list_gmpy_1(evaluations.lookup_evals.q_lookup_eval)
        leze=from_list_gmpy_1(evaluations.lookup_evals.z2_next_eval)
        lehe1=from_list_gmpy_1(evaluations.lookup_evals.h1_eval)
        lehne=from_list_gmpy_1(evaluations.lookup_evals.h1_next_eval)
        lehe2=from_list_gmpy_1(evaluations.lookup_evals.h2_eval)
        
        # Add evaluations to transcript.
        # First wire evals
        transcript.append(b"a_eval",wea)
        transcript.append(b"b_eval", web)
        transcript.append(b"c_eval", wec)
        transcript.append(b"d_eval", wed)

        # Second permutation evals
        transcript.append(b"left_sig_eval", lse)
        transcript.append(b"right_sig_eval",rse)
        transcript.append(b"out_sig_eval", ose)
        transcript.append(b"perm_eval", ppe)

        # Third lookup evals
        transcript.append(b"f_eval", lef)
        transcript.append(b"q_lookup_eval", leqe)
        transcript.append(b"lookup_perm_eval",leze)
        transcript.append(b"h_1_eval", lehe1)
        transcript.append(b"h_1_next_eval", lehne)
        transcript.append(b"h_2_eval", lehe2)

        # Fourth, all evals needed for custom gates
        for label, eval in evaluations.custom_evals.vals:
            static_label = label.encode('utf-8')
            eval=eval.tolist()
            eval=from_list_gmpy_1(eval)
            transcript.append(static_label, eval)

        # 5. Compute Openings using KZG10
        #
        # We merge the quotient polynomial using the `z_challenge` so the SRS
        # is linear in the circuit size `n`

        # Compute aggregate witness to polynomials evaluated at the evaluation
        # challenge `z`
        aw_challenge = transcript.challenge_scalar(b"aggregate_witness")

        # XXX: The quotient polynomials is used here and then in the
        # opening poly. It is being left in for now but it may not
        # be necessary. Warrants further investigation.
        # Ditto with the out_sigma poly.
        pk_permutation['left_sigma']['coeffs']=torch.tensor(pk_permutation['left_sigma']['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        pk_permutation['right_sigma']['coeffs'] =torch.tensor(pk_permutation['right_sigma']['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        pk_permutation['out_sigma']['coeffs'] =torch.tensor(pk_permutation['out_sigma']['coeffs'],dtype=torch.BLS12_381_Fr_G1_Mont)
        aw_polys = [kzg10.LabeledPoly.new(label="lin_poly",hiding_bound=None,poly=lin_poly),
                    kzg10.LabeledPoly.new(label="prover_key.permutation.left_sigma.0.clone()",hiding_bound=None,poly=pk_permutation['left_sigma']['coeffs']),
                    kzg10.LabeledPoly.new(label="prover_key.permutation.right_sigma.0.clone()",hiding_bound=None,poly=pk_permutation['right_sigma']['coeffs']),
                    kzg10.LabeledPoly.new(label="prover_key.permutation.out_sigma.0.clone()",hiding_bound=None,poly=pk_permutation['out_sigma']['coeffs']),
                    kzg10.LabeledPoly.new(label="f_poly",hiding_bound=None,poly=f_poly),
                    kzg10.LabeledPoly.new(label="h_2_poly",hiding_bound=None,poly=h_2_poly),
                    kzg10.LabeledPoly.new(label="table_poly",hiding_bound=None,poly=table_poly)]
        

        aw_commits, aw_rands = kzg10.commit_poly_new(pp,aw_polys)
        aw_opening = kzg10.open(
            pp,
            itertools.chain(aw_polys, w_polys),
            itertools.chain(aw_commits, w_commits),
            z_challenge,
            aw_challenge,
            itertools.chain(aw_rands, w_rands),
            None
        )

        saw_challenge = transcript.challenge_scalar(b"aggregate_witness")
        saw_polys = [kzg10.LabeledPoly.new(label="z_poly",hiding_bound=None,poly=z_poly),
                    kzg10.LabeledPoly.new(label="w_l_poly",hiding_bound=None,poly=w_l_poly),
                    kzg10.LabeledPoly.new(label="w_r_poly",hiding_bound=None,poly=w_r_poly),
                    kzg10.LabeledPoly.new(label="w_4_poly",hiding_bound=None,poly=w_4_poly),
                    kzg10.LabeledPoly.new(label="h_1_poly",hiding_bound=None,poly=h_1_poly),
                    kzg10.LabeledPoly.new(label="z_2_poly",hiding_bound=None,poly=z_2_poly),
                    kzg10.LabeledPoly.new(label="table_poly",hiding_bound=None,poly=table_poly)]
        
        saw_commits, saw_rands = kzg10.commit_poly_new(pp,saw_polys)
        saw_opening = kzg10.open(
            pp,
            saw_polys,
            saw_commits,
            z_challenge.mul(domain.element(1)),
            saw_challenge,
            saw_rands,
            None
        )

        Proof = kzg10.Proof(
                a_comm = w_commits[0].commitment.value,
                b_comm = w_commits[1].commitment.value,
                c_comm = w_commits[2].commitment.value,
                d_comm = w_commits[3].commitment.value,
                z_comm = saw_commits[0].commitment.value,
                f_comm = f_poly_commit[0].commitment.value,
                h_1_comm = h_1_poly_commit[0].commitment.value,
                h_2_comm = h_2_poly_commit[0].commitment.value,
                z_2_comm = z_2_poly_commit[0].commitment.value,
                t_1_comm = t_commits[0].commitment.value,
                t_2_comm = t_commits[1].commitment.value,
                t_3_comm = t_commits[2].commitment.value,
                t_4_comm = t_commits[3].commitment.value,
                t_5_comm = t_commits[4].commitment.value,
                t_6_comm = t_commits[5].commitment.value,
                t_7_comm = t_commits[6].commitment.value,
                t_8_comm = t_commits[7].commitment.value,
                aw_opening = aw_opening,
                saw_opening = saw_opening,
                evaluations = evaluations)
        
        def write_to_file(data, filename):
            try:
                with open(filename, 'a') as file:
                    file.write(str(data) + '\n')  
                print(f"Data is successfully written to the file {filename}")
            except Exception as e:
                print(f"An error occurred while writing to the file: {e}")


        attributes = vars(Proof)
        for attribute, value in attributes.items():
            try:
                my_data = {f"{attribute}: {value[0]},{value[1]}"}
                print(f"{attribute}: {value[0]},{value[1]}")
                write_to_file(my_data, 'proof_data_new.txt')
            except :  # Check whether the attribute is of openproof type
                try:
                    my_data = {f"{attribute}: {value.w[0]},{value.w[1]}"}
                    print(f"{attribute}: {value.w[0]},{value.w[1]}")
                    write_to_file(my_data, 'proof_data_new.txt')
                except:
                    pass
        
        return Proof




