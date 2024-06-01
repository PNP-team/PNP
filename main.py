import time
from py_plonk.composer import StandardComposer
from py_plonk.gen_proof import gen_proof
from py_plonk.transcript import transcript
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

#data_set2=["../../data/MERKLE-HEIGHT-3/pp-3.npz","../../data/MERKLE-HEIGHT-3/pk-3.npz","../../data/MERKLE-HEIGHT-3/cs-3.npz","../../data/MERKLE-HEIGHT-3/w_l_scalar-3.npy","../../data/MERKLE-HEIGHT-3/w_r_scalar-3.npy","../../data/MERKLE-HEIGHT-3/w_o_scalar-3.npy","../../data/MERKLE-HEIGHT-3/w_4_scalar-3.npy"]
data_set2=["../../data/MERKLE-HEIGHT-9/pp-9.npz","../../data/MERKLE-HEIGHT-9/pk-9.npz","../../data/MERKLE-HEIGHT-9/cs-9.npz","../../data/MERKLE-HEIGHT-9/w_l_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_r_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_o_scalar-9.npy","../../data/MERKLE-HEIGHT-9/w_4_scalar-9.npy"]
if __name__ == "__main__":

    start_time = time.time()
    pp = np.load(data_set2[0],allow_pickle=True)
    pk = np.load(data_set2[1], allow_pickle=True)
    csdata = np.load(data_set2[2],allow_pickle=True)
    end_time = time.time()
    load_time = end_time - start_time
    print(f"load time: {load_time} s")
    cs=StandardComposer(n=csdata["n"],q_m=csdata["q_m"],q_l=csdata["q_l"],q_r=csdata["q_r"],
                        q_o=csdata["q_o"],q_4=csdata["q_4"],q_c=csdata["q_c"],q_hl=csdata["q_hl"],
                        q_hr=csdata["q_hr"],q_h4=csdata["q_h4"],q_arith=csdata["q_arith"],
                        q_range=csdata["q_range"],q_logic=csdata["q_logic"],
                        q_fixed_group_add=csdata["q_fixed"],public_inputs=csdata["public_inputs"],
                        q_variable_group_add=csdata["q_variable"],
                        q_lookup=csdata["q_lookup"],intended_pi_pos=csdata["intended_pi_pos"],
                        w_l=csdata["w_l"],w_r=csdata["w_r"],w_o=csdata["w_o"],w_4=csdata["w_4"],
                        lookup_table=csdata["lookup_table"],zero_var=csdata["zero_var"])

    
    
    transcript_init = b"Merkle tree"
    preprocessed_transcript = transcript.Transcript.new(transcript_init)
    # with profile(activities=[ProfilerActivity.CPU]) as prof:
    #     with record_function("gen_proof"):
    # pi = gen_proof(pp,pk,cs,preprocessed_transcript)
    model = gen_proof()

    y = model(pp,pk,cs,preprocessed_transcript)

    end_time = time.time()
    print("Generate proof successfully\n")
    execution_time = end_time - start_time
    print(f"execution time: {execution_time} s")