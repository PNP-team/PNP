import time
from torch.py_plonk.load import read_pk_data,read_pp_data,read_cs_data
from torch.py_plonk.composer import StandardComposer
from torch.py_plonk.gen_proof import gen_proof
from torch.py_plonk.transcript import transcript


if __name__ == "__main__":

    pp_file = "torch/py_plonk/params.txt"
    pk_file = "torch/py_plonk/pk.txt"
    cs_file = "torch/py_plonk/cs.txt"

    pp = read_pp_data(pp_file)
    pk = read_pk_data(pk_file)
    csdata = read_cs_data(cs_file)

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

    pi = gen_proof(pp,pk,cs,preprocessed_transcript)
