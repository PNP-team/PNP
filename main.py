import time
from py_plonk.composer import StandardComposer
from py_plonk.gen_proof import gen_proof
from py_plonk.transcript import transcript
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from py_plonk.load import load


# data_set2=["../../data/MERKLE-HEIGHT-3/pp-3.npz","../../data/MERKLE-HEIGHT-3/pk-3.npz","../../data/MERKLE-HEIGHT-3/cs-3.npz","../../data/MERKLE-HEIGHT-3/w_l_scalar-3.npy","../../data/MERKLE-HEIGHT-3/w_r_scalar-3.npy","../../data/MERKLE-HEIGHT-3/w_o_scalar-3.npy","../../data/MERKLE-HEIGHT-3/w_4_scalar-3.npy"]
data_set2 = [
    "../../data/MERKLE-HEIGHT-9/pp-9.npz",
    "../../data/MERKLE-HEIGHT-9/pk-9.npz",
    "../../data/MERKLE-HEIGHT-9/cs-9.npz",
    "../../data/MERKLE-HEIGHT-9/w_l_scalar-9.npy",
    "../../data/MERKLE-HEIGHT-9/w_r_scalar-9.npy",
    "../../data/MERKLE-HEIGHT-9/w_o_scalar-9.npy",
    "../../data/MERKLE-HEIGHT-9/w_4_scalar-9.npy",
]
if __name__ == "__main__":

    pp, pk, cs = load("../../data/MERKLE-HEIGHT-9/")

    start_time = time.time()
    # pp = np.load(data_set2[0], allow_pickle=True)
    pk_old = np.load(data_set2[1], allow_pickle=True)

    # csdata = np.load(data_set2[2], allow_pickle=True)
    # end_time = time.time()
    # load_time = end_time - start_time
    # print(f"load time: {load_time} s")
    # cs = StandardComposer(
    #     n=csdata["n"],
    #     public_inputs=csdata["public_inputs"],
    #     q_lookup=csdata["q_lookup"],
    #     intended_pi_pos=csdata["intended_pi_pos"],
    #     lookup_table=csdata["lookup_table"],
    # )

    transcript_init = b"Merkle tree"
    preprocessed_transcript = transcript.Transcript(transcript_init)

    model = gen_proof()

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     y = model(pp,pk,cs,preprocessed_transcript)
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    y = model(pp, pk_old, pk, cs, preprocessed_transcript)

    end_time = time.time()
    print("Generate proof successfully\n")
    execution_time = end_time - start_time
    print(f"execution time: {execution_time} s")
