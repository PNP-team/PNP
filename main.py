import time
from py_plonk.composer import StandardComposer
from py_plonk.gen_proof import gen_proof
from py_plonk.transcript import transcript
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity



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

    start_time = time.time()
    model = gen_proof()
    end_time = time.time()
    print("Load Time: {}".format(end_time - start_time))

    start_time = time.time()
    transcript_init = b"Merkle tree"
    preprocessed_transcript = transcript.Transcript(transcript_init)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        y = model(preprocessed_transcript)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # y = model(preprocessed_transcript)

    end_time = time.time()
    print("Generate proof successfully\n")
    execution_time = end_time - start_time
    print(f"execution time: {execution_time} s")
