import torch

def gt_zkp(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.tolist() > b.tolist()

def padding(input: torch.Tensor, pad_len: int) -> torch.Tensor:
    return torch.padding(input, pad_len)

def compress(polylist: "List[torch.Tensor]", challenge: torch.Tensor):
    t_0 = polylist[0]
    t_1 = polylist[1]
    t_2 = polylist[2]
    t_3 = polylist[3]
    compress_poly = torch.compress(t_0, t_1, t_2, t_3, challenge)   
    return compress_poly

def compute_query_table(q_lookup: torch.Tensor, 
                        w_l_scalar: torch.Tensor, w_r_scalar: torch.Tensor, w_o_scalar: torch.Tensor,
                        w_4_scalar: torch.Tensor, t_poly: torch.Tensor, challenge: torch.Tensor) -> torch.Tensor:
    n = w_l_scalar.size(0)
    padded_q_lookup = padding(q_lookup, n - q_lookup.size(0))
    concatenated_f_scalars = torch.compute_query_table(padded_q_lookup, w_l_scalar, w_r_scalar, w_o_scalar, w_4_scalar, t_poly[0])

    f_0 = concatenated_f_scalars[0:n]
    f_1 = concatenated_f_scalars[n:2*n]
    f_2 = concatenated_f_scalars[2*n:3*n]
    f_3 = concatenated_f_scalars[3*n:4*n]
    compressed_f = torch.compress(f_0, f_1, f_2, f_3, challenge)      
    return compressed_f

