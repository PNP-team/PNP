from .transcript import flags
import torch
import torch.nn.functional as F
from .serialize import buffer_byte_size
from .arithmetic import neg_fq
from .ele import into_repr_fq
from .bytes import write
def is_zero_AffinePointG1(self):
        one = torch.tensor([8505329371266088957, 17002214543764226050, 6865905132761471162, 8632934651105793861, 6631298214892334189, 1582556514881692819],dtype=torch.BLS12_381_Fq_G1_Mont)
        return torch.equal(self[0] ,torch.zeros(1,6,dtype=torch.BLS12_381_Fq_G1_Mont) )and  torch.equal(self[1] ,one) # x=0 y=one

def write_field(self,writer):
    content = into_repr_fq(self)
    new_writer = write(content,writer)
    return new_writer

def serialize_with_flags(self, writer:list, flag:flags):# fq
        if flag.BIT_SIZE > 8:
            print("Not enough space")
            return
        MODULUS_BITS=381
        BYTE_SIZE_fq = 48
        output_byte_size = buffer_byte_size( MODULUS_BITS+ flag.BIT_SIZE)

        bytes = bytearray(BYTE_SIZE_fq+1)
        modified_bytes = write_field(self,bytes[:BYTE_SIZE_fq])
        bytes = modified_bytes+bytes[BYTE_SIZE_fq:]
        bytes[output_byte_size - 1] |= flag.u8_bitmask()
        writer.extend(bytes[:output_byte_size])
        return writer


def compare_tensors(tensor_a, tensor_b):
    """
    Compare two tensors representing large numbers in little-endian format.
    
    Args:
    tensor_a (torch.Tensor): The first tensor to compare.
    tensor_b (torch.Tensor): The second tensor to compare.
    
    Returns:
    int: 1 if tensor_a > tensor_b, -1 if tensor_a < tensor_b, 0 if equal
    """
    # Start comparing from the highest index (most significant digit)
    tensor_a=tensor_a.tolist()
    tensor_b=tensor_b.tolist()
    for i in reversed(range(len(tensor_a))):
        if tensor_a[i] > tensor_b[i]:
            return 1
        elif tensor_a[i] < tensor_b[i]:
            return -1
    return 0  # If all digits are the same, the numbers are equal


def serialize(item,writer): # item AffinePointG1
    if is_zero_AffinePointG1(item):
        flag = flags.SWFlags.infinity()
        zero = torch.tensor(1,6,dtype=torch.BLS12_381_Fq_G1_Mont)
        writer = serialize_with_flags(zero,writer,flag)
        return writer
    else: 
        neg_y = neg_fq(item[1])
        a = into_repr_fq(item[1])
        b = into_repr_fq(neg_y)
        #flag = flags.SWFlags.from_y_sign(a > b)
        truth_val=compare_tensors(a,b)
        flag = flags.SWFlags.from_y_sign(truth_val)
        writer = serialize_with_flags(item[0],writer, flag)
        return writer 
          
    