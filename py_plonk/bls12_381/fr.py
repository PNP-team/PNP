import gmpy2
import torch
from ..field import field
from ..transcript import flags
from ..serialize import buffer_byte_size
from ..bytes import read

class Fr(field):
    def __init__(self, value: torch.tensor):
        self.value = value

    Dtype = torch.BLS12_381_Fr_G1_Mont

    Base_type = torch.BLS12_381_Fr_G1_Base
    
    TWO_ADICITY: int = 32

    Limbs: int = 4

    TWO_ADIC_ROOT_OF_UNITY = torch.tensor(
        [13381757501831005802,  6564924994866501612,  789602057691799140,6625830629041353339], 
        dtype=torch.BLS12_381_Fr_G1_Mont)
    
    MODULUS = torch.tensor(
        [18446744069414584321,  6034159408538082302,  3691218898639771653,8353516859464449352], 
        dtype=torch.BLS12_381_Fr_G1_Mont)

    MODULUS_BITS: int = 255

    CAPACITY: int = MODULUS_BITS - 1

    REPR_SHAVE_BITS: int = 1

    R = torch.tensor(
        [8589934590,  6378425256633387010, 11064306276430008309,1739710354780652911], 
        dtype=torch.BLS12_381_Fr_G1_Mont)

    R2 = torch.tensor([14526898881837571181,  3129137299524312099,   419701826671360399,
          524908885293268753], dtype=torch.BLS12_381_Fr_G1_Mont)

    R_INV = torch.tensor([1438719116766986304, 12353315018595135583,  8215699910850717290,
         1999183251322740055], dtype=torch.BLS12_381_Fr_G1_Mont)

    GENERATOR = torch.tensor([64424509425,  1721329240476523535, 18418692815241631664,
         3824455624000121028], dtype=torch.BLS12_381_Fr_G1_Mont)

    MODULUS_MINUS_ONE_DIV_TWO = torch.tensor([9223372034707292160, 12240451741123816959,  1845609449319885826,
         4176758429732224676], dtype=torch.BLS12_381_Fr_G1_Mont)
        
    T = torch.tensor([18446282274530918399,   694073334983140354,  2998690675949164552,
                  1944954707], dtype=torch.BLS12_381_Fr_G1_Mont)

    T_MINUS_ONE_DIV_TWO = torch.tensor([9223141137265459199,   347036667491570177, 10722717374829358084,
                   972477353], dtype=torch.BLS12_381_Fr_G1_Mont)

    #256bits
    BYTE_SIZE:int = 32
    
def deserialize(reader):
    output_byte_size = buffer_byte_size(Fr.MODULUS_BITS + flags.EmptyFlags.BIT_SIZE)

    masked_bytes = bytearray([0] * (Fr.BYTE_SIZE + 1))
    masked_bytes[:output_byte_size] = reader[:output_byte_size]

    flag = flags.EmptyFlags.from_u8_remove_flags(masked_bytes[output_byte_size - 1])

    element = read(masked_bytes,Fr)
    field_element = Fr.from_repr(element)
    return field_element, flag

    
def from_random_bytes(bytes: bytes):
    limbs = (len(bytes) + 1) // 8
    if flags.EmptyFlags.BIT_SIZE > 8:
        return None

    result_bytes = bytearray([0] * (limbs * 8 + 1))
    result_bytes[:len(bytes)] = bytes
    last_bytes_mask = bytearray(9)
    last_limb_mask = ((2 ** 64 - 1)>>Fr.REPR_SHAVE_BITS).to_bytes(8, byteorder='little')
    last_bytes_mask[:8] = last_limb_mask[:]
    output_byte_size = buffer_byte_size(Fr.MODULUS_BITS + flags.EmptyFlags.BIT_SIZE)
    flag_location = output_byte_size - 1
    flag_location_in_last_limb = flag_location - (8 * (limbs - 1))

    last_bytes = result_bytes[8 * (limbs - 1):]

    flags_mask = 0xFF >> (8 - flags.EmptyFlags.BIT_SIZE)
    flag = 0
    for i, (b, m) in enumerate(zip(last_bytes, last_bytes_mask)):
        if i == flag_location_in_last_limb:
            flag = b & flags_mask
        b &= m

    field_element,flag = deserialize(result_bytes[:limbs * 8])
    #flags_obj = flags.SWFlags.from_u8(flag)

    return field_element


