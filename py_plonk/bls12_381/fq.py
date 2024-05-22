import torch
from dataclasses import dataclass
from ..field import field


class Fq(field):
    def __init__(self, value: torch.Tensor):
        self.value = value
    
    Dtype = torch.BLS12_381_Fq_G1_Mont

    Base_type = torch.BLS12_381_Fq_G1_Base

    TWO_ADICITY: int = 1

    Limbs: int = 6

    TWO_ADIC_ROOT_OF_UNITY = torch.tensor([ 4897101644811774638,  3654671041462534141,  9116311052841768592,
        17053147383018470266, 17227549637287919721,  4659873644253552736],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    MODULUS = torch.tensor([13402431016077863595,  2210141511517208575,  7435674573564081700,
         7239337960414712511,  5412103778470702295,  1873798617647539866],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    MODULUS_BITS: int = 381

    CAPACITY: int = MODULUS_BITS - 1

    REPR_SHAVE_BITS: int = 3

    R = torch.tensor([ 8505329371266088957, 17002214543764226050,  6865905132761471162,
         8632934651105793861,  6631298214892334189,  1582556514881692819],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    R2 = torch.tensor([17644856173732828998,   754043588434789617, 10224657059481499349,
         7488229067341005760, 11130996698012816685,  1267921511277847466],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    R_INV = torch.tensor([17641587485044328480,  9214666605636156155,  3764632676029787592,
         1700015488230788585, 17104218554478056331,  1512865335860727017],
       dtype=torch.BLS12_381_Fq_G1_Mont)
    
    GENERATOR = torch.tensor([ 3608227726454314319, 13347543502301691909,  6296135691958860625,
        10026531341796875211,  7850492651313966083,  1291314412115845772],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    MODULUS_MINUS_ONE_DIV_TWO = torch.tensor([15924587544893707605, 17681132092137668592, 12941209323636816658,
        12843041017062132063,  2706051889235351147, 14990388941180318928],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    T = torch.tensor([15924587544893707605, 17681132092137668592, 12941209323636816658,
        12843041017062132063,  2706051889235351147, 14990388941180318928],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    T_MINUS_ONE_DIV_TWO = torch.tensor([17185665809301629610,  8840566046068834288, 15693976698673184137,
        15644892545385841839, 10576397981472451381,  7495194470590159456],
       dtype=torch.BLS12_381_Fq_G1_Mont)

    #384bits
    BYTE_SIZE: int = 48
    # # Return the Multiplicative identity
    # def one(cls):
    #     return cls(cls.R)
    
    # # Returns the 2^s root of unity.
    # def two_adic_root_of_unity(self):
    #     return self.TWO_ADIC_ROOT_OF_UNITY 

    # # Returns the 2^s * small_subgroup_base^small_subgroup_base_adicity root of unity
    # # if a small subgroup is defined.
    # def large_subgroup_root_of_unity():
    #     pass

    # # Returns the multiplicative generator of `char()` - 1 order.
    # def multiplicative_generator(cls):
    #     return cls.GENERATOR

    # # Returns the root of unity of order n, if one exists.
    # # If no small multiplicative subgroup is defined, this is the 2-adic root of unity of order n
    # # (for n a power of 2).
    # def get_root_of_unity(self,n):
    #     size = 2 ** (n.bit_length()-1)
    #     log_size_of_group = int(math.log2(size))

    #     if n != size or log_size_of_group > self.TWO_ADICITY:
    #         return None

    #     # Compute the generator for the multiplicative subgroup.
    #     # It should be 2^(log_size_of_group) root of unity.
    #     omega = self.two_adic_root_of_unity()
    #     R_inv=gmpy2.invert(self.R,self.MODULUS)
    #     for _ in range(log_size_of_group, self.TWO_ADICITY):
    #         #modsquare
    #         omega *=omega
    #         omega *=R_inv
    #         omega %=self.MODULUS
    #     return omega

FQ_ONE = 1
FQ_ZERO = 0
COEFF_A = 0
COEFF_B = 4