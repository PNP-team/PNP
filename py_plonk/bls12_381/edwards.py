import gmpy2
from dataclasses import dataclass
from ..bls12_381.fr import Fr as Fq
from ..bls12_381.edwards_fr import Fr
import torch


GENERATOR_X = Fq(gmpy2.mpz(8076246640662884909881801758704306714034609987455869804520522091855516602923))
GENERATOR_Y = Fq(gmpy2.mpz(13262374693698910701929044844600465831413122818447359594527400194675274060458))
@dataclass
class EdwardsParameters:
    COEFF_A = torch.tensor([18446744060824649731, 18102478225614246908, 11073656695919314959, 6613806504683796440],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')

    COEFF_D = torch.tensor([3049539848285517488, 18189135023605205683, 8793554888777148625, 6339087681201251886],dtype=torch.BLS12_381_Fr_G1_Mont).to('cuda')
    
    COFACTOR = [8]

    COFACTOR_INV = Fr(gmpy2.mpz(819310549611346726241370945440405716213240158234039660170669895299022906775))
    
    # AFFINE_GENERATOR_COEFFS = (GENERATOR_X, GENERATOR_Y)
    AFFINE_GENERATOR_COEFFS = (GENERATOR_X, GENERATOR_Y)

