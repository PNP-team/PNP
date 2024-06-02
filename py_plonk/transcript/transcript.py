import torch
from dataclasses import dataclass
from ..transcript import strobe
<<<<<<< HEAD
from ..structure import serialize_BTreeMap
from ..bls12_381 import fr
import torch
from ..serialize import todo_serialize_with_flags
=======
from ..structure import BTreeMap
from ..bls12_381 import fr
from ..serialize import serialize, deserialize
>>>>>>> origin/zrji_plonk


MERLIN_PROTOCOL_LABEL = b"Merlin v1.0"


class Transcript:

    def __init__(self, label):
        self.strobe = strobe.Strobe128.new(MERLIN_PROTOCOL_LABEL)
        self.append_message(b"dom-sep", label)

    def append_message(self, label, message):
        data_len = len(message).to_bytes(4, byteorder="little")
        self.strobe.meta_ad(label, False)
        self.strobe.meta_ad(data_len, True)
        self.strobe.ad(message, False)

    def append_pi(self, label, item, pos):
<<<<<<< HEAD
        buf = []
        buf = serialize_BTreeMap(item, pos, buf)
        self.append_message(label,buf)
=======
        buf = serialize(BTreeMap(item, pos))
        self.append_message(label, buf)
>>>>>>> origin/zrji_plonk

    def append(self, label, item):
        self.append_message(label, serialize(item))

    def challenge_bytes(self, label, dest):
        data_len = len(dest).to_bytes(4, byteorder="little")
        self.strobe.meta_ad(label, False)
        self.strobe.meta_ad(data_len, True)
        modified_dest = self.strobe.prf(dest, False)
        return modified_dest

    def challenge_scalar(self, label: bytes):
        size = fr.MODULUS_BITS() // 8
        buf = bytes([0] * size)
        modified_buf = self.challenge_bytes(label, buf)
        c_s = deserialize(modified_buf)
        return c_s
