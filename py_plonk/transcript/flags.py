class SWFlags:
    BIT_SIZE = 2

    Infinity = 0
    PositiveY = 1
    NegativeY = 2

    def __init__(self, flag):
        self.flag = flag

    @classmethod
    def infinity(cls):
        return SWFlags(cls.Infinity)

    @classmethod
    def from_y_sign(cls, is_positive: bool):
        if is_positive:
            return SWFlags(cls.PositiveY)
        else:
            return SWFlags(cls.NegativeY)

    def u8_bitmask(self):
        mask = 0
        if self.flag == self.Infinity:
            mask |= 1 << 6
        elif self.flag == self.PositiveY:
            mask |= 1 << 7
        return mask


class EmptyFlags:
    BIT_SIZE = 0

    def u8_bitmask():
        return 0
