import hashlib
import hmac
from random import randint
from typing import Optional, Union
from unittest import TestCase

from typing_extensions import Self


class FieldElement:
    def __init__(self, num: int, prime: int):
        if not (0 <= num < prime):
            raise ValueError(f'{num} not in range [0, {prime})')
        self.num = num
        self.prime = prime

    def create_in_same_order(self, num: int) -> Self:
        return self.__class__(num % self.prime, self.prime)

    def check_same_order(self, other: 'FieldElement', op: str):
        if self.prime != other.prime:
            raise TypeError(f'Cannot {op} two numbers in different Fields')

    def __repr__(self):
        return f'FieldElement({self.num}, {self.prime})'

    def __eq__(self, other: object):
        if isinstance(other, FieldElement):
            return self.num == other.num and self.prime == other.prime
        return False

    def __add__(self, other: 'FieldElement') -> Self:
        if isinstance(other, FieldElement):
            self.check_same_order(other, 'add')
            return self.create_in_same_order(self.num + other.num)
        return NotImplemented

    def __sub__(self, other: 'FieldElement') -> Self:
        if isinstance(other, FieldElement):
            self.check_same_order(other, 'subtract')
            return self.create_in_same_order(self.num - other.num)
        return NotImplemented

    def __mul__(self, other: Union['FieldElement', int]) -> Self:
        if isinstance(other, FieldElement):
            self.check_same_order(other, 'multiply')
            return self.create_in_same_order(self.num * other.num)
        if isinstance(other, int):
            return self.create_in_same_order(self.num * other)
        return NotImplemented

    def __rmul__(self, other: int) -> Self:
        if isinstance(other, int):
            return self.create_in_same_order(self.num * other)
        return NotImplemented

    def __pow__(self, exponent: int) -> Self:
        if isinstance(exponent, int):
            n = exponent % (self.prime - 1)
            return self.create_in_same_order(pow(self.num, n, self.prime))
        return NotImplemented

    def __truediv__(self, other: object) -> Self:
        if isinstance(other, FieldElement):
            self.check_same_order(other, 'divide')
            return self * other ** -1
        return NotImplemented


class FieldElementTest(TestCase):

    def test_ne(self):
        a = FieldElement(2, 31)
        b = FieldElement(2, 31)
        c = FieldElement(15, 31)
        self.assertEqual(a, b)
        self.assertTrue(a != c)
        self.assertFalse(a != b)

    def test_add(self):
        a = FieldElement(2, 31)
        b = FieldElement(15, 31)
        self.assertEqual(a + b, FieldElement(17, 31))
        a = FieldElement(17, 31)
        b = FieldElement(21, 31)
        self.assertEqual(a + b, FieldElement(7, 31))

    def test_sub(self):
        a = FieldElement(29, 31)
        b = FieldElement(4, 31)
        self.assertEqual(a - b, FieldElement(25, 31))
        a = FieldElement(15, 31)
        b = FieldElement(30, 31)
        self.assertEqual(a - b, FieldElement(16, 31))

    def test_mul(self):
        a = FieldElement(24, 31)
        b = FieldElement(19, 31)
        self.assertEqual(a * b, FieldElement(22, 31))

    def test_pow(self):
        a = FieldElement(17, 31)
        self.assertEqual(a**3, FieldElement(15, 31))
        a = FieldElement(5, 31)
        b = FieldElement(18, 31)
        self.assertEqual(a**5 * b, FieldElement(16, 31))

    def test_div(self):
        a = FieldElement(3, 31)
        b = FieldElement(24, 31)
        self.assertEqual(a / b, FieldElement(4, 31))
        a = FieldElement(17, 31)
        self.assertEqual(a**-3, FieldElement(29, 31))
        a = FieldElement(4, 31)
        b = FieldElement(11, 31)
        self.assertEqual(a**-4 * b, FieldElement(13, 31))


class Point:
    def __init__(self,
                 x: Optional[FieldElement],
                 y: Optional[FieldElement],
                 a: FieldElement,
                 b: FieldElement):
        if x is None or y is None:
            if x != y:
                raise ValueError(f'x와 y가 둘 중 하나만 None일 수 없습니다.')
        elif y**2 != x**3 + a * x + b:
            raise ValueError(f'({x}, {y}) is not on the curve')
        self.a = a
        self.b = b
        self.x = x
        self.y = y

    def __eq__(self, other: object):
        if not isinstance(other, Point):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.a == other.a
            and self.b == other.b
        )

    def __repr__(self):
        return f'Point({self.x}, {self.y}, {self.a}, {self.b})'

    def __add__(self, other: 'Point') -> Self:
        if isinstance(other, Point):
            if self.a != other.a or self.b != other.b:
                raise TypeError(f'Points {self!r}, {other!r} are not on the same curve')
            if self.is_at_infinite():
                return other
            if other.is_at_infinite():
                return self
            if self.x == other.x:
                if self.y != other.y or self.y == other.y == 0 * self.y:
                    return self.new_at_infinite()
                s = (3 * self.x**2 + self.a) / (2 * self.y)
                x3 = s**2 - 2 * self.x
            else:
                s = (other.y - self.y) / (other.x - self.x)
                x3 = s**2 - self.x - other.x
            return self.__class__(
                x3,
                s * (self.x - x3) - self.y,
                self.a,
                self.b,
            )
        return NotImplemented

    def new_at_infinite(self) -> Self:
        return self.__class__(None, None, self.a, self.b)

    def is_at_infinite(self) -> bool:
        return self.x is None or self.y is None

    def __mul__(self, other: int) -> Self:
        if isinstance(other, int):
            current = self
            result = self.new_at_infinite()
            while other:
                if other & 1:
                    result += current
                current += current
                other >>= 1
            return result
        return NotImplemented

    def __rmul__(self, other: int) -> Self:
        return self.__mul__(other)


class PointTest(TestCase):

    def test_ne(self):
        a = Point(x=3, y=-7, a=5, b=7)
        b = Point(x=18, y=77, a=5, b=7)
        self.assertTrue(a != b)
        self.assertFalse(a != a)

    def test_add0(self):
        a = Point(x=None, y=None, a=5, b=7)
        b = Point(x=2, y=5, a=5, b=7)
        c = Point(x=2, y=-5, a=5, b=7)
        self.assertEqual(a + b, b)
        self.assertEqual(b + a, b)
        self.assertEqual(b + c, a)

    def test_add1(self):
        a = Point(x=3, y=7, a=5, b=7)
        b = Point(x=-1, y=-1, a=5, b=7)
        self.assertEqual(a + b, Point(x=2, y=-5, a=5, b=7))

    def test_add2(self):
        a = Point(x=-1, y=-1, a=5, b=7)
        self.assertEqual(a + a, Point(x=18, y=77, a=5, b=7))


class ECCTest(TestCase):

    def test_on_curve(self):
        prime = 223
        a = FieldElement(0, prime)
        b = FieldElement(7, prime)
        valid_points = ((192, 105), (17, 56), (1, 193))
        invalid_points = ((200, 119), (42, 99))
        for x_raw, y_raw in valid_points:
            x = FieldElement(x_raw, prime)
            y = FieldElement(y_raw, prime)
            Point(x, y, a, b)
        for x_raw, y_raw in invalid_points:
            x = FieldElement(x_raw, prime)
            y = FieldElement(y_raw, prime)
            with self.assertRaises(ValueError):
                Point(x, y, a, b)

    def test_add(self):
        # tests the following additions on curve y^2=x^3-7 over F_223:
        # (192,105) + (17,56)
        # (47,71) + (117,141)
        # (143,98) + (76,66)
        prime = 223
        a = FieldElement(0, prime)
        b = FieldElement(7, prime)

        additions = (
            # (x1, y1, x2, y2, x3, y3)
            (192, 105, 17, 56, 170, 142),
            (47, 71, 117, 141, 60, 139),
            (143, 98, 76, 66, 47, 71),
        )
        for x1, y1, x2, y2, x3, y3 in additions:
            p1 = Point(FieldElement(x1, prime), FieldElement(y1, prime), a, b)
            p2 = Point(FieldElement(x2, prime), FieldElement(y2, prime), a, b)
            p3 = Point(FieldElement(x3, prime), FieldElement(y3, prime), a, b)
            self.assertEqual(p1 + p2, p3)

    def test_rmul(self):
        # tests the following scalar multiplications
        # 2*(192,105)
        # 2*(143,98)
        # 2*(47,71)
        # 4*(47,71)
        # 8*(47,71)
        # 21*(47,71)
        prime = 223
        a = FieldElement(0, prime)
        b = FieldElement(7, prime)

        multiplications = (
            # (coefficient, x1, y1, x2, y2)
            (2, 192, 105, 49, 71),
            (2, 143, 98, 64, 168),
            (2, 47, 71, 36, 111),
            (4, 47, 71, 194, 51),
            (8, 47, 71, 116, 55),
            (21, 47, 71, None, None),
        )

        # iterate over the multiplications
        for s, x1_raw, y1_raw, x2_raw, y2_raw in multiplications:
            x1 = FieldElement(x1_raw, prime)
            y1 = FieldElement(y1_raw, prime)
            p1 = Point(x1, y1, a, b)
            # initialize the second point based on whether it's the point at infinity
            if x2_raw is None:
                p2 = Point(None, None, a, b)
            else:
                x2 = FieldElement(x2_raw, prime)
                y2 = FieldElement(y2_raw, prime)
                p2 = Point(x2, y2, a, b)

            # check that the product is equal to the expected point
            self.assertEqual(s * p1, p2)


A = 0
B = 7
P = 2**256 - 2**32 - 977
N = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141


class S256Field(FieldElement):
    def __init__(self, num: int, prime=None):
        super().__init__(num=num, prime=P)

    def __repr__(self):
        return f'S256Field(0x{self.num:x})'

    def __str__(self):
        return f'0x{self.num:064x}'


class S256Point(Point):
    def __init__(self, x: Optional[int], y: Optional[int], a=None, b=None):
        a, b = S256Field(A), S256Field(B)
        if isinstance(x, int) and isinstance(y, int):
            super().__init__(S256Field(x), S256Field(y), a, b)
        else:
            super().__init__(x, y, a, b)

    def __repr__(self):
        if self.x is None:
            return 'S256Point(None, None)'
        return f'S256Point(0x{self.x.num}, 0x{self.y.num})'

    def __str__(self):
        if self.x is None:
            return 'infinity'
        return f'({self.x}, {self.y})'

    def __mul__(self, other: int) -> Self:
        coefficient = other % N
        return super().__mul__(coefficient)

    def verify(self, z, sig: 'Signature') -> bool:
        s_inv = pow(sig.s, N - 2, N)
        u = z * s_inv % N
        v = sig.r * s_inv % N
        total = u * G + v * self
        return total.x.num == sig.r


# tag::source10[]
G = S256Point(
    0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798,
    0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8)
# end::source10[]


class S256Test(TestCase):

    def test_order(self):
        point = N * G
        self.assertIsNone(point.x)

    def test_pubpoint(self):
        # write a test that tests the public point for the following
        points = (
            # secret, x, y
            (7, 0x5cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc, 0x6aebca40ba255960a3178d6d861a54dba813d0b813fde7b5a5082628087264da),
            (1485, 0xc982196a7466fbbbb0e27a940b6af926c1a74d5ad07128c82824a11b5398afda, 0x7a91f9eae64438afb9ce6448a1c133db2d8fb9254e4546b6f001637d50901f55),
            (2**128, 0x8f68b9d2f63b5f339239c1ad981f162ee88c5678723ea3351b7b444c9ec4c0da, 0x662a9f2dba063986de1d90c2b6be215dbbea2cfe95510bfdf23cbf79501fff82),
            (2**240 + 2**31, 0x9577ff57c8234558f293df502ca4f09cbc65a6572c842b39b366f21717945116, 0x10b49c67fa9365ad7b90dab070be339a1daf9052373ec30ffae4f72d5e66d053),
        )

        # iterate over points
        for secret, x, y in points:
            # initialize the secp256k1 point (S256Point)
            point = S256Point(x, y)
            # check that the secret*G is the same as the point
            self.assertEqual(secret * G, point)

    def test_verify(self):
        point = S256Point(
            0x887387e452b8eacc4acfde10d9aaf7f6d9a0f975aabb10d006e4da568744d06c,
            0x61de6d95231cd89026e286df3b6ae4a894a3378e393e93a0f45b666329a0ae34)
        z = 0xec208baa0fc1c19f708a9ca96fdeff3ac3f230bb4a7ba4aede4942ad003c0f60
        r = 0xac8d1c87e51d0d441be8b3dd5b05c8795b48875dffe00b7ffcfac23010d3a395
        s = 0x68342ceff8935ededd102dd876ffd6ba72d6a427a3edb13d26eb0781cb423c4
        self.assertTrue(point.verify(z, Signature(r, s)))
        z = 0x7c076ff316692a3d7eb3c3bb0f8b1488cf72e1afcd929e29307032997a838a3d
        r = 0xeff69ef2b1bd93a66ed5219add4fb51e11a840f404876325a1e8ffe0529a2c
        s = 0xc7207fee197d27c618aea621406f6bf5ef6fca38681d82b2f06fddbdce6feab6
        self.assertTrue(point.verify(z, Signature(r, s)))


class Signature:
    def __init__(self, r, s):
        self.r = r
        self.s = s

    def __repr__(self):
        return f'Signature(0x{self.r:x}, 0x{self.s:x})'


class PrivateKey:
    def __init__(self, secret):
        self.secret = secret
        self.point = secret * G

    def hex(self):
        return '{:x}'.format(self.secret).zfill(64)

    def sign(self, z):
        k = self.deterministic_k(z)
        r = (k * G).x.num
        k_inv = pow(k, N - 2, N)
        s = (z + r * self.secret) * k_inv % N
        if s > N / 2:
            s = N - s
        return Signature(r, s)

    def deterministic_k(self, z):
        k = b'\x00' * 32
        v = b'\x01' * 32
        if z > N:
            z -= N
        z_bytes = z.to_bytes(32, 'big')
        secret_bytes = self.secret.to_bytes(32, 'big')
        s256 = hashlib.sha256
        k = hmac.new(k, v + b'\x00' + secret_bytes + z_bytes, s256).digest()
        v = hmac.new(k, v, s256).digest()
        k = hmac.new(k, v + b'\x01' + secret_bytes + z_bytes, s256).digest()
        v = hmac.new(k, v, s256).digest()
        while True:
            v = hmac.new(k, v, s256).digest()
            candidate = int.from_bytes(v, 'big')
            if candidate >= 1 and candidate < N:
                return candidate  # <2>
            k = hmac.new(k, v + b'\x00', s256).digest()
            v = hmac.new(k, v, s256).digest()
    # end::source14[]


class PrivateKeyTest(TestCase):

    def test_sign(self):
        pk = PrivateKey(randint(0, N))
        z = randint(0, 2**256)
        sig = pk.sign(z)
        self.assertTrue(pk.point.verify(z, sig))
