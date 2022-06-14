import hashlib
from unittest import TestCase, TestSuite, TextTestRunner

_BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def run(test):
    suite = TestSuite()
    suite.addTest(test)
    TextTestRunner().run(suite)


def hash256(s: bytes) -> bytes:
    return hashlib.sha256(
        hashlib.sha256(s).digest()
    ).digest()


def hash160(s: bytes) -> bytes:
    return hashlib.new('ripemd160', hashlib.sha256(s).digest()).digest()


def encode_base58(s: bytes) -> str:
    count = 0
    for c in s:
        if c == 0:
            count += 1
        else:
            break
    num = int.from_bytes(s, 'big')
    prefix = '1' * count
    result = ''
    while num > 0:
        num, mod = divmod(num, 58)
        result = _BASE58_ALPHABET[mod] + result
    return prefix + result


def encode_base58_checksum(b: bytes) -> str:
    return encode_base58(b + hash256(b)[:4])


def decode_base58(s: str) -> bytes:
    num = 0
    for c in s:
        num *= 58
        num += _BASE58_ALPHABET.index(c)
    combined = num.to_bytes(25, byteorder='big')
    checksum = combined[-4:]
    if hash256(combined[:-4])[:4] != checksum:
        raise ValueError('bad address: {} {}'.format(checksum, hash256(combined[:-4])[:4]))
    return combined[1:-4]


def little_endian_to_int(b: bytes) -> int:
    '''little_endian_to_int takes byte sequence as a little-endian number.
    Returns an integer'''
    # use int.from_bytes()
    return int.from_bytes(b, 'little')


def int_to_little_endian(n: int, length: int) -> bytes:
    '''endian_to_little_endian takes an integer and returns the little-endian
    byte sequence of length'''

    # use n.to_bytes()
    return n.to_bytes(length, 'little')


class HelperTest(TestCase):
    def test_little_endian_to_int(self):
        h = bytes.fromhex('99c3980000000000')
        want = 10011545
        self.assertEqual(little_endian_to_int(h), want)
        h = bytes.fromhex('a135ef0100000000')
        want = 32454049
        self.assertEqual(little_endian_to_int(h), want)

    def test_int_to_little_endian(self):
        n = 1
        want = b'\x01\x00\x00\x00'
        self.assertEqual(int_to_little_endian(n, 4), want)
        n = 10011545
        want = b'\x99\xc3\x98\x00\x00\x00\x00\x00'
        self.assertEqual(int_to_little_endian(n, 8), want)
