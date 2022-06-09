import hashlib
from unittest import TestSuite, TextTestRunner


def run(test):
    suite = TestSuite()
    suite.addTest(test)
    TextTestRunner().run(suite)


def hash256(s: bytes) -> bytes:
    return hashlib.sha256(
        hashlib.sha256(s).digest()
    ).digest()
