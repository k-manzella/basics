import pytest
import basics


def test_sum_as_string():
    assert basics.sum_as_string(1, 1) == "2"

def test_is_prime():
    assert basics.is_prime(3)
    assert not basics.is_prime(4)
    assert not basics.is_prime(100)
    assert basics.is_prime(29)
