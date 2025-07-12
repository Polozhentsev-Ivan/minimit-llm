from src.function_for_test import function


def test_is_prime():
    cases = [
        (0, False),
        (1, False),
        (2, True),
        (3, True),
        (4, False),
        (17, True),
        (100, False),
        (7919, True),
        (8000, False),
    ]
    for value, expected in cases:
        result = function(value)
        assert result is expected, (
            f"is_prime({value}) returned {result}, expected {expected}"
        )


def test_non_int():
    try:
        function(3.111)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError for non-int input")


if __name__ == "__main__":
    test_is_prime()
    test_non_int()
    print("All tests passed.") 