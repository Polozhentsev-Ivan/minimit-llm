import math

def function(n = int) -> bool:
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")

    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    lim = math.isqrt(n)
    for d in range(3, lim + 1, 2):
        if n % d == 0:
            return False
    return True    