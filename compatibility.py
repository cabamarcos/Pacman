"""Small Python 3 replacement for the legacy python-future division helper."""
from numbers import Integral


def old_div(left, right):
    """Preserve integer division only when both operands are integral."""
    return left // right if isinstance(left, Integral) and isinstance(right, Integral) else left / right
