# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import math as pymath

from .triton_compat import math, tl, triton


_PI: tl.constexpr = tl.constexpr(pymath.pi)
_HALF_PI: tl.constexpr = tl.constexpr(pymath.pi / 2.0)
_QUARTER_PI: tl.constexpr = tl.constexpr(pymath.pi / 4.0)
_INV_LN10: tl.constexpr = tl.constexpr(1.0 / pymath.log(10.0))
_LOG_SQRT_2PI: tl.constexpr = tl.constexpr(0.91893853320467274178032973640562)
_TWO_OVER_SQRT_PI: tl.constexpr = tl.constexpr(2.0 / pymath.sqrt(pymath.pi))
_SQRT_PI: tl.constexpr = tl.constexpr(pymath.sqrt(pymath.pi))
_LOG_PI: tl.constexpr = tl.constexpr(pymath.log(pymath.pi))



@triton.jit
def _ts_abs(x):
    return math.abs(x)


@triton.jit
def _ts_signbit_impl(x):
    if x.dtype is tl.float16 or x.dtype is tl.bfloat16:
        bits = x.to(tl.uint16, bitcast=True)
        return (bits & tl.full(x.shape, 0x8000, tl.uint16)) != 0
    if x.dtype is tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        return (bits & tl.full(x.shape, 0x80000000, tl.uint32)) != 0
    if x.dtype is tl.float64:
        bits = x.to(tl.uint64, bitcast=True)
        return (bits & tl.full(x.shape, 0x8000000000000000, tl.uint64)) != 0
    return x < 0


@triton.jit
def _ts_copysign_impl(x, y):
    if x.dtype is tl.float16 or x.dtype is tl.bfloat16:
        mag = x.to(tl.uint16, bitcast=True) & tl.full(x.shape, 0x7FFF, tl.uint16)
        sign = y.to(tl.uint16, bitcast=True) & tl.full(x.shape, 0x8000, tl.uint16)
        return (mag | sign).to(x.dtype, bitcast=True)
    if x.dtype is tl.float32:
        mag = x.to(tl.uint32, bitcast=True) & tl.full(x.shape, 0x7FFFFFFF, tl.uint32)
        sign = y.to(tl.uint32, bitcast=True) & tl.full(x.shape, 0x80000000, tl.uint32)
        return (mag | sign).to(tl.float32, bitcast=True)
    if x.dtype is tl.float64:
        mag = x.to(tl.uint64, bitcast=True) & tl.full(x.shape, 0x7FFFFFFFFFFFFFFF, tl.uint64)
        sign = y.to(tl.uint64, bitcast=True) & tl.full(x.shape, 0x8000000000000000, tl.uint64)
        return (mag | sign).to(tl.float64, bitcast=True)
    return tl.where(y < 0, -_ts_abs(x), _ts_abs(x))


@triton.jit
def _ts_trunc_impl(x):
    return tl.where(x >= 0, math.floor(x), math.ceil(x))


@triton.jit
def _ts_nearbyint_impl(x):
    if x.dtype is tl.float32:
        integral_limit = tl.full(x.shape, 8388608.0, x.dtype)
    else:
        integral_limit = tl.full(x.shape, 4503599627370496.0, x.dtype)
    ax = _ts_abs(x)
    flo = math.floor(ax)
    frac = ax - flo
    flo_i64 = flo.to(tl.int64)
    odd = (flo_i64 & 1) != 0
    rounded = tl.where((frac > 0.5) | ((frac == 0.5) & odd), flo + 1.0, flo)
    rounded = tl.where(ax >= integral_limit, ax, rounded)
    return _ts_copysign_impl(rounded, x)


@triton.jit
def _ts_atan_impl(x):
    ax = _ts_abs(x)
    invert = ax > 1.0
    safe_ax = tl.where(ax > 0.0, ax, 1.0)
    z = tl.where(invert, 1.0 / safe_ax, ax)
    u = z * z
    p = (((((-0.00974449 * u + 0.04739016) * u - 0.11135463) * u + 0.19140896) * u - 0.33225925) * u + 0.99995998)
    base = z * p
    result = tl.where(invert, _HALF_PI - base, base)
    return _ts_copysign_impl(result, x)


@triton.jit
def _ts_frexp_impl(x):
    zero = x == 0
    isnan = _ts_isnan_impl(x)
    isinf = _ts_isinf_impl(x)
    special = isnan | isinf

    if x.dtype is tl.float16:
        bits = x.to(tl.uint16, bitcast=True)
        sign = bits & tl.full(x.shape, 0x8000, tl.uint16)
        abs_bits = bits & tl.full(x.shape, 0x7FFF, tl.uint16)
        exp_bits = (abs_bits >> 10) & tl.full(x.shape, 0x1F, tl.uint16)
        frac_bits = abs_bits & tl.full(x.shape, 0x03FF, tl.uint16)
        normal = (exp_bits != 0) & (exp_bits != tl.full(x.shape, 0x1F, tl.uint16))
        subnormal = (exp_bits == 0) & (frac_bits != 0)
        exp_normal = exp_bits.to(tl.int32) - 14
        mant_normal_bits = sign | (tl.full(x.shape, 14, tl.uint16) << 10) | frac_bits
        mant_normal = mant_normal_bits.to(tl.float16, bitcast=True)
        frac_work = frac_bits
        exp_sub = tl.full(x.shape, -14, tl.int32)
        top = tl.full(x.shape, 0x0400, tl.uint16)
        for _ in range(10):
            need = subnormal & ((frac_work & top) == 0)
            frac_work = tl.where(need, frac_work << 1, frac_work)
            exp_sub = tl.where(need, exp_sub - 1, exp_sub)
        mant_sub_frac = (frac_work << 1) & tl.full(x.shape, 0x03FF, tl.uint16)
        mant_sub_bits = sign | (tl.full(x.shape, 14, tl.uint16) << 10) | mant_sub_frac
        mant_sub = mant_sub_bits.to(tl.float16, bitcast=True)
        mantissa = tl.where(normal, mant_normal, mant_sub)
        exponent = tl.where(normal, exp_normal, exp_sub)
        mantissa = tl.where(zero, 0.0, mantissa)
        exponent = tl.where(zero, 0, exponent)
        mantissa = tl.where(special, x, mantissa)
        exponent = tl.where(special, 0, exponent)
        return mantissa, exponent

    if x.dtype is tl.bfloat16:
        bits = x.to(tl.uint16, bitcast=True)
        sign = bits & tl.full(x.shape, 0x8000, tl.uint16)
        abs_bits = bits & tl.full(x.shape, 0x7FFF, tl.uint16)
        exp_bits = (abs_bits >> 7) & tl.full(x.shape, 0xFF, tl.uint16)
        frac_bits = abs_bits & tl.full(x.shape, 0x007F, tl.uint16)
        normal = (exp_bits != 0) & (exp_bits != tl.full(x.shape, 0xFF, tl.uint16))
        subnormal = (exp_bits == 0) & (frac_bits != 0)
        exp_normal = exp_bits.to(tl.int32) - 126
        mant_normal_bits = sign | (tl.full(x.shape, 126, tl.uint16) << 7) | frac_bits
        mant_normal = mant_normal_bits.to(tl.bfloat16, bitcast=True)
        frac_work = frac_bits
        exp_sub = tl.full(x.shape, -126, tl.int32)
        top = tl.full(x.shape, 0x0080, tl.uint16)
        for _ in range(7):
            need = subnormal & ((frac_work & top) == 0)
            frac_work = tl.where(need, frac_work << 1, frac_work)
            exp_sub = tl.where(need, exp_sub - 1, exp_sub)
        mant_sub_frac = (frac_work << 1) & tl.full(x.shape, 0x007F, tl.uint16)
        mant_sub_bits = sign | (tl.full(x.shape, 126, tl.uint16) << 7) | mant_sub_frac
        mant_sub = mant_sub_bits.to(tl.bfloat16, bitcast=True)
        mantissa = tl.where(normal, mant_normal, mant_sub)
        exponent = tl.where(normal, exp_normal, exp_sub)
        mantissa = tl.where(zero, 0.0, mantissa)
        exponent = tl.where(zero, 0, exponent)
        mantissa = tl.where(special, x, mantissa)
        exponent = tl.where(special, 0, exponent)
        return mantissa, exponent

    if x.dtype is tl.float32:
        bits = x.to(tl.uint32, bitcast=True)
        sign = bits & tl.full(x.shape, 0x80000000, tl.uint32)
        abs_bits = bits & tl.full(x.shape, 0x7FFFFFFF, tl.uint32)
        exp_bits = (abs_bits >> 23) & tl.full(x.shape, 0xFF, tl.uint32)
        frac_bits = abs_bits & tl.full(x.shape, 0x7FFFFF, tl.uint32)
        normal = (exp_bits != 0) & (exp_bits != tl.full(x.shape, 0xFF, tl.uint32))
        subnormal = (exp_bits == 0) & (frac_bits != 0)
        exp_normal = exp_bits.to(tl.int32) - 126
        mant_normal_bits = sign | (tl.full(x.shape, 126, tl.uint32) << 23) | frac_bits
        mant_normal = mant_normal_bits.to(tl.float32, bitcast=True)
        frac_work = frac_bits
        exp_sub = tl.full(x.shape, -126, tl.int32)
        top = tl.full(x.shape, 0x400000, tl.uint32)
        for _ in range(23):
            need = subnormal & ((frac_work & top) == 0)
            frac_work = tl.where(need, frac_work << 1, frac_work)
            exp_sub = tl.where(need, exp_sub - 1, exp_sub)
        mant_sub_frac = (frac_work << 1) & tl.full(x.shape, 0x7FFFFF, tl.uint32)
        mant_sub_bits = sign | (tl.full(x.shape, 126, tl.uint32) << 23) | mant_sub_frac
        mant_sub = mant_sub_bits.to(tl.float32, bitcast=True)
        mantissa = tl.where(normal, mant_normal, mant_sub)
        exponent = tl.where(normal, exp_normal, exp_sub)
        mantissa = tl.where(zero, 0.0, mantissa)
        exponent = tl.where(zero, 0, exponent)
        mantissa = tl.where(special, x, mantissa)
        exponent = tl.where(special, 0, exponent)
        return mantissa, exponent

    if x.dtype is tl.float64:
        bits = x.to(tl.uint64, bitcast=True)
        sign = bits & tl.full(x.shape, 0x8000000000000000, tl.uint64)
        abs_bits = bits & tl.full(x.shape, 0x7FFFFFFFFFFFFFFF, tl.uint64)
        exp_bits = (abs_bits >> 52) & tl.full(x.shape, 0x7FF, tl.uint64)
        frac_bits = abs_bits & tl.full(x.shape, 0xFFFFFFFFFFFFF, tl.uint64)
        normal = (exp_bits != 0) & (exp_bits != tl.full(x.shape, 0x7FF, tl.uint64))
        subnormal = (exp_bits == 0) & (frac_bits != 0)
        exp_normal = exp_bits.to(tl.int32) - 1022
        mant_normal_bits = sign | (tl.full(x.shape, 1022, tl.uint64) << 52) | frac_bits
        mant_normal = mant_normal_bits.to(tl.float64, bitcast=True)
        frac_work = frac_bits
        exp_sub = tl.full(x.shape, -1022, tl.int32)
        top = tl.full(x.shape, 0x8000000000000, tl.uint64)
        for _ in range(52):
            need = subnormal & ((frac_work & top) == 0)
            frac_work = tl.where(need, frac_work << 1, frac_work)
            exp_sub = tl.where(need, exp_sub - 1, exp_sub)
        mant_sub_frac = (frac_work << 1) & tl.full(x.shape, 0xFFFFFFFFFFFFF, tl.uint64)
        mant_sub_bits = sign | (tl.full(x.shape, 1022, tl.uint64) << 52) | mant_sub_frac
        mant_sub = mant_sub_bits.to(tl.float64, bitcast=True)
        mantissa = tl.where(normal, mant_normal, mant_sub)
        exponent = tl.where(normal, exp_normal, exp_sub)
        mantissa = tl.where(zero, 0.0, mantissa)
        exponent = tl.where(zero, 0, exponent)
        mantissa = tl.where(special, x, mantissa)
        exponent = tl.where(special, 0, exponent)
        return mantissa, exponent

    exponent = tl.where(zero, 0, (math.floor(math.log2(_ts_abs(x))) + 1).to(tl.int32))
    mantissa = tl.where(zero, 0.0, x * math.exp2(-exponent.to(x.dtype)))
    mantissa = tl.where(special, x, mantissa)
    exponent = tl.where(special, 0, exponent)
    return mantissa, exponent


@triton.jit
def _ts_hypot_impl(x, y):
    ax = _ts_abs(x)
    ay = _ts_abs(y)
    mx = tl.where(ax > ay, ax, ay)
    mn = tl.where(ax > ay, ay, ax)
    safe_mx = tl.where(mx == 0.0, 1.0, mx)
    r = mn / safe_mx
    return tl.where(mx == 0.0, 0.0, mx * math.sqrt(1.0 + r * r))


@triton.jit
def _ts_pow_impl(a, b):
    nan = tl.full(a.shape, float("nan"), a.dtype)
    inf = tl.full(a.shape, float("inf"), a.dtype)
    one = tl.full(a.shape, 1.0, a.dtype)
    zero = tl.full(a.shape, 0.0, a.dtype)

    abs_a = _ts_abs(a)
    a_isnan = _ts_isnan_impl(a)
    b_isnan = _ts_isnan_impl(b)
    b_isinf = _ts_isinf_impl(b)
    a_zero = a == 0
    a_neg = _ts_signbit_impl(a)

    if a.dtype is tl.float32:
        integral_limit = tl.full(a.shape, 8388608.0, a.dtype)
    else:
        integral_limit = tl.full(a.shape, 4503599627370496.0, a.dtype)
    b_trunc = _ts_trunc_impl(b)
    b_is_int = b == b_trunc
    b_small_int = _ts_abs(b) < integral_limit
    b_odd = b_small_int & ((b_trunc.to(tl.int64) & 1) != 0)

    safe_abs_a = tl.where(abs_a > 0, abs_a, one)
    mag = math.exp(math.log(safe_abs_a) * b)
    signed_mag = tl.where(a_neg & b_is_int & b_odd, -mag, mag)

    zero_pos = tl.where(b_odd & a_neg, _ts_copysign_impl(zero, a), zero)
    zero_neg = tl.where(b_odd & a_neg, _ts_copysign_impl(inf, a), inf)
    zero_case = tl.where(b > 0, zero_pos, zero_neg)

    finite_case = tl.where(a_neg & (~b_is_int), nan, signed_mag)
    inf_case = tl.where(
        abs_a == 1.0,
        one,
        tl.where(abs_a > 1.0, tl.where(b > 0, inf, zero), tl.where(b > 0, zero, inf)),
    )

    res = finite_case
    res = tl.where(a_zero & (b != 0), zero_case, res)
    res = tl.where(b_isinf, inf_case, res)
    res = tl.where(a_isnan | b_isnan, nan, res)
    res = tl.where((a == 1.0) | (b == 0.0), one, res)
    res = tl.where((a == -1.0) & b_isinf, one, res)
    return res


@triton.jit
def _ts_atan2_impl(y, x):
    nan = tl.full(y.shape, float("nan"), y.dtype)
    ax = _ts_abs(x)
    ay = _ts_abs(y)
    x_zero = x == 0
    y_zero = y == 0
    x_neg = _ts_signbit_impl(x)
    y_neg = _ts_signbit_impl(y)
    base = _ts_atan_impl(tl.where(ax == 0, float("inf"), ay / ax))
    pos_branch = _ts_copysign_impl(base, y)
    neg_branch = _ts_copysign_impl(_PI - base, y)
    zero_zero = tl.where(x_neg, _ts_copysign_impl(_PI, y), _ts_copysign_impl(0.0, y))
    x_zero_branch = tl.where(y_zero, zero_zero, _ts_copysign_impl(_HALF_PI, y))
    result = tl.where(x_zero, x_zero_branch, tl.where(x > 0, pos_branch, neg_branch))
    return tl.where((x != x) | (y != y), nan, result)


@triton.jit
def _ts_log1p_impl(x):
    ax = _ts_abs(x)
    u = 1.0 + x
    small = ax < 1.0e-4
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    poly = x - 0.5 * x2 + x3 / 3.0 - 0.25 * x4 + 0.2 * x5
    neg_inf = tl.full(x.shape, float("-inf"), x.dtype)
    nan = tl.full(x.shape, float("nan"), x.dtype)
    out = tl.where(small, poly, math.log(u))
    out = tl.where(x == -1.0, neg_inf, out)
    return tl.where(x < -1.0, nan, out)


@triton.jit
def _ts_expm1_impl(x):
    ax = _ts_abs(x)
    small = ax < 1.0e-5
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    poly = x + 0.5 * x2 + x3 / 6.0 + x4 / 24.0 + x5 / 120.0
    return tl.where(small, poly, math.exp(x) - 1.0)


@triton.jit
def _ts_tan_impl(x):
    return math.sin(x) / math.cos(x)


@triton.jit
def _ts_tanh_impl(x):
    ax = _ts_abs(x)
    e = math.exp(-2.0 * ax)
    return _ts_copysign_impl((1.0 - e) / (1.0 + e), x)


@triton.jit
def _ts_sinh_impl(x):
    ex = math.exp(x)
    emx = math.exp(-x)
    return 0.5 * (ex - emx)


@triton.jit
def _ts_cosh_impl(x):
    ex = math.exp(x)
    emx = math.exp(-x)
    return 0.5 * (ex + emx)


@triton.jit
def _ts_asinh_impl(x):
    return math.log(x + math.sqrt(x * x + 1.0))


@triton.jit
def _ts_acosh_impl(x):
    nan = tl.full(x.shape, float("nan"), x.dtype)
    safe_x = tl.where(x >= 1.0, x, 1.0)
    out = math.log(safe_x + math.sqrt((safe_x - 1.0) * (safe_x + 1.0)))
    return tl.where(x < 1.0, nan, out)


@triton.jit
def _ts_atanh_impl(x):
    nan = tl.full(x.shape, float("nan"), x.dtype)
    inf = tl.full(x.shape, float("inf"), x.dtype)
    ax = _ts_abs(x)
    safe_x = tl.where(ax < 1.0, x, 0.0)
    out = 0.5 * (_ts_log1p_impl(safe_x) - _ts_log1p_impl(-safe_x))
    out = tl.where(x == 1.0, inf, tl.where(x == -1.0, -inf, out))
    return tl.where(ax > 1.0, nan, out)


@triton.jit
def _ts_erfc_impl(x):
    ax = _ts_abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    pos = poly * math.exp(-(ax * ax))
    return tl.where(x >= 0.0, pos, 2.0 - pos)


@triton.jit
def _ts_erfinv_impl(y):
    a0 = 0.886226899
    a1 = -1.645349621
    a2 = 0.914624893
    a3 = -0.140543331
    b0 = -2.118377725
    b1 = 1.442710462
    b2 = -0.329097515
    b3 = 0.012229801
    c0 = -1.970840454
    c1 = -1.624906493
    c2 = 3.429567803
    c3 = 1.641345311
    d0 = 3.543889200
    d1 = 1.637067800
    ay = _ts_abs(y)
    nan = tl.full(y.shape, float("nan"), y.dtype)
    inf = tl.full(y.shape, float("inf"), y.dtype)
    z = y * y
    num = (((a3 * z + a2) * z + a1) * z + a0)
    den = ((((b3 * z + b2) * z + b1) * z + b0) * z + 1.0)
    x0 = y * num / den
    safe_tail = tl.where(ay < 1.0, (1.0 - ay) / 2.0, 0.5)
    zr = math.sqrt(-math.log(safe_tail))
    numr = ((c3 * zr + c2) * zr + c1) * zr + c0
    denr = (d1 * zr + d0) * zr + 1.0
    x1 = _ts_copysign_impl(numr / denr, y)
    x = tl.where(ay <= 0.7, x0, x1)
    deriv = _TWO_OVER_SQRT_PI * math.exp(-(x * x))
    x = x - (math.erf(x) - y) / deriv
    deriv = _TWO_OVER_SQRT_PI * math.exp(-(x * x))
    x = x - (math.erf(x) - y) / deriv
    x = tl.where(ay == 1.0, _ts_copysign_impl(inf, y), x)
    return tl.where(ay > 1.0, nan, x)


@triton.jit
def _ts_log10_impl(x):
    return math.log(x) * _INV_LN10


@triton.jit
def _ts_nextafter_impl(x, y):
    nan = tl.full(x.shape, float("nan"), x.dtype)
    if x.dtype is tl.float16 or x.dtype is tl.bfloat16:
        sign_mask = tl.full(x.shape, 0x8000, tl.uint16)
        mag_mask = tl.full(x.shape, 0x7FFF, tl.uint16)
        one = tl.full(x.shape, 1, tl.uint16)
        bits = x.to(tl.uint16, bitcast=True)
        ordered = tl.where((bits & sign_mask) != 0, ~bits, bits | sign_mask)
        ordered = tl.where(x < y, ordered + one, ordered - one)
        new_bits = tl.where((ordered & sign_mask) != 0, ordered & mag_mask, ~ordered)
        tiny_bits = tl.where(_ts_signbit_impl(y), sign_mask | one, one)
        out = tl.where(x == 0, tiny_bits.to(x.dtype, bitcast=True), new_bits.to(x.dtype, bitcast=True))
    elif x.dtype is tl.float32:
        sign_mask = tl.full(x.shape, 0x80000000, tl.uint32)
        mag_mask = tl.full(x.shape, 0x7FFFFFFF, tl.uint32)
        one = tl.full(x.shape, 1, tl.uint32)
        bits = x.to(tl.uint32, bitcast=True)
        ordered = tl.where((bits & sign_mask) != 0, ~bits, bits | sign_mask)
        ordered = tl.where(x < y, ordered + one, ordered - one)
        new_bits = tl.where((ordered & sign_mask) != 0, ordered & mag_mask, ~ordered)
        tiny_bits = tl.where(_ts_signbit_impl(y), sign_mask | one, one)
        out = tl.where(x == 0, tiny_bits.to(tl.float32, bitcast=True), new_bits.to(tl.float32, bitcast=True))
    elif x.dtype is tl.float64:
        sign_mask = tl.full(x.shape, 0x8000000000000000, tl.uint64)
        mag_mask = tl.full(x.shape, 0x7FFFFFFFFFFFFFFF, tl.uint64)
        one = tl.full(x.shape, 1, tl.uint64)
        bits = x.to(tl.uint64, bitcast=True)
        ordered = tl.where((bits & sign_mask) != 0, ~bits, bits | sign_mask)
        ordered = tl.where(x < y, ordered + one, ordered - one)
        new_bits = tl.where((ordered & sign_mask) != 0, ordered & mag_mask, ~ordered)
        tiny_bits = tl.where(_ts_signbit_impl(y), sign_mask | one, one)
        out = tl.where(x == 0, tiny_bits.to(tl.float64, bitcast=True), new_bits.to(tl.float64, bitcast=True))
    else:
        out = x + tl.where(x < y, 1.0, -1.0)
    out = tl.where(x == y, y, out)
    return tl.where((x != x) | (y != y), nan, out)


@triton.jit
def _ts_round_impl(x):
    return _ts_nearbyint_impl(x)


@triton.jit
def _ts_floor_impl(x):
    return math.floor(x)


@triton.jit
def _ts_ceil_impl(x):
    return math.ceil(x)


@triton.jit
def _ts_isinf_impl(x):
    inf = tl.full(x.shape, float("inf"), x.dtype)
    return (_ts_abs(x) == inf) & (x == x)


@triton.jit
def _ts_isnan_impl(x):
    return x != x


@triton.jit
def _ts_fmod_impl(x, y):
    return x - _ts_trunc_impl(x / y) * y


@triton.jit
def _ts_lgamma_pos_impl(z):
    x = 0.99999999999980993
    x = x + 676.5203681218851 / z
    x = x - 1259.1392167224028 / (z + 1.0)
    x = x + 771.3234287776531 / (z + 2.0)
    x = x - 176.6150291621406 / (z + 3.0)
    x = x + 12.507343278686905 / (z + 4.0)
    x = x - 0.13857109526572012 / (z + 5.0)
    x = x + 9.984369578019572e-06 / (z + 6.0)
    x = x + 1.5056327351493116e-07 / (z + 7.0)
    t = z + 6.5
    return _LOG_SQRT_2PI + (z - 0.5) * math.log(t) - t + math.log(x)


@triton.jit
def _ts_lgamma_impl(z):
    inf = tl.full(z.shape, float("inf"), z.dtype)
    nan = tl.full(z.shape, float("nan"), z.dtype)
    ztrunc = _ts_trunc_impl(z)
    pole = (z <= 0.0) & (z == ztrunc)
    reflect = z < 0.5
    sin_term = math.sin(_PI * z)
    reflected = _LOG_PI - math.log(_ts_abs(sin_term)) - _ts_lgamma_pos_impl(1.0 - z)
    shifted = _ts_lgamma_pos_impl(z)
    out = tl.where(reflect, reflected, shifted)
    out = tl.where(z != z, nan, out)
    return tl.where(pole, inf, out)


@triton.jit
def _ts_asin_impl(x):
    nan = tl.full(x.shape, float("nan"), x.dtype)
    rad = tl.where(1.0 - x * x > 0.0, 1.0 - x * x, 0.0)
    out = _ts_atan2_impl(x, math.sqrt(rad))
    return tl.where(_ts_abs(x) > 1.0, nan, out)


@triton.jit
def _ts_acos_impl(x):
    nan = tl.full(x.shape, float("nan"), x.dtype)
    rad = tl.where(1.0 - x * x > 0.0, 1.0 - x * x, 0.0)
    out = _ts_atan2_impl(math.sqrt(rad), x)
    return tl.where(_ts_abs(x) > 1.0, nan, out)


@triton.jit
def _ts_ilogb_impl(x):
    _m, e = _ts_frexp_impl(x)
    int_min = tl.full(x.shape, -(2**31), tl.int32)
    int_max = tl.full(x.shape, 2**31 - 1, tl.int32)
    return tl.where(x == 0, int_min, tl.where(_ts_isinf_impl(x) | _ts_isnan_impl(x), int_max, e - 1))


@triton.jit
def _ts_ldexp_impl(x, e):
    return x * math.exp2(e.to(x.dtype))


@triton.jit
def _ts_j0_impl(x):
    PP = (
        7.96936729297347051624e-04, 8.28352392107440799803e-02,
        1.23953371646414299388e+00, 5.44725003058768775090e+00,
        8.74716500199817011941e+00, 5.30324038235394892183e+00,
        9.99999999999999997821e-01,
    )
    PQ = (
        9.24408810558863637013e-04, 8.56288474354474431428e-02,
        1.25352743901058953537e+00, 5.47097740330417105182e+00,
        8.76190883237069594232e+00, 5.30605288235394617618e+00,
        1.00000000000000000218e+00,
    )
    QP = (
        -1.13663838898469149931e-02, -1.28252718670509318512e+00,
        -1.95539544257735972385e+01, -9.32060152123768231369e+01,
        -1.77681167980488050595e+02, -1.47077505154951170175e+02,
        -5.14105326766599330220e+01, -6.05014350600728481186e+00,
    )
    QQ = (
        6.43178256118178023184e+01, 8.56430025976980587198e+02,
        3.88240183605401609683e+03, 7.24046774195652478189e+03,
        5.93072701187316984827e+03, 2.06209331660327847417e+03,
        2.42005740240291393179e+02,
    )
    RP = (
        -4.79443220978201773821e+09, 1.95617491946556577543e+12,
        -2.49248344360967716204e+14, 9.70862251047306323952e+15,
    )
    RQ = (
        4.99563147152651017219e+02, 1.73785401676374683123e+05,
        4.84409658339962045305e+07, 1.11855537045356834862e+10,
        2.11277520115489217587e+12, 3.10518229857422583814e+14,
        3.18121955943204943306e+16, 1.71086294081043136091e+18,
    )
    ax = _ts_abs(x)
    safe_ax = tl.where(ax > 0.0, ax, 1.0)
    xx = ax * ax
    safe_xx = tl.where(xx > 0.0, xx, 1.0)
    small = 1.0 - xx / 4.0
    rp = 0.0
    for c in RP:
        rp = rp * xx + c
    rq = 0.0
    for c in RQ:
        rq = rq * xx + c
    low = (xx - 5.78318596294678452118) * (xx - 30.4712623436620863991) * rp / rq
    inv_xx = 25.0 / safe_xx
    pp = 0.0
    for c in PP:
        pp = pp * inv_xx + c
    pq = 0.0
    for c in PQ:
        pq = pq * inv_xx + c
    qp = 0.0
    for c in QP:
        qp = qp * inv_xx + c
    qq = 0.0
    for c in QQ:
        qq = qq * inv_xx + c
    high = (pp / pq * math.cos(ax - 0.7853981633974483) - 5.0 / safe_ax * (qp / qq) * math.sin(ax - 0.7853981633974483)) * 0.7978845608028654 / math.sqrt(safe_ax)
    return tl.where(ax < 1.0e-5, small, tl.where(ax <= 5.0, low, high))


@triton.jit
def _ts_j1_impl(x):
    PP = (
        7.62125616208173112003e-04, 7.31397056940917570436e-02,
        1.12719608129684925192e+00, 5.11207951146807644818e+00,
        8.42404590141772420927e+00, 5.21451598682361504063e+00,
        1.00000000000000000254e+00,
    )
    PQ = (
        5.71323128072548699714e-04, 6.88455908754495404082e-02,
        1.10514232634061696926e+00, 5.07386386128601488557e+00,
        8.39985554327604159757e+00, 5.20982848682361821619e+00,
        9.99999999999999997461e-01,
    )
    QP = (
        5.10862594750176621635e-02, 4.98213872951233449420e+00,
        7.58238284132545283818e+01, 3.66779609360150777800e+02,
        7.10856304998926107277e+02, 5.97489612400613639965e+02,
        2.11688757100572135698e+02, 2.52070205858023719784e+01,
    )
    QQ = (
        7.42373277035675149943e+01, 1.05644886038262816351e+03,
        4.98641058337653607651e+03, 9.56231892404756170795e+03,
        7.99704160447350683650e+03, 2.82619278517639096600e+03,
        3.36093607810698293419e+02,
    )
    RP = (
        -8.99971225705559398224e+08, 4.52228297998194034323e+11,
        -7.27494245221818276015e+13, 3.68295732863852883286e+15,
    )
    RQ = (
        6.20836478118054335476e+02, 2.56987256757748830383e+05,
        8.35146791431949253037e+07, 2.21511595479792499675e+10,
        4.74914122079991414898e+12, 7.84369607876235854894e+14,
        8.95222336184627338078e+16, 5.32278620332680085395e+18,
    )
    ax = _ts_abs(x)
    safe_ax = tl.where(ax > 0.0, ax, 1.0)
    xx = ax * ax
    rp = 0.0
    for c in RP:
        rp = rp * xx + c
    rq = 0.0
    for c in RQ:
        rq = rq * xx + c
    low = rp / rq * ax * (xx - 14.6819706421238932572) * (xx - 49.2184563216946036703)
    inv_x = 5.0 / safe_ax
    inv_x2 = inv_x * inv_x
    pp = 0.0
    for c in PP:
        pp = pp * inv_x2 + c
    pq = 0.0
    for c in PQ:
        pq = pq * inv_x2 + c
    qp = 0.0
    for c in QP:
        qp = qp * inv_x2 + c
    qq = 0.0
    for c in QQ:
        qq = qq * inv_x2 + c
    high = (pp / pq * math.cos(ax - 2.356194490192345) - 5.0 / safe_ax * (qp / qq) * math.sin(ax - 2.356194490192345)) * 0.7978845608028654 / math.sqrt(safe_ax)
    out = tl.where(ax <= 5.0, low, high)
    return _ts_copysign_impl(out, x)


@triton.jit
def _ts_y0_impl(x):
    PP = (
        7.96936729297347051624e-04, 8.28352392107440799803e-02,
        1.23953371646414299388e+00, 5.44725003058768775090e+00,
        8.74716500199817011941e+00, 5.30324038235394892183e+00,
        9.99999999999999997821e-01,
    )
    PQ = (
        9.24408810558863637013e-04, 8.56288474354474431428e-02,
        1.25352743901058953537e+00, 5.47097740330417105182e+00,
        8.76190883237069594232e+00, 5.30605288235394617618e+00,
        1.00000000000000000218e+00,
    )
    QP = (
        -1.13663838898469149931e-02, -1.28252718670509318512e+00,
        -1.95539544257735972385e+01, -9.32060152123768231369e+01,
        -1.77681167980488050595e+02, -1.47077505154951170175e+02,
        -5.14105326766599330220e+01, -6.05014350600728481186e+00,
    )
    QQ = (
        6.43178256118178023184e+01, 8.56430025976980587198e+02,
        3.88240183605401609683e+03, 7.24046774195652478189e+03,
        5.93072701187316984827e+03, 2.06209331660327847417e+03,
        2.42005740240291393179e+02,
    )
    YP = (
        1.55924367855235737965e+04, -1.46639295903971606143e+07,
        5.43526477051876500413e+09, -9.82136065717911466409e+11,
        8.75906394395366999549e+13, -3.46628303384729719441e+15,
        4.42733268572569800351e+16, -1.84950800436986690637e+16,
    )
    YQ = (
        1.04128353664259848412e+03, 6.26107330137134956842e+05,
        2.68919633393814121987e+08, 8.64002487103935000337e+10,
        2.02979612750105546709e+13, 3.17157752842975028269e+15,
        2.50596256172653059228e+17,
    )
    nan = tl.full(x.shape, float("nan"), x.dtype)
    neg_inf = tl.full(x.shape, float("-inf"), x.dtype)
    safe_x = tl.where(x > 0.0, x, 1.0)
    xx = safe_x * safe_x
    yp = 0.0
    for c in YP:
        yp = yp * xx + c
    yq = 0.0
    for c in YQ:
        yq = yq * xx + c
    low = yp / yq + 0.6366197723675814 * math.log(safe_x) * _ts_j0_impl(safe_x)
    inv_xx = 25.0 / xx
    pp = 0.0
    for c in PP:
        pp = pp * inv_xx + c
    pq = 0.0
    for c in PQ:
        pq = pq * inv_xx + c
    qp = 0.0
    for c in QP:
        qp = qp * inv_xx + c
    qq = 0.0
    for c in QQ:
        qq = qq * inv_xx + c
    high = (pp / pq * math.sin(safe_x - 0.7853981633974483) + 5.0 / safe_x * (qp / qq) * math.cos(safe_x - 0.7853981633974483)) * 0.7978845608028654 / math.sqrt(safe_x)
    out = tl.where(x <= 5.0, low, high)
    out = tl.where(x == 0.0, neg_inf, out)
    return tl.where(x < 0.0, nan, out)


@triton.jit
def _ts_y1_impl(x):
    PP = (
        7.62125616208173112003e-04, 7.31397056940917570436e-02,
        1.12719608129684925192e+00, 5.11207951146807644818e+00,
        8.42404590141772420927e+00, 5.21451598682361504063e+00,
        1.00000000000000000254e+00,
    )
    PQ = (
        5.71323128072548699714e-04, 6.88455908754495404082e-02,
        1.10514232634061696926e+00, 5.07386386128601488557e+00,
        8.39985554327604159757e+00, 5.20982848682361821619e+00,
        9.99999999999999997461e-01,
    )
    QP = (
        5.10862594750176621635e-02, 4.98213872951233449420e+00,
        7.58238284132545283818e+01, 3.66779609360150777800e+02,
        7.10856304998926107277e+02, 5.97489612400613639965e+02,
        2.11688757100572135698e+02, 2.52070205858023719784e+01,
    )
    QQ = (
        7.42373277035675149943e+01, 1.05644886038262816351e+03,
        4.98641058337653607651e+03, 9.56231892404756170795e+03,
        7.99704160447350683650e+03, 2.82619278517639096600e+03,
        3.36093607810698293419e+02,
    )
    YP = (
        1.26320474790178026440e+09, -6.47355876379160291031e+11,
        1.14509511541823727583e+14, -8.12770255501325109621e+15,
        2.02439475713594898196e+17, -7.78877196265950026825e+17,
    )
    YQ = (
        5.94301592346128195359e+02, 2.35564092943068577943e+05,
        7.34811944459721705660e+07, 1.87601316108706159478e+10,
        3.88231277496238566008e+12, 6.20557727146953693363e+14,
        6.87141087355300489866e+16, 3.97270608116560655612e+18,
    )
    nan = tl.full(x.shape, float("nan"), x.dtype)
    neg_inf = tl.full(x.shape, float("-inf"), x.dtype)
    safe_x = tl.where(x > 0.0, x, 1.0)
    xx = safe_x * safe_x
    yp = 0.0
    for c in YP:
        yp = yp * xx + c
    yq = 0.0
    for c in YQ:
        yq = yq * xx + c
    low = safe_x * (yp / yq) + 0.6366197723675814 * (_ts_j1_impl(safe_x) * math.log(safe_x) - 1.0 / safe_x)
    inv_x = 5.0 / safe_x
    inv_x2 = inv_x * inv_x
    pp = 0.0
    for c in PP:
        pp = pp * inv_x2 + c
    pq = 0.0
    for c in PQ:
        pq = pq * inv_x2 + c
    qp = 0.0
    for c in QP:
        qp = qp * inv_x2 + c
    qq = 0.0
    for c in QQ:
        qq = qq * inv_x2 + c
    high = (pp / pq * math.sin(safe_x - 2.356194490192345) + 5.0 / safe_x * (qp / qq) * math.cos(safe_x - 2.356194490192345)) * 0.7978845608028654 / math.sqrt(safe_x)
    out = tl.where(x <= 5.0, low, high)
    out = tl.where(x == 0.0, neg_inf, out)
    return tl.where(x < 0.0, nan, out)


@triton.jit
def _ts_i0_impl(x):
    A = (
        -4.41534164647933937950e-18, 3.33079451882223809783e-17, -2.43127984654795469359e-16,
        1.71539128555513303061e-15, -1.16853328779934516808e-14, 7.67618549860493561688e-14,
        -4.85644678311192946090e-13, 2.95505266312963983461e-12, -1.72682629144155570723e-11,
        9.67580903537323691224e-11, -5.18979560163526290666e-10, 2.65982372468238665035e-09,
        -1.30002500998624804212e-08, 6.04699502254191894932e-08, -2.67079385394061173391e-07,
        1.11738753912010371815e-06, -4.41673835845875056359e-06, 1.64484480707288970893e-05,
        -5.75419501008210370398e-05, 1.88502885095841655729e-04, -5.76375574538582365885e-04,
        1.63947561694133579842e-03, -4.32430999505057594430e-03, 1.05464603945949983183e-02,
        -2.37374148058994688156e-02, 4.93052842396707084878e-02, -9.49010970480476444210e-02,
        1.71620901522208775349e-01, -3.04682672343198398683e-01, 6.76795274409476084995e-01,
    )
    B = (
        -7.23318048787475395456e-18, -4.83050448594418207126e-18, 4.46562142029675999901e-17,
        3.46122286769746109310e-17, -2.82762398051658348494e-16, -3.42548561967721913462e-16,
        1.77256013305652638360e-15, 3.81168066935262242075e-15, -9.55484669882830764870e-15,
        -4.15056934728722208663e-14, 1.54008621752140982691e-14, 3.85277838274214270114e-13,
        7.18012445138366623367e-13, -1.79417853150680611778e-12, -1.32158118404477131188e-11,
        -3.14991652796324136454e-11, 1.18891471078464383424e-11, 4.94060238822496958910e-10,
        3.39623202570838634515e-09, 2.26666899049817806459e-08, 2.04891858946906374183e-07,
        2.89137052083475648297e-06, 6.88975834691682398426e-05, 3.36911647825569408990e-03,
        8.04490411014108831608e-01,
    )
    ax = _ts_abs(x)
    safe_ax = tl.where(ax > 0.0, ax, 1.0)
    p = 0.0
    q = 0.0
    a = A[0]
    for c in A[1:]:
        p = q
        q = a
        a = ((safe_ax / 2.0) - 2.0) * q - p + c
    low = math.exp(ax) * (0.5 * (a - p))
    p2 = 0.0
    q2 = 0.0
    b = B[0]
    for c in B[1:]:
        p2 = q2
        q2 = b
        b = (32.0 / safe_ax - 2.0) * q2 - p2 + c
    high = math.exp(ax) * (0.5 * (b - p2)) / math.sqrt(safe_ax)
    return tl.where(ax <= 8.0, low, high)


@triton.jit
def _ts_i1_impl(x):
    A = (
        2.77791411276104639959e-18, -2.11142121435816608115e-17, 1.55363195773620046921e-16,
        -1.10559694773538630805e-15, 7.60068429473540693410e-15, -5.04218550472791168711e-14,
        3.22379336594557470981e-13, -1.98397439776494371520e-12, 1.17361862988909016308e-11,
        -6.66348972350202774223e-11, 3.62559028155211703701e-10, -1.88724975172282928790e-09,
        9.38153738649577178388e-09, -4.44505912879632808065e-08, 2.00329475355213526229e-07,
        -8.56872026469545474066e-07, 3.47025130813767847674e-06, -1.32731636560394358279e-05,
        4.78156510755005422638e-05, -1.61760815825896745588e-04, 5.12285956168575772895e-04,
        -1.51357245063125314899e-03, 4.15642294431288815669e-03, -1.05640848946261981558e-02,
        2.47264490306265168283e-02, -5.29459812080949914269e-02, 1.02643658689847095384e-01,
        -1.76416518357834055153e-01, 2.52587186443633654823e-01,
    )
    B = (
        7.51729631084210481353e-18, 4.41434832307170791151e-18, -4.65030536848935832153e-17,
        -3.20952592199342395980e-17, 2.96262899764595013876e-16, 3.30820231092092828324e-16,
        -1.88035477551078244854e-15, -3.81440307243700780478e-15, 1.04202769841288027642e-14,
        4.27244001671195135429e-14, -2.10154184277266431302e-14, -4.08355111109219731823e-13,
        -7.19855177624590851209e-13, 2.03562854414708950722e-12, 1.41258074366137813316e-11,
        3.25260358301548823856e-11, -1.89749581235054123450e-11, -5.58974346219658380687e-10,
        -3.83538038596423702205e-09, -2.63146884688951950684e-08, -2.51223623787020892529e-07,
        -3.88256480887769039346e-06, -1.10588938762623716291e-04, -9.76109749136146840777e-03,
        7.78576235018280120474e-01,
    )
    ax = _ts_abs(x)
    safe_ax = tl.where(ax > 0.0, ax, 1.0)
    p = 0.0
    q = 0.0
    a = A[0]
    for c in A[1:]:
        p = q
        q = a
        a = ((safe_ax / 2.0) - 2.0) * q - p + c
    low_mag = 0.5 * (a - p) * ax * math.exp(ax)
    low = _ts_copysign_impl(low_mag, x)
    p2 = 0.0
    q2 = 0.0
    b = B[0]
    for c in B[1:]:
        p2 = q2
        q2 = b
        b = (32.0 / safe_ax - 2.0) * q2 - p2 + c
    high_mag = math.exp(ax) * (0.5 * (b - p2)) / math.sqrt(safe_ax)
    high = _ts_copysign_impl(high_mag, x)
    return tl.where(ax <= 8.0, low, high)


@triton.jit
def _ts_erfcx_impl(x):
    inf = tl.full(x.shape, float("inf"), x.dtype)
    ax = _ts_abs(x)
    safe_ax = tl.where(ax > 0.0, ax, 1.0)
    small_pos = math.exp(ax * ax) * _ts_erfc_impl(ax)
    inv = 1.0 / safe_ax
    inv2 = inv * inv
    asym_pos = (1.0 + 0.5 * inv2 + 0.75 * inv2 * inv2 + 1.875 * inv2 * inv2 * inv2) / (_SQRT_PI * safe_ax)
    pos = tl.where(ax <= 5.0, small_pos, asym_pos)
    neg = 2.0 * math.exp(ax * ax) - pos
    neg = tl.where(x < -26.0, inf, neg)
    return tl.where(x < 0.0, neg, pos)

# Stable helper surface used by Triton codegen for libdevice-like math calls.
# Keep backend selection out of codegen and route through these wrappers instead.

@triton.jit
def acos(x):
    return _ts_acos_impl(x)

@triton.jit
def acosh(x):
    return _ts_acosh_impl(x)

@triton.jit
def asin(x):
    return _ts_asin_impl(x)

@triton.jit
def asinh(x):
    return _ts_asinh_impl(x)

@triton.jit
def atan(x):
    return _ts_atan_impl(x)

@triton.jit
def atan2(x, y):
    return _ts_atan2_impl(x, y)

@triton.jit
def atanh(x):
    return _ts_atanh_impl(x)

@triton.jit
def ceil(x):
    return _ts_ceil_impl(x)

@triton.jit
def copysign(x, y):
    return _ts_copysign_impl(x, y)

@triton.jit
def cos(x):
    return math.cos(x)

@triton.jit
def cosh(x):
    return _ts_cosh_impl(x)

@triton.jit
def cyl_bessel_i0(x):
    return _ts_i0_impl(x)

@triton.jit
def cyl_bessel_i1(x):
    return _ts_i1_impl(x)

@triton.jit
def erf(x):
    return math.erf(x)

@triton.jit
def erfc(x):
    return _ts_erfc_impl(x)

@triton.jit
def erfcx(x):
    return _ts_erfcx_impl(x)

@triton.jit
def erfinv(x):
    return _ts_erfinv_impl(x)

@triton.jit
def exp2(x):
    return math.exp2(x)

@triton.jit
def expm1(x):
    return _ts_expm1_impl(x)

@triton.jit
def floor(x):
    return _ts_floor_impl(x)

@triton.jit
def fma(x, y, z):
    return math.fma(x, y, z)

@triton.jit
def fmod(x, y):
    return _ts_fmod_impl(x, y)

@triton.jit
def hypot(x, y):
    return _ts_hypot_impl(x, y)

@triton.jit
def isinf(x):
    return _ts_isinf_impl(x)

@triton.jit
def isnan(x):
    return _ts_isnan_impl(x)

@triton.jit
def j0(x):
    return _ts_j0_impl(x)

@triton.jit
def j1(x):
    return _ts_j1_impl(x)

@triton.jit
def lgamma(x):
    return _ts_lgamma_impl(x)

@triton.jit
def llrint(x):
    return _ts_nearbyint_impl(x)

@triton.jit
def log10(x):
    return _ts_log10_impl(x)

@triton.jit
def log1p(x):
    return _ts_log1p_impl(x)

@triton.jit
def log2(x):
    return math.log2(x)

@triton.jit
def nearbyint(x):
    return _ts_nearbyint_impl(x)

@triton.jit
def nextafter(x, y):
    return _ts_nextafter_impl(x, y)

@triton.jit
def pow(x, y):
    return _ts_pow_impl(x, y)

@triton.jit
def rsqrt(x):
    return math.rsqrt(x)

@triton.jit
def signbit(x):
    return _ts_signbit_impl(x)

@triton.jit
def sin(x):
    return math.sin(x)

@triton.jit
def sinh(x):
    return _ts_sinh_impl(x)

@triton.jit
def sqrt(x):
    return math.sqrt(x)

@triton.jit
def tan(x):
    return _ts_tan_impl(x)

@triton.jit
def tanh(x):
    return _ts_tanh_impl(x)

@triton.jit
def trunc(x):
    return _ts_trunc_impl(x)

@triton.jit
def y0(x):
    return _ts_y0_impl(x)

@triton.jit
def y1(x):
    return _ts_y1_impl(x)
