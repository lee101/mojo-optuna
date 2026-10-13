"""C ABI kernels for TPE mixture-density evaluation.

Mixture parameters are always laid out dimension-major, that is
``[dimension, component]``, so the component axis is contiguous and every
scoring loop is a unit-stride vector loop.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import abs, exp, log, log1p, sqrt
from std.memory import Pointer
from std.sys.info import simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime GPUFPtr = Pointer[Float64, MutAnyOrigin]
comptime LOG_SQRT_2PI = 0.91893853320467274178
comptime W = simd_width_of[DType.float64]()


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def erf_cephes(x: Float64) -> Float64:
    var z = x * x
    var numerator = (
        (
            (9.604973739870516e0 * z + 9.002601972038427e1) * z
            + 2.232005345946843e3
        )
        * z
        + 7.003325141128051e3
    ) * z + 5.559230130103950e4
    var denominator = (
        (
            ((z + 3.356171416475031e1) * z + 5.213579497801527e2) * z
            + 4.594323829709801e3
        )
        * z
        + 2.262900006138910e4
    ) * z + 4.926739426086359e4
    return x * numerator / denominator


def erfc_cephes(x: Float64) -> Float64:
    if x < 1.0:
        return 1.0 - erf_cephes(x)
    var numerator: Float64
    var denominator: Float64
    if x < 8.0:
        numerator = (
            (
                (
                    (
                        (
                            (
                                (
                                    2.461969814735305e-10 * x
                                    + 5.641895648310688e-1
                                )
                                * x
                                + 7.463210564422699e0
                            )
                            * x
                            + 4.863719709856814e1
                        )
                        * x
                        + 1.965208329560771e2
                    )
                    * x
                    + 5.264451949954774e2
                )
                * x
                + 9.345285271719576e2
            )
            * x
            + 1.027551886895157e3
        ) * x + 5.575353353693993e2
        denominator = (
            (
                (
                    (
                        (
                            (
                                (x + 1.322819511547450e1) * x
                                + 8.670721408859897e1
                            )
                            * x
                            + 3.549377788878199e2
                        )
                        * x
                        + 9.757085017432055e2
                    )
                    * x
                    + 1.823909166879097e3
                )
                * x
                + 2.246337608187110e3
            )
            * x
            + 1.656663091941613e3
        ) * x + 5.575353408177277e2
    else:
        numerator = (
            (
                (
                    (5.641895835477551e-1 * x + 1.275366707599781e0) * x
                    + 5.019050422511805e0
                )
                * x
                + 6.160210979930536e0
            )
            * x
            + 7.409742699504489e0
        ) * x + 2.978866653721003e0
        denominator = (
            (
                (
                    ((x + 2.260528632201173e0) * x + 9.396035249380014e0) * x
                    + 1.204895398080967e1
                )
                * x
                + 1.708144507475659e1
            )
            * x
            + 9.608968090632859e0
        ) * x + 3.369076451000816e0
    return exp(-x * x) * numerator / denominator


def normal_cdf(x: Float64) -> Float64:
    var scaled = x * 0.70710678118654752440
    if scaled < -1.0:
        return 0.5 * erfc_cephes(-scaled)
    if scaled < 1.0:
        return 0.5 + 0.5 * erf_cephes(scaled)
    return 1.0 - 0.5 * erfc_cephes(scaled)


def log_normal_cdf(x: Float64) -> Float64:
    if x < -5.0:
        var inv_x2 = 1.0 / (x * x)
        var total = 1.0
        var term = 1.0
        var previous = 2.0
        var i = Int32(1)
        while i < 64:
            term *= -Float64(2 * i - 1) * inv_x2
            if abs(term) >= previous:
                break
            total += term
            previous = abs(term)
            if previous < 1.0e-16:
                break
            i += 1
        return -0.5 * x * x - log(-x) - LOG_SQRT_2PI + log(total)
    if x > 6.0:
        return -normal_cdf(-x)
    return log(normal_cdf(x))


def log_gauss_mass(a: Float64, b: Float64) -> Float64:
    if b <= 0.0:
        var lb = log_normal_cdf(b)
        var la = log_normal_cdf(a)
        return lb + log1p(-exp(la - lb))
    if a > 0.0:
        var lb = log_normal_cdf(-a)
        var la = log_normal_cdf(-b)
        return lb + log1p(-exp(la - lb))
    return log1p(-normal_cdf(a) - normal_cdf(-b))


def score_numeric_gpu_kernel(
    x: GPUFPtr,
    inv_sigmas: GPUFPtr,
    centers: GPUFPtr,
    normalizers: GPUFPtr,
    accum: GPUFPtr,
    n: Int32,
    kernels: Int32,
    dims: Int32,
):
    var index = Int32(block_idx.x * block_dim.x + thread_idx.x)
    if index >= n * kernels:
        return
    var row = index // kernels
    var kernel = index - row * kernels
    var score = 0.0
    var dim = Int32(0)
    while dim < dims:
        var parameter_index = dim * kernels + kernel
        var z = (
            x.unsafe_load(row * dims + dim) * inv_sigmas.unsafe_load(parameter_index)
            - centers.unsafe_load(parameter_index)
        )
        score += normalizers.unsafe_load(parameter_index) - 0.5 * z * z
        dim += 1
    accum.unsafe_store(index, score)


@export("mot_score_numeric_gpu")
def mot_score_numeric_gpu(
    x_addr: Int,
    inv_sigma_addr: Int,
    center_addr: Int,
    normalizer_addr: Int,
    accum_addr: Int,
    n: Int,
    kernels: Int,
    dims: Int,
) abi("C") -> Int:
    try:
        var x = fp(x_addr)
        var inv_sigmas = fp(inv_sigma_addr)
        var centers = fp(center_addr)
        var normalizers = fp(normalizer_addr)
        var accum = fp(accum_addr)
        var ctx = DeviceContext()
        var x_device = ctx.enqueue_create_buffer[DType.float64](n * dims)
        var inv_device = ctx.enqueue_create_buffer[DType.float64](kernels * dims)
        var center_device = ctx.enqueue_create_buffer[DType.float64](
            kernels * dims
        )
        var normalizer_device = ctx.enqueue_create_buffer[DType.float64](
            kernels * dims
        )
        var accum_device = ctx.enqueue_create_buffer[DType.float64](n * kernels)
        ctx.enqueue_copy(x_device, x)
        ctx.enqueue_copy(inv_device, inv_sigmas)
        ctx.enqueue_copy(center_device, centers)
        ctx.enqueue_copy(normalizer_device, normalizers)
        ctx.enqueue_function[score_numeric_gpu_kernel](
            x_device,
            inv_device,
            center_device,
            normalizer_device,
            accum_device,
            Int32(n),
            Int32(kernels),
            Int32(dims),
            grid_dim=(n * kernels + 255) // 256,
            block_dim=256,
        )
        ctx.enqueue_copy(accum, accum_device)
        ctx.synchronize()
        return 1
    except:
        return 0


@export("mot_score_numeric")
def mot_score_numeric(
    x_addr: Int,
    inv_sigma_addr: Int,
    center_addr: Int,
    normalizer_addr: Int,
    accum_addr: Int,
    n: Int,
    kernels: Int,
    dims: Int,
) abi("C"):
    var x = fp(x_addr)
    var inv_sigmas = fp(inv_sigma_addr)
    var centers = fp(center_addr)
    var normalizers = fp(normalizer_addr)
    var accum = fp(accum_addr)

    var block = 0
    while block + W <= kernels:
        for row in range(n):
            var x_base = row * dims
            var row_base = row * kernels
            var score = SIMD[DType.float64, W](0.0)
            for dim in range(dims):
                var base = dim * kernels + block
                var z = (
                    SIMD[DType.float64, W](x.unsafe_load(x_base + dim))
                    * inv_sigmas.unsafe_load[width=W](base)
                    - centers.unsafe_load[width=W](base)
                )
                score += normalizers.unsafe_load[width=W](base) - 0.5 * z * z
            accum.unsafe_store(row_base + block, score)
        block += W
    while block < kernels:
        for row in range(n):
            var x_base = row * dims
            var score = 0.0
            for dim in range(dims):
                var base = dim * kernels + block
                var z = (
                    x.unsafe_load(x_base + dim)
                    * inv_sigmas.unsafe_load(base)
                    - centers.unsafe_load(base)
                )
                score += normalizers.unsafe_load(base) - 0.5 * z * z
            accum.unsafe_store(row * kernels + block, score)
        block += 1


@export("mot_log_gauss_mass")
def mot_log_gauss_mass(
    za_addr: Int,
    zb_addr: Int,
    out_addr: Int,
    count: Int,
) abi("C"):
    var za = fp(za_addr)
    var zb = fp(zb_addr)
    var out = fp(out_addr)
    for i in range(count):
        out.unsafe_store(i, log_gauss_mass(za.unsafe_load(i), zb.unsafe_load(i)))


@export("mot_add_table")
def mot_add_table(
    accum_addr: Int,
    table_addr: Int,
    row_offset_addr: Int,
    col_offset_addr: Int,
    accumulate: Int,
    n: Int,
    kernels: Int,
) abi("C"):
    var accum = fp(accum_addr)
    var table = fp(table_addr)
    var row_offsets = ip(row_offset_addr)
    var col_offsets = ip(col_offset_addr)
    for row in range(n):
        var base = Int(row_offsets.unsafe_load(row))
        var row_base = row * kernels
        var k = 0
        while k + W <= kernels:
            var offsets = col_offsets.unsafe_load[width=W](k)
            var values = SIMD[DType.float64, W](0.0)
            for lane in range(W):
                values[lane] = table.unsafe_load(base + Int(offsets[lane]))
            if accumulate == 0:
                accum.unsafe_store(row_base + k, values)
            else:
                accum.unsafe_store(
                    row_base + k, accum.unsafe_load[width=W](row_base + k) + values
                )
            k += W
        while k < kernels:
            var value = table.unsafe_load(base + Int(col_offsets.unsafe_load(k)))
            if accumulate != 0:
                value += accum.unsafe_load(row_base + k)
            accum.unsafe_store(row_base + k, value)
            k += 1


@export("mot_compute_normalizers")
def mot_compute_normalizers(
    mu_addr: Int,
    sigma_addr: Int,
    low_addr: Int,
    high_addr: Int,
    normalizer_addr: Int,
    kernels: Int,
    dims: Int,
) abi("C"):
    var mus = fp(mu_addr)
    var sigmas = fp(sigma_addr)
    var lows = fp(low_addr)
    var highs = fp(high_addr)
    var normalizers = fp(normalizer_addr)
    for dim in range(dims):
        var lo = lows.unsafe_load(dim)
        var hi = highs.unsafe_load(dim)
        for kernel in range(kernels):
            var base = dim * kernels + kernel
            var mu = mus.unsafe_load(base)
            var sigma = sigmas.unsafe_load(base)
            var a = (lo - mu) / sigma
            var b = (hi - mu) / sigma
            normalizers.unsafe_store(
                base,
                -LOG_SQRT_2PI - log(sigma) - log_gauss_mass(a, b),
            )


@export("mot_score_categorical")
def mot_score_categorical(
    values_addr: Int,
    log_probabilities_addr: Int,
    accum_addr: Int,
    accumulate: Int,
    n: Int,
    kernels: Int,
) abi("C"):
    var values = fp(values_addr)
    var log_probabilities = fp(log_probabilities_addr)
    var accum = fp(accum_addr)

    for row in range(n):
        var base = Int(values.unsafe_load(row)) * kernels
        var row_base = row * kernels
        var k = 0
        while k + W <= kernels:
            var value = log_probabilities.unsafe_load[width=W](base + k)
            if accumulate != 0:
                value += accum.unsafe_load[width=W](row_base + k)
            accum.unsafe_store(row_base + k, value)
            k += W
        while k < kernels:
            var value = log_probabilities.unsafe_load(base + k)
            if accumulate != 0:
                value += accum.unsafe_load(row_base + k)
            accum.unsafe_store(row_base + k, value)
            k += 1


@export("mot_finish_log_pdf")
def mot_finish_log_pdf(
    accum_addr: Int,
    log_weights_addr: Int,
    result_addr: Int,
    n: Int,
    kernels: Int,
) abi("C"):
    var accum = fp(accum_addr)
    var log_weights = fp(log_weights_addr)
    var result = fp(result_addr)

    for row in range(n):
        var row_base = row * kernels
        var peak = SIMD[DType.float64, W](
            -1.7976931348623157e308
        )
        var k = 0
        while k + W <= kernels:
            peak = max(
                peak, accum.unsafe_load[width=W](row_base + k) + log_weights.unsafe_load[width=W](k)
            )
            k += W
        var maximum = peak.reduce_max()
        while k < kernels:
            maximum = max(
                maximum,
                accum.unsafe_load(row_base + k) + log_weights.unsafe_load(k),
            )
            k += 1
        var sums = SIMD[DType.float64, W](0.0)
        k = 0
        while k + W <= kernels:
            sums += exp(
                accum.unsafe_load[width=W](row_base + k) + log_weights.unsafe_load[width=W](k)
                - maximum
            )
            k += W
        while k < kernels:
            sums[0] += exp(
                accum.unsafe_load(row_base + k)
                + log_weights.unsafe_load(k)
                - maximum
            )
            k += 1
        result.unsafe_store(row, log(sums.reduce_add()) + maximum)


@export("mot_best_acquisition")
def mot_best_acquisition(
    below_addr: Int, above_addr: Int, n: Int
) abi("C") -> Int:
    var below = fp(below_addr)
    var above = fp(above_addr)
    var best = 0
    var best_value = below.unsafe_load(0) - above.unsafe_load(0)
    for i in range(1, n):
        var value = below.unsafe_load(i) - above.unsafe_load(i)
        if value > best_value:
            best = i
            best_value = value
    return best


@export("mot_matern52_cross")
def mot_matern52_cross(
    a_addr: Int, bt_addr: Int, m: Int, n: Int, d: Int, out_addr: Int
) abi("C"):
    """Matern-5/2: out[i, j] = k(|a_i - b_j|) for length-scale-normalised a [m, d] and b^T [d, n]."""
    var a = fp(a_addr)
    var bt = fp(bt_addr)
    var out = fp(out_addr)
    comptime S5 = 2.23606797749978969641
    for i in range(m):
        var row = i * n
        var j = 0
        while j + W <= n:
            var acc = SIMD[DType.float64, W](0.0)
            for k in range(d):
                var diff = bt.unsafe_load[width=W](k * n + j) - a.unsafe_load(i * d + k)
                acc += diff * diff
            var r = sqrt(acc)
            out.unsafe_store(row + j, (1.0 + S5 * r + (5.0 / 3.0) * acc) * exp(-S5 * r))
            j += W
        while j < n:
            var acc: Float64 = 0.0
            for k in range(d):
                var diff = bt.unsafe_load(k * n + j) - a.unsafe_load(i * d + k)
                acc += diff * diff
            var r = sqrt(acc)
            out.unsafe_store(row + j, (1.0 + S5 * r + (5.0 / 3.0) * acc) * exp(-S5 * r))
            j += 1


@export("mot_cscv")
def mot_cscv(
    b_addr: Int,
    s: Int,
    n: Int,
    masks_addr: Int,
    combos: Int,
    is_addr: Int,
    total_addr: Int,
    lam_addr: Int,
    best_is_addr: Int,
    best_oos_addr: Int,
) abi("C"):
    """CSCV over S row blocks of B [S, N]: per combination, logit rank of the IS-best trial OOS."""
    var b = fp(b_addr)
    var masks = ip(masks_addr)
    var isv = fp(is_addr)
    var total = fp(total_addr)
    var lam = fp(lam_addr)
    var best_is = fp(best_is_addr)
    var best_oos = fp(best_oos_addr)
    var half = Float64(s // 2)
    for j in range(n):
        var t: Float64 = 0.0
        for r in range(s):
            t += b.unsafe_load(r * n + j)
        total.unsafe_store(j, t)
    for c in range(combos):
        var mask = Int(masks.unsafe_load(c))
        var first = True
        for r in range(s):
            if (mask >> r) & 1 == 0:
                continue
            var j = 0
            if first:
                while j + W <= n:
                    isv.unsafe_store(j, b.unsafe_load[width=W](r * n + j))
                    j += W
                while j < n:
                    isv.unsafe_store(j, b.unsafe_load(r * n + j))
                    j += 1
                first = False
            else:
                while j + W <= n:
                    isv.unsafe_store(j, isv.unsafe_load[width=W](j) + b.unsafe_load[width=W](r * n + j))
                    j += W
                while j < n:
                    isv.unsafe_store(j, isv.unsafe_load(j) + b.unsafe_load(r * n + j))
                    j += 1
        var arg = 0
        var bv = isv.unsafe_load(0)
        for j in range(1, n):
            var v = isv.unsafe_load(j)
            if v > bv:
                bv = v
                arg = j
        var target = total.unsafe_load(arg) - isv.unsafe_load(arg)
        var below: Float64 = 0.0
        var equal: Float64 = 0.0
        for j in range(n):
            var o = total.unsafe_load(j) - isv.unsafe_load(j)
            if o < target:
                below += 1.0
            elif o == target:
                equal += 1.0
        var rank = below + 0.5 * (equal - 1.0) + 1.0
        var w = rank / Float64(n + 1)
        lam.unsafe_store(c, log(w / (1.0 - w)))
        best_is.unsafe_store(c, bv / half)
        best_oos.unsafe_store(c, target / half)
