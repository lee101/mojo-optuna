"""C ABI kernels for TPE mixture-density evaluation."""

from std.algorithm import parallelize
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import abs, exp, log, log1p
from std.memory import UnsafePointer
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime GPUFPtr = UnsafePointer[Float64, MutAnyOrigin]
comptime GPUIPtr = UnsafePointer[Int64, MutAnyOrigin]
comptime LOG_SQRT_2PI = 0.91893853320467274178
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_WORK_THRESHOLD = 131072
comptime PARALLEL_STREAM_THRESHOLD = 1048576


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
        for i in range(1, 64):
            term *= -Float64(2 * i - 1) * inv_x2
            if abs(term) >= previous:
                break
            total += term
            previous = abs(term)
            if previous < 1.0e-16:
                break
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
    mus: GPUFPtr,
    sigmas: GPUFPtr,
    steps: GPUFPtr,
    kinds: GPUIPtr,
    normalizers: GPUFPtr,
    accum: GPUFPtr,
    n: Int,
    kernels: Int,
    dims: Int,
    dimension_major: Int,
):
    var index = Int(block_idx.x * block_dim.x + thread_idx.x)
    if index >= n * kernels:
        return
    var row = index // kernels
    var kernel = index - row * kernels
    var score = 0.0
    for dim in range(dims):
        var parameter_index = kernel * dims + dim
        if dimension_major != 0:
            parameter_index = dim * kernels + kernel
        var kind = Int(kinds[dim])
        var value = x[row * dims + dim]
        var mu = mus[parameter_index]
        var sigma = sigmas[parameter_index]
        var step = steps[dim]
        if kind == 1:
            value = log(value)
        elif kind == 2:
            var za = (value - 0.5 * step - mu) / sigma
            var zb = (value + 0.5 * step - mu) / sigma
            score += log_gauss_mass(za, zb) + normalizers[parameter_index]
            continue
        elif kind == 3:
            var za = (log(value - 0.5 * step) - mu) / sigma
            var zb = (log(value + 0.5 * step) - mu) / sigma
            score += log_gauss_mass(za, zb) + normalizers[parameter_index]
            continue
        var z = (value - mu) / sigma
        score += -0.5 * z * z + normalizers[parameter_index]
    accum[index] = score


@export("mot_score_numeric_gpu")
def mot_score_numeric_gpu(
    x_addr: Int,
    mu_addr: Int,
    sigma_addr: Int,
    step_addr: Int,
    kind_addr: Int,
    normalizer_addr: Int,
    accum_addr: Int,
    n: Int,
    kernels: Int,
    dims: Int,
    dimension_major: Int,
) abi("C") -> Int:
    try:
        var x = fp(x_addr)
        var mus = fp(mu_addr)
        var sigmas = fp(sigma_addr)
        var steps = fp(step_addr)
        var kinds = ip(kind_addr)
        var normalizers = fp(normalizer_addr)
        var accum = fp(accum_addr)
        var ctx = DeviceContext()
        var x_device = ctx.enqueue_create_buffer[DType.float64](n * dims)
        var mu_device = ctx.enqueue_create_buffer[DType.float64](kernels * dims)
        var sigma_device = ctx.enqueue_create_buffer[DType.float64](
            kernels * dims
        )
        var step_device = ctx.enqueue_create_buffer[DType.float64](dims)
        var kind_device = ctx.enqueue_create_buffer[DType.int64](dims)
        var normalizer_device = ctx.enqueue_create_buffer[DType.float64](
            kernels * dims
        )
        var accum_device = ctx.enqueue_create_buffer[DType.float64](n * kernels)
        ctx.enqueue_copy(x_device, x)
        ctx.enqueue_copy(mu_device, mus)
        ctx.enqueue_copy(sigma_device, sigmas)
        ctx.enqueue_copy(step_device, steps)
        ctx.enqueue_copy(kind_device, kinds)
        ctx.enqueue_copy(normalizer_device, normalizers)
        ctx.enqueue_function[score_numeric_gpu_kernel](
            x_device,
            mu_device,
            sigma_device,
            step_device,
            kind_device,
            normalizer_device,
            accum_device,
            n,
            kernels,
            dims,
            dimension_major,
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
    mu_addr: Int,
    sigma_addr: Int,
    step_addr: Int,
    kind_addr: Int,
    normalizer_addr: Int,
    accum_addr: Int,
    n: Int,
    kernels: Int,
    dims: Int,
    dimension_major: Int,
) abi("C"):
    var x = fp(x_addr)
    var mus = fp(mu_addr)
    var sigmas = fp(sigma_addr)
    var steps = fp(step_addr)
    var kinds = ip(kind_addr)
    var normalizers = fp(normalizer_addr)
    var accum = fp(accum_addr)

    @__copy_capture(
        x,
        mus,
        sigmas,
        steps,
        kinds,
        normalizers,
        accum,
        kernels,
        dims,
        dimension_major,
    )
    @parameter
    def score_row(row: Int):
        var row_base = row * kernels
        var kernel = 0
        while kernel + W <= kernels:
            accum.store(row_base + kernel, SIMD[DType.float64, W](0.0))
            kernel += W
        while kernel < kernels:
            accum[row_base + kernel] = 0.0
            kernel += 1

        if dimension_major != 0:
            for dim in range(dims):
                var kind = Int(kinds[dim])
                var value = x[row * dims + dim]
                var step = steps[dim]
                if kind == 1:
                    value = log(value)
                if kind < 2:
                    kernel = 0
                    var parameter_base = dim * kernels
                    while kernel + W <= kernels:
                        var mu = mus.load[width=W](parameter_base + kernel)
                        var sigma = sigmas.load[width=W](
                            parameter_base + kernel
                        )
                        var z = (value - mu) / sigma
                        var score = accum.load[width=W](row_base + kernel)
                        score += -0.5 * z * z + normalizers.load[width=W](
                            parameter_base + kernel
                        )
                        accum.store(row_base + kernel, score)
                        kernel += W
                    while kernel < kernels:
                        var mu = mus[parameter_base + kernel]
                        var sigma = sigmas[parameter_base + kernel]
                        var z = (value - mu) / sigma
                        accum[row_base + kernel] += (
                            -0.5 * z * z + normalizers[parameter_base + kernel]
                        )
                        kernel += 1
                    continue
                for scalar_kernel in range(kernels):
                    var parameter_index = dim * kernels + scalar_kernel
                    var mu = mus[parameter_index]
                    var sigma = sigmas[parameter_index]
                    if kind == 2:
                        var za = (value - 0.5 * step - mu) / sigma
                        var zb = (value + 0.5 * step - mu) / sigma
                        accum[row_base + scalar_kernel] += (
                            log_gauss_mass(za, zb)
                            + normalizers[parameter_index]
                        )
                    else:
                        var za = (log(value - 0.5 * step) - mu) / sigma
                        var zb = (log(value + 0.5 * step) - mu) / sigma
                        accum[row_base + scalar_kernel] += (
                            log_gauss_mass(za, zb)
                            + normalizers[parameter_index]
                        )
            return

        for scalar_kernel in range(kernels):
            var score = 0.0
            for dim in range(dims):
                var kind = Int(kinds[dim])
                var value = x[row * dims + dim]
                var parameter_index = scalar_kernel * dims + dim
                var mu = mus[parameter_index]
                var sigma = sigmas[parameter_index]
                if kind == 1:
                    value = log(value)
                var z = (value - mu) / sigma
                score += -0.5 * z * z + normalizers[parameter_index]
            accum[row_base + scalar_kernel] = score

    var work = n * kernels * max(dims, 1)
    if dimension_major != 0 and work >= PARALLEL_WORK_THRESHOLD:
        parallelize[score_row](n, min(n, 8))
    else:
        for row in range(n):
            score_row(row)


@export("mot_compute_normalizers")
def mot_compute_normalizers(
    mu_addr: Int,
    sigma_addr: Int,
    low_addr: Int,
    high_addr: Int,
    step_addr: Int,
    kind_addr: Int,
    normalizer_addr: Int,
    kernels: Int,
    dims: Int,
    dimension_major: Int,
) abi("C"):
    var mus = fp(mu_addr)
    var sigmas = fp(sigma_addr)
    var lows = fp(low_addr)
    var highs = fp(high_addr)
    var steps = fp(step_addr)
    var kinds = ip(kind_addr)
    var normalizers = fp(normalizer_addr)
    for kernel in range(kernels):
        for dim in range(dims):
            var parameter_index = kernel * dims + dim
            if dimension_major != 0:
                parameter_index = dim * kernels + kernel
            var kind = Int(kinds[dim])
            var mu = mus[parameter_index]
            var sigma = sigmas[parameter_index]
            var lo = lows[dim]
            var hi = highs[dim]
            var step = steps[dim]
            if kind == 1:
                lo = log(lo)
                hi = log(hi)
            elif kind == 2:
                var a = (lo - 0.5 * step - mu) / sigma
                var b = (hi + 0.5 * step - mu) / sigma
                normalizers[parameter_index] = -log_gauss_mass(a, b)
                continue
            elif kind == 3:
                var a = (log(lo - 0.5 * step) - mu) / sigma
                var b = (log(hi + 0.5 * step) - mu) / sigma
                normalizers[parameter_index] = -log_gauss_mass(a, b)
                continue
            var a = (lo - mu) / sigma
            var b = (hi - mu) / sigma
            normalizers[parameter_index] = (
                -LOG_SQRT_2PI - log_gauss_mass(a, b) - log(sigma)
            )


@export("mot_score_categorical")
def mot_score_categorical(
    values_addr: Int,
    probabilities_addr: Int,
    accum_addr: Int,
    n: Int,
    kernels: Int,
    choices: Int,
) abi("C"):
    var values = fp(values_addr)
    var probabilities = fp(probabilities_addr)
    var accum = fp(accum_addr)

    @__copy_capture(values, probabilities, accum, kernels, choices)
    @parameter
    def score_row(row: Int):
        var choice = Int(values[row])
        var row_base = row * kernels
        var kernel = 0
        while kernel + W <= kernels:
            var probability = probabilities.load[width=W](
                kernel * choices + choice
            )
            if choices != 1:
                for lane in range(1, W):
                    probability[lane] = probabilities[
                        (kernel + lane) * choices + choice
                    ]
            var score = accum.load[width=W](row_base + kernel)
            accum.store(row_base + kernel, score + log(probability))
            kernel += W
        while kernel < kernels:
            accum[row_base + kernel] += log(
                probabilities[kernel * choices + choice]
            )
            kernel += 1

    if n * kernels >= PARALLEL_STREAM_THRESHOLD:
        parallelize[score_row](n, min(n, 8))
    else:
        for row in range(n):
            score_row(row)


@export("mot_finish_log_pdf")
def mot_finish_log_pdf(
    accum_addr: Int,
    weights_addr: Int,
    result_addr: Int,
    n: Int,
    kernels: Int,
) abi("C"):
    var accum = fp(accum_addr)
    var weights = fp(weights_addr)
    var result = fp(result_addr)

    @__copy_capture(accum, weights, result, kernels)
    @parameter
    def finish_row(row: Int):
        var row_base = row * kernels
        var maximum = -1.7976931348623157e308
        var kernel = 0
        while kernel + W <= kernels:
            var value = accum.load[width=W](row_base + kernel) + log(
                weights.load[width=W](kernel)
            )
            accum.store(row_base + kernel, value)
            maximum = max(maximum, value.reduce_max())
            kernel += W
        while kernel < kernels:
            var value = accum[row_base + kernel] + log(weights[kernel])
            accum[row_base + kernel] = value
            maximum = max(maximum, value)
            kernel += 1
        var total = 0.0
        kernel = 0
        while kernel + W <= kernels:
            total += exp(
                accum.load[width=W](row_base + kernel) - maximum
            ).reduce_add()
            kernel += W
        while kernel < kernels:
            total += exp(accum[row_base + kernel] - maximum)
            kernel += 1
        result[row] = log(total) + maximum

    if n * kernels >= PARALLEL_STREAM_THRESHOLD:
        parallelize[finish_row](n, min(n, 8))
    else:
        for row in range(n):
            finish_row(row)


@export("mot_best_acquisition")
def mot_best_acquisition(
    below_addr: Int, above_addr: Int, n: Int
) abi("C") -> Int:
    var below = fp(below_addr)
    var above = fp(above_addr)
    var best = 0
    var best_value = below[0] - above[0]
    for i in range(1, n):
        var value = below[i] - above[i]
        if value > best_value:
            best = i
            best_value = value
    return best
