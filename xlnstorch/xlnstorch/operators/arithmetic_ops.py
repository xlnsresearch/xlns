import torch
from xlnstorch import CSRC_AVAILABLE, LNS_ZERO, LNS_ONE, LNS_NEG_ONE, LNSTensor, lnstensor, format_lnstensor_operands, implements, full_like
from xlnstorch.autograd import LNSFunction
from . import (
    lns_add,
    lns_mul,
    lns_div,
    lns_square,
    lns_reciprocal,
    lns_pow,
    lns_matmul,
    lns_sum,
    lns_sub,
    lns_sum_to_size,
)

class LNSMulFunction(LNSFunction):
    """
    Multiplication becomes addition in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x * y) = y
    d/dy(x * y) = x
    """

    @staticmethod
    def forward(x, y, base):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
        result = (x_packed + y_packed - (y_packed & 1)) ^ (y_packed & 1)

        # overflow check for reference
        # torch.lt(x_packed, 0) & torch.lt(y_packed, -9223372036854775808 - x_packed)
        return torch.where(torch.eq(x_packed | 1, LNS_ZERO) | torch.eq(y_packed | 1, LNS_ZERO),
                             LNS_ZERO, result.to(torch.float64))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors

        grad_x = lns_mul(grad_output, y, base)
        grad_y = lns_mul(grad_output, x, base)

        grad_x = lns_sum_to_size(grad_x, base, x.shape)
        grad_y = lns_sum_to_size(grad_y, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.mul, LNSMulFunction.forward, key='default', default=True)
def mul(x, y, *, out=None):

    x, y = format_lnstensor_operands(x, y)
    result = LNSMulFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSquareFunction(LNSFunction):
    """
    Squaring becomes doubling in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x ^ 2) = 2 * x
    """

    @staticmethod
    def forward(x, base):
        return lns_mul(x, x, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_mul(x, LNSTensor.get_internal_tensor(2.0, base), base)
        grad_x = lns_mul(grad_output, grad_x, base)

        return grad_x, None

@implements(torch.square, LNSSquareFunction.forward, key='default', default=True)
def square(x, *, out=None):

    result = LNSSquareFunction.apply(x, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSSqrtFunction(LNSFunction):
    """
    Square rooting becomes halving in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
    """

    @staticmethod
    def forward(x, base):
        x_packed = x.to(torch.int64)
        result = ((x_packed & (-2)) / 2).to(torch.int64) & (-2)

        return torch.where(torch.eq(x_packed | 1, LNS_ZERO), LNS_ZERO, result.to(torch.float64))

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        sqrt_x = output
        ctx.save_for_backward(sqrt_x, base)

    @staticmethod
    def backward(ctx, grad_output):
        sqrt_x, base = ctx.saved_tensors

        grad_x = lns_mul(sqrt_x, LNSTensor.get_internal_tensor(2.0, base), base)
        grad_x = lns_div(grad_output, grad_x, base)

        return grad_x, None

@implements(torch.sqrt, LNSSqrtFunction.forward, key='default', default=True)
def sqrt(x, *, out=None):

    result = LNSSqrtFunction.apply(x, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSPowFunction(LNSFunction):
    """
    Raising to a power becomes multiplication in the logarithmic domain.
    This function relies on the fact that the exponent is a floating
    point or integer which allows us to compute the power directly.

    Gradients are computed as follows:
    d/dx(x ^ n) = n * x ^ (n - 1)
    """

    @staticmethod
    def forward(x, n, base):
        x_packed = x.to(torch.int64)

        if torch.is_floating_point(n):
            result = ((x_packed & (-2)) * n).to(torch.int64) & (-2)

        else:
            abs_result = ((x_packed & (-2)) * n) & (-2)
            result = torch.where(n & 1 == 0, abs_result, abs_result | (x_packed & 1))

        return result.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, n, base = inputs
        ctx.save_for_backward(x, n, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, n, base = ctx.saved_tensors

        grad_x = lns_pow(x, n - 1, base)
        grad_x = lns_mul(grad_x, LNSTensor.get_internal_tensor(n, base), base)
        grad_x = lns_mul(grad_output, grad_x, base)

        return grad_x, None, None

@implements(torch.pow, LNSPowFunction.forward, key='default', default=True)
def pow(x, n, *, out=None):

    if isinstance(x, LNSTensor) and not isinstance(n, LNSTensor):

        if not isinstance(n, torch.Tensor):
            dtype = torch.int64 if (isinstance(n, int) or isinstance(n, float) and n.is_integer()) else torch.float64
            n = torch.tensor(n, dtype=dtype)

        x._lns, n = torch.broadcast_tensors(x._lns, n)

        result = LNSPowFunction.apply(x, n, x.base)

    else:

        x, n = format_lnstensor_operands(x, n)
        x._lns, n._lns = torch.broadcast_tensors(x._lns, n._lns)

        result = LNSPowFunction.apply(x, n.value, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSDivFunction(LNSFunction):
    """
    Division becomes subtraction in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x / y) = 1 / y
    d/dy(x / y) = -x / (y^2)
    """

    @staticmethod
    def forward(x, y, base):
        x_packed, y_packed = x.to(torch.int64), y.to(torch.int64)
        result = (x_packed - y_packed + (y_packed & 1)) ^ (y_packed & 1)

        # overflow check for reference
        # torch.gt(y_packed, 0) & torch.lt(x_packed, -9223372036854775808 + y_packed)
        return torch.where(torch.eq(x_packed | 1, LNS_ZERO), LNS_ZERO, result.to(torch.float64))

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, base = inputs
        ctx.save_for_backward(x, y, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, y, base = ctx.saved_tensors

        grad_x = lns_div(grad_output, y, base)
        grad_y = lns_square(y, base)
        grad_y = lns_div(x, grad_y, base)
        grad_y = lns_mul(grad_y, LNS_NEG_ONE, base)
        grad_y = lns_mul(grad_output, grad_y, base)

        grad_x = lns_sum_to_size(grad_x, base, x.shape)
        grad_y = lns_sum_to_size(grad_y, base, y.shape)

        return grad_x, grad_y, None

@implements(torch.div, LNSDivFunction.forward, key='default', default=True)
def div(x, y, *, out=None):

    x, y = format_lnstensor_operands(x, y)

    result = LNSDivFunction.apply(x, y, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSReciprocalFunction(LNSFunction):
    """
    See LNSDivFunction for details on the internal computation.

    Gradients are calculated as follows:
    d/dx(1 / x) = -1 / (x ^ 2)
    """

    @staticmethod
    def forward(x, base):
        return lns_div(LNS_ONE, x, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_square(x, base)
        grad_x = lns_reciprocal(grad_x, base)
        grad_x = lns_mul(grad_x, LNS_NEG_ONE, base)
        grad_x = lns_mul(grad_output, grad_x, base)

        return grad_x, None

@implements(torch.reciprocal, LNSReciprocalFunction.forward, key='default', default=True)
def reciprocal(x, *, out=None):

    result = LNSReciprocalFunction.apply(x, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSExpFunction(LNSFunction):
    """
    Exponentiation in the logarithmic domain requires us to
    convert the input to its floating point representation
    and then compute raising to the power of e.

    Gradients are computed as follows:
    d/dx(e ^ x) = e ^ x
    """

    @staticmethod
    def forward(x, base):
        e = full_like(x, torch.exp(torch.tensor(1.0)), b=base)
        return lns_pow(e._lns, lnstensor(x, from_lns=True, b=base).value, base)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, base = inputs
        ctx.save_for_backward(output, base)

    @staticmethod
    def backward(ctx, grad_output):
        exp_x, base = ctx.saved_tensors
        return lns_mul(grad_output, exp_x, base), None

@implements(torch.exp, LNSExpFunction.forward, key='default', default=True)
def exp(x, *, out=None):

    result = LNSExpFunction.apply(x, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSLogFunction(LNSFunction):
    """
    Taking the logarithm in the logarithmic domain requires us to
    convert the input to its floating point representation and then
    compute the logarithm with respect to the base.

    Gradients are computed as follows:
    d/dx(log(x)) = 1 / x
    """

    @staticmethod
    def forward(x, base):
        x_log = torch.log(lnstensor(x, from_lns=True, b=base).value)
        return lnstensor(x_log, b=base)._lns

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base = inputs
        ctx.save_for_backward(x, base)

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        grad_x = lns_div(grad_output, x, base)
        return grad_x, None

@implements(torch.log, LNSLogFunction.forward, key='default', default=True)
def log(x, *, out=None):

    result = LNSLogFunction.apply(x, x.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSProdFunction(LNSFunction):
    """
    Product is computed using the multiplication operation.

    Gradients are computed as follows:
    d/dx(prod(x)) = prod(x) / x
    """

    @staticmethod
    def forward(x, base, dim=None, keepdim=False):
        x_packed = x.to(torch.int64)

        if dim is None:
            flat = x_packed.reshape(-1)

            out = flat[0]
            for i in range(1, flat.numel()):
                out = lns_mul(out, flat[i], base)

            if keepdim:
                out = out.reshape([1] * x.dim())

            return out

        # Reduction over a subset of the dimensions
        red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
        red_dims = tuple(sorted(d % x.dim() for d in red_dims))

        # transpose so that the reduction dimensions are at the end, then flatten.
        permute_order = [d for d in range(x.dim()) if d not in red_dims] + list(red_dims)
        transposed = x_packed.permute(*permute_order)
        outer_shape = transposed.shape[:-len(red_dims)]
        transposed = transposed.reshape(*outer_shape, -1)

        out = transposed[..., 0]
        for i in range(1, transposed.shape[-1]):
            out = lns_mul(out, transposed[..., i], base)

        # re-insert the reduced axes
        if keepdim:
            for d in red_dims:
                out = out.unsqueeze(d)

        return out.to(torch.float64)
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, dim, keepdim = inputs
        ctx.save_for_backward(x, output, base)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output):
        x, output, base = ctx.saved_tensors
        x_packed, output_packed = x.to(torch.int64), output.to(torch.int64)

        # 1. Broadcast the forward result so it matches x's shape
        if ctx.dim is not None and not ctx.keepdim:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(sorted(d % x.dim() for d in red_dims))

            for d in red_dims:
                output_packed = output_packed.unsqueeze(d)

        output_broadcast = output_packed.expand_as(x_packed)
        ratio = lns_div(output_broadcast, x_packed, base)

        # broadcast grad_output to match x's shape
        if ctx.dim is not None and not ctx.keepdim:
            for d in red_dims:
                grad_output = grad_output.unsqueeze(d)

        grad_output = grad_output.expand_as(x)
        grad_x = lns_mul(grad_output, ratio, base)

        return grad_x, None, None, None

@implements(torch.prod, LNSProdFunction.forward, "default", default=True)
def prod(x, dim=None, keepdim=False, *, out=None):

    result = LNSProdFunction.apply(x, x.base, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMeanFunction(LNSFunction):

    @staticmethod
    def forward(x, base, dim=None, keepdim=False):
        x_packed = x.to(torch.int64)

        if dim is None:
            dims = None
        else:
            if isinstance(dim, int):
                dims = (dim,)
            else:
                dims = tuple(dim)
            # canonicalise negative indices
            dims = tuple(d % x.dim() for d in dims)

        if dims is None:
            n_elem = x.numel()
        else:
            n_elem = 1
            for d in dims:
                n_elem *= x.shape[d]

        total = lns_sum(x_packed, base, dims, keepdim)
        return lns_div(total, LNSTensor.get_internal_tensor(n_elem, base), base).to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, dim, keepdim = inputs
        ctx.save_for_backward(x, base)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        if ctx.dim is None:
            dims = None
        else:
            if isinstance(ctx.dim, int):
                dims = (ctx.dim,)
            else:
                dims = tuple(ctx.dim)
            # canonicalise negative indices
            dims = tuple(d % x.dim() for d in dims)

        if dims is None:
            n_elem = x.numel()
        else:
            n_elem = 1
            for d in dims:
                n_elem *= x.shape[d]

        grad_x = lns_div(grad_output, LNSTensor.get_internal_tensor(n_elem, base), base)
        if dims is None:
            grad_x = grad_x.expand(x.shape)

        else:
            if not ctx.keepdim:
                for d in sorted(dims):
                    grad_x = grad_x.unsqueeze(d)
            grad_x = grad_x.expand(x.shape)

        return grad_x, None, None, None

@implements(torch.mean, LNSMeanFunction.forward, "default", default=True)
def mean(x, dim=None, keepdim=False, *, out=None):

    result = LNSMeanFunction.apply(x, x.base, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSVarFunction(LNSFunction):

    @staticmethod
    def forward(x, base, correction, dim=None, keepdim=False):

        if dim is None:
            red_dims = None
            N = x.numel()

        else:
            red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
            red_dims = tuple(d % x.dim() for d in red_dims)
            N = 1
            for d in red_dims:
                N *= x.shape[d]

        n_elems = LNSTensor.get_internal_tensor(N, base)

        denom = lns_sub(n_elems, correction, base)
        if denom <= 0:
            raise ValueError("Degrees of freedom <= 0 for slice")

        total_x = lns_sum(x, base, dim=red_dims, keepdim=True)
        mean = lns_div(total_x, n_elems, base)

        diff = lns_sub(x, mean, base)
        sq_diff = lns_mul(diff, diff, base)
        total_sq = lns_sum(sq_diff, base, dim=red_dims, keepdim=keepdim)
        var = lns_div(total_sq, denom, base)

        return var.to(torch.float64)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, base, correction, dim, keepdim = inputs
        ctx.save_for_backward(x, base)
        ctx.dim = dim
        ctx.correction = correction
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, grad_output):
        x, base = ctx.saved_tensors

        if ctx.dim is None:
            red_dims = None
            N = x.numel()

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            N = 1
            for d in red_dims:
                N *= x.shape[d]

        total_x = lns_sum(x, base, dim=red_dims, keepdim=True)
        n_elems = LNSTensor.get_internal_tensor(N, base)
        denom = lns_sub(n_elems, ctx.correction, base)
        mean = lns_div(total_x, n_elems, base)

        diff = lns_sub(x, mean, base)
        scale = lns_div(LNSTensor.get_internal_tensor(2.0, base), denom, base)

        grad_x = grad_output
        if red_dims is None:
            grad_x = grad_x.expand(x.shape)

        else:
            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)
            grad_x = grad_x.expand(x.shape)

        grad_x = lns_mul(grad_x, lns_mul(diff, scale, base), base)

        return grad_x, None, None, None, None

@implements(torch.var, LNSVarFunction.forward, "default", default=True)
def var(x, dim=None, *, correction=1, keepdim=False, out=None):

    x, correction = format_lnstensor_operands(x, correction)
    result = LNSVarFunction.apply(x, x.base, dim, correction, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=x.base)

class LNSMatmulFunction(LNSFunction):
    """
    Matrix multiplication uses the lns addition and
    multiplication functions to compute the result.

    Gradients are computed as follows:
    d/dA(A @ B) = B^T
    d/dB(A @ B) = A^T
    """

    @staticmethod
    def forward(A, B, base):
        # 1. (..., M, K)  @  (..., K, N)  -> (..., M, N)          (regular case)
        # 2. (..., M, K)  @  (..., K)     -> (..., M)             (rhs vector)
        # 3. (..., K)     @  (..., K, N)  -> (..., N)             (lhs vector)
        # 4. (..., K)     @  (..., K)     -> (..., K)             (dot product)
        orig_A_dim = A.dim()
        orig_B_dim = B.dim()

        prepended_A = False
        appended_B = False

        if orig_A_dim == 1:
            A = A.unsqueeze(0) # (K,) -> (1, K)
            prepended_A = True

        if orig_B_dim == 1:
            B = B.unsqueeze(-1) # (K,) -> (K, 1)
            appended_B = True

        # Now perform the actual matrix multiplication
        # A has shape (..., M, K) and B has shape (..., K, N)
        # For broadcasting, align batch dimensions
        M, K_A = A.shape[-2:]
        K_B, N = B.shape[-2:]

        assert K_A == K_B, "Inner dimensions of A and B must match for matrix multiplication: {K_A} vs {K_B}"

        # Handle broadcasting of batch dimensions - get batch shapes (everything except last 2 dims)
        A_batch_shape = A.shape[:-2]
        B_batch_shape = B.shape[:-2]

        try:
            output_batch_shape = torch.broadcast_shapes(A_batch_shape, B_batch_shape)
        except RuntimeError as e:
            raise RuntimeError(f"Batch dimensions are not broadcastable: {A_batch_shape} vs {B_batch_shape}") from e

        # Expand A and B to have the same batch dimensions
        A = A.expand(*output_batch_shape, M, K_A)
        B = B.expand(*output_batch_shape, K_B, N)

        result = torch.full((*output_batch_shape, M, N), fill_value=LNS_ZERO,
                            dtype=torch.float64, device=A.device)

        # Perform matrix multiplication in log space
        for k in range(K_A):
            term = lns_mul(
                A[..., :, k].unsqueeze(-1), # (..., M, 1)
                B[..., k, :].unsqueeze(-2), # (..., 1, N)
                base)
            result = lns_add(result, term, base)

        if prepended_A:
            result = result.squeeze(-2) # Remove extra M dimension
        if appended_B:
            result = result.squeeze(-1) # Remove extra N dimension

        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        # Here we must repeat the unsqueezing logic from forward
        # to ensure that we can correctly compute the gradients.
        # todo: This is a bit of a hack, we should ideally handle
        # the unsqueezing in a more elegant way. 
        A, B, base = inputs

        ctx.prepended_A = False
        ctx.appended_B = False

        if A.dim() == 1:
            A = A.unsqueeze(0)
            ctx.prepended_A = True

        if B.dim() == 1:
            B = B.unsqueeze(-1)
            ctx.appended_B = True

        ctx.A_shape_before_broadcast = A.shape
        ctx.B_shape_before_broadcast = B.shape

        A_batch_shape = A.shape[:-2]
        B_batch_shape = B.shape[:-2]
        output_batch_shape = torch.broadcast_shapes(A_batch_shape, B_batch_shape)

        A = A.expand(*output_batch_shape, *A.shape[-2:])
        B = B.expand(*output_batch_shape, *B.shape[-2:])

        ctx.save_for_backward(A, B, base)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, base = ctx.saved_tensors

        #  Re-introduce squeezed dimensions
        if ctx.prepended_A and not ctx.appended_B:
            grad_output = grad_output.unsqueeze(-2)
        elif ctx.appended_B and not ctx.prepended_A:
            grad_output = grad_output.unsqueeze(-1)
        elif ctx.prepended_A and ctx.appended_B:
            grad_output = grad_output.unsqueeze(-1).unsqueeze(-1)

        # Compute gradients w.r.t A and B after broadcasting
        grad_A = lns_matmul(grad_output, B.transpose(-1, -2), base)
        grad_B = lns_matmul(A.transpose(-1, -2), grad_output, base)

        # Reduce gradients to match original shapes before broadcasting
        # We need to sum over dimensions that were broadcasted

        # For grad_A: reduce to shape before broadcasting
        A_shape_before_broadcast = ctx.A_shape_before_broadcast
        while grad_A.dim() > len(A_shape_before_broadcast):
            grad_A = lns_sum(grad_A, base, dim=0)

        # Sum over any dimensions that were size 1 and got broadcasted
        for i in range(len(A_shape_before_broadcast) - 2):  # Don't touch matrix dims
            if A_shape_before_broadcast[i] == 1 and grad_A.shape[i] > 1:
                grad_A = lns_sum(grad_A, base, dim=i, keepdim=True)

        # For grad_B: reduce to shape before broadcasting
        B_shape_before_broadcast = ctx.B_shape_before_broadcast
        while grad_B.dim() > len(B_shape_before_broadcast):
            grad_B = lns_sum(grad_B, base, dim=0)

        # Sum over any dimensions that were size 1 and got broadcasted
        for i in range(len(B_shape_before_broadcast) - 2):  # Don't touch matrix dims
            if B_shape_before_broadcast[i] == 1 and grad_B.shape[i] > 1:
                grad_B = lns_sum(grad_B, base, dim=i, keepdim=True)

        if ctx.prepended_A:
            grad_A = grad_A.squeeze(0) # Remove extra M dimension
        if ctx.appended_B:
            grad_B = grad_B.squeeze(-1) # Remove extra N dimension

        return grad_A, grad_B, None

@implements(torch.matmul, LNSMatmulFunction.forward, "default", default=not CSRC_AVAILABLE)
def matmul(A, B, *, out=None):

    A, B = format_lnstensor_operands(A, B)
    result = LNSMatmulFunction.apply(A, B, A.base)

    if out is not None:
        return out._inplace_copy(result)

    return lnstensor(result, from_lns=True, b=A.base)

class LNSTransposeFunction(LNSFunction):
    """
    Transpose operation simply rearranges the dimensions
    of the input tensor. It doesn't change the underlying
    representations, so the forward pass isn't special.

    Gradients are computed as follows:
    d/dx(A.T) = 1
    """

    @staticmethod
    def forward(A, dim0, dim1):
        return torch.transpose(A, dim0, dim1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, dim0, dim1 = inputs
        ctx.dim0 = dim0
        ctx.dim1 = dim1

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = torch.transpose(grad_output, ctx.dim0, ctx.dim1)
        return grad_x, None, None

@implements(torch.transpose, LNSTransposeFunction.forward, "default", default=True)
def transpose(A, dim0, dim1):

    result = LNSTransposeFunction.apply(A, dim0, dim1)
    return lnstensor(result, from_lns=True, b=A.base)
