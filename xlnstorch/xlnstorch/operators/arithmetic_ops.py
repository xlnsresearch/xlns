import math
import torch
from xlnstorch import CSRC_AVAILABLE, LNS_ZERO, LNS_ONE, LNS_NEG_ONE, LNSTensor, lnstensor, format_lnstensor_operands, implements
from xlnstorch.autograd import LNSFunction

def _mul(ops, x, y):
    result = (x + y - (y & 1)) ^ (y & 1)

    # overflow check for reference
    # torch.lt(x_packed, 0) & torch.lt(y_packed, -9223372036854775808 - x_packed)
    return torch.where(torch.eq(x | 1, LNS_ZERO) | torch.eq(y | 1, LNS_ZERO),
                       LNS_ZERO, result)

class LNSMulFunction(LNSFunction):
    """
    Multiplication becomes addition in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x * y) = y
    d/dy(x * y) = x
    """

    @staticmethod
    def forward(ops, x, y):
        return _mul(ops, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.mul(grad_output, y)
        grad_y = ops.mul(grad_output, x)

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x, grad_y

@implements(torch.mul, _mul, key='default', default=True)
def mul(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSMulFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _square(ops, x):
    return ops.mul(x, x)

class LNSSquareFunction(LNSFunction):
    """
    Squaring becomes doubling in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x ^ 2) = 2 * x
    """

    @staticmethod
    def forward(ops, x):
        return _square(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = ops.mul(x, ops.to_lns(2.0))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.square, _square, key='default', default=True)
def square(x, *, out=None):
    result = LNSSquareFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _sqrt(ops, x):
    result = ((x & (-2)) // 2) & (-2)
    return torch.where(torch.eq(x | 1, LNS_ZERO), LNS_ZERO, result)

class LNSSqrtFunction(LNSFunction):
    """
    Square rooting becomes halving in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(sqrt(x)) = 1 / (2 * sqrt(x))
    """

    @staticmethod
    def forward(ops, x):
        return _sqrt(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        sqrt_x = output
        ctx.save_for_backward(sqrt_x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        sqrt_x, = ctx.saved_tensors

        grad_x = ops.mul(sqrt_x, ops.to_lns(2.0))
        grad_x = ops.div(grad_output, grad_x)

        return grad_x

@implements(torch.sqrt, _sqrt, key='default', default=True)
def sqrt(x, *, out=None):
    result = LNSSqrtFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _pow(ops, x, n):
    if torch.is_floating_point(n):
        return ((x & (-2)) * n).to(torch.int64) & (-2)
    else:
        abs_result = ((x & (-2)) * n) & (-2)
        return torch.where(n & 1 == 0, abs_result, abs_result | (x & 1))

class LNSPowFunction(LNSFunction):
    """
    Raising to a power becomes multiplication in the logarithmic domain.
    This function relies on the fact that the exponent is a floating
    point or integer which allows us to compute the power directly.

    Gradients are computed as follows:
    d/dx(x ^ n) = n * x ^ (n - 1)
    """

    @staticmethod
    def forward(ops, x, n):
        return _pow(ops, x, n)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, n = inputs
        ctx.save_for_backward(x, n)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, n, = ctx.saved_tensors

        grad_x = ops.pow(x, n - 1)
        grad_x = ops.mul(grad_x, ops.to_lns(n))
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x, None

@implements(torch.pow, _pow, key='default', default=True)
def pow(x, n, *, out=None):
    if isinstance(x, LNSTensor) and not isinstance(n, LNSTensor):

        if not isinstance(n, torch.Tensor):
            dtype = torch.int64 if (isinstance(n, int) or isinstance(n, float) and n.is_integer()) else torch.float64
            n = torch.tensor(n, dtype=dtype)

        result = LNSPowFunction.apply(x, n)

    else:
        x, n = format_lnstensor_operands(x, n)
        result = LNSPowFunction.apply(x, n.value)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _div(ops, x, y):
    result = (x - y + (y & 1)) ^ (y & 1)

    # overflow check for reference
    # torch.gt(y_packed, 0) & torch.lt(x_packed, -9223372036854775808 + y_packed)
    return torch.where(torch.eq(x | 1, LNS_ZERO), LNS_ZERO, result)

class LNSDivFunction(LNSFunction):
    """
    Division becomes subtraction in the logarithmic domain.

    Gradients are computed as follows:
    d/dx(x / y) = 1 / y
    d/dy(x / y) = -x / (y^2)
    """

    @staticmethod
    def forward(ops, x, y):
        return _div(ops, x, y)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, y = inputs
        ctx.save_for_backward(x, y)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, y = ctx.saved_tensors

        grad_x = ops.div(grad_output, y)
        grad_y = ops.square(y)
        grad_y = ops.div(x, grad_y)
        grad_y = ops.mul(grad_y, LNS_NEG_ONE)
        grad_y = ops.mul(grad_output, grad_y)

        grad_x = ops.sum_to_size(grad_x, x.shape)
        grad_y = ops.sum_to_size(grad_y, y.shape)

        return grad_x, grad_y

@implements(torch.div, _div, key='default', default=True)
def div(x, y, *, out=None):
    x, y = format_lnstensor_operands(x, y)
    result = LNSDivFunction.apply(x, y)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _reciprocal(ops, x):
    return ops.div(LNS_ONE, x)

class LNSReciprocalFunction(LNSFunction):
    """
    See LNSDivFunction for details on the internal computation.

    Gradients are calculated as follows:
    d/dx(1 / x) = -1 / (x ^ 2)
    """

    @staticmethod
    def forward(ops, x):
        return _reciprocal(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = ops.square(x)
        grad_x = ops.reciprocal(grad_x)
        grad_x = ops.mul(grad_x, LNS_NEG_ONE)
        grad_x = ops.mul(grad_output, grad_x)

        return grad_x

@implements(torch.reciprocal, _reciprocal, key='default', default=True)
def reciprocal(x, *, out=None):
    result = LNSReciprocalFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _exp(ops, x):
    e = ops.full_like(x, math.e)
    return ops.pow(e, ops.from_lns(x))

class LNSExpFunction(LNSFunction):
    """
    Exponentiation in the logarithmic domain requires us to
    convert the input to its floating point representation
    and then compute raising to the power of e.

    Gradients are computed as follows:
    d/dx(e ^ x) = e ^ x
    """

    @staticmethod
    def forward(ops, x):
        return _exp(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, ops, grad_output):
        exp_x, = ctx.saved_tensors

        grad_x = ops.mul(grad_output, exp_x)

        return grad_x

@implements(torch.exp, _exp, key='default', default=True)
def exp(x, *, out=None):
    result = LNSExpFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _log(ops, x):
    log_x = torch.log(ops.from_lns(x))
    return ops.to_lns(log_x)

class LNSLogFunction(LNSFunction):
    """
    Taking the logarithm in the logarithmic domain requires us to
    convert the input to its floating point representation and then
    compute the logarithm with respect to the base.

    Gradients are computed as follows:
    d/dx(log(x)) = 1 / x
    """

    @staticmethod
    def forward(ops, x):
        return _log(ops, x)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, = inputs
        ctx.save_for_backward(x)

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

        grad_x = ops.div(grad_output, x)

        return grad_x

@implements(torch.log, _log, key='default', default=True)
def log(x, *, out=None):
    result = LNSLogFunction.apply(x)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _prod(ops, x, dim=None, keepdim=False):
    if dim is None:
        flat = x.reshape(-1)

        out = flat[0]
        for i in range(1, flat.numel()):
            out = ops.mul(out, flat[i])

        if keepdim:
            out = out.reshape([1] * x.dim())

        return out

    # Reduction over a subset of the dimensions
    red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    red_dims = tuple(sorted(d % x.dim() for d in red_dims))

    # transpose so that the reduction dimensions are at the end, then flatten.
    permute_order = [d for d in range(x.dim()) if d not in red_dims] + list(red_dims)
    transposed = x.permute(*permute_order)
    outer_shape = transposed.shape[:-len(red_dims)]
    transposed = transposed.reshape(*outer_shape, -1)

    out = transposed[..., 0]
    for i in range(1, transposed.shape[-1]):
        out = ops.mul(out, transposed[..., i])

    # re-insert the reduced axes
    if keepdim:
        for d in red_dims:
            out = out.unsqueeze(d)

    return out

class LNSProdFunction(LNSFunction):
    """
    Product is computed using the multiplication operation.

    Gradients are computed as follows:
    d/dx(prod(x)) = prod(x) / x
    """

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return _prod(ops, x, dim, keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x, output)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, output = ctx.saved_tensors

        # 1. Broadcast the forward result so it matches x's shape
        if ctx.dim is not None and not ctx.keepdim:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(sorted(d % x.dim() for d in red_dims))

            for d in red_dims:
                output = output.unsqueeze(d)

        output_broadcast = output.expand_as(x)
        ratio = ops.div(output_broadcast, x)

        # broadcast grad_output to match x's shape
        if ctx.dim is not None and not ctx.keepdim:
            for d in red_dims:
                grad_output = grad_output.unsqueeze(d)

        grad_output = grad_output.expand_as(x)
        grad_x = ops.mul(grad_output, ratio)

        return grad_x, None, None

@implements(torch.prod, _prod, "default", default=True)
def prod(x, dim=None, keepdim=False, *, out=None):
    result = LNSProdFunction.apply(x, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _mean(ops, x, dim=None, keepdim=False):
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

    total = ops.sum(x, dims, keepdim)
    return ops.div(total, ops.to_lns(n_elem))

class LNSMeanFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, dim=None, keepdim=False):
        return _mean(ops, x, dim, keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, dim, keepdim = inputs
        ctx.save_for_backward(x)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, = ctx.saved_tensors

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

        grad_x = ops.div(grad_output, ops.to_lns(n_elem))
        if dims is None:
            grad_x = grad_x.expand(x.shape)

        else:
            if not ctx.keepdim:
                for d in sorted(dims):
                    grad_x = grad_x.unsqueeze(d)
            grad_x = grad_x.expand(x.shape)

        return grad_x, None, None

@implements(torch.mean, _mean, "default", default=True)
def mean(x, dim=None, keepdim=False, *, out=None):
    result = LNSMeanFunction.apply(x, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _var(ops, x, correction, dim=None, keepdim=False):
    if dim is None:
        red_dims = None
        N = x.numel()

    else:
        red_dims = (dim,) if isinstance(dim, int) else tuple(dim)
        red_dims = tuple(d % x.dim() for d in red_dims)
        N = 1
        for d in red_dims:
            N *= x.shape[d]

    n_elems = ops.to_lns(N)

    denom = ops.sub(n_elems, correction)
    if denom <= 0:
        raise ValueError("Degrees of freedom <= 0 for slice")

    total_x = ops.sum(x, dim=red_dims, keepdim=True)
    mean = ops.div(total_x, n_elems)

    diff = ops.sub(x, mean)
    sq_diff = ops.mul(diff, diff)
    total_sq = ops.sum(sq_diff, dim=red_dims, keepdim=keepdim)
    var = ops.div(total_sq, denom)

    return var

class LNSVarFunction(LNSFunction):

    @staticmethod
    def forward(ops, x, correction, dim=None, keepdim=False):
        return _var(ops, x, correction, dim, keepdim)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        x, correction, dim, keepdim = inputs
        ctx.save_for_backward(x, correction)
        ctx.dim = dim
        ctx.keepdim = keepdim

    @staticmethod
    def backward(ctx, ops, grad_output):
        x, correction = ctx.saved_tensors

        if ctx.dim is None:
            red_dims = None
            N = x.numel()

        else:
            red_dims = (ctx.dim,) if isinstance(ctx.dim, int) else tuple(ctx.dim)
            red_dims = tuple(d % x.dim() for d in red_dims)

            N = 1
            for d in red_dims:
                N *= x.shape[d]

        total_x = ops.sum(x, dim=red_dims, keepdim=True)
        n_elems = ops.to_lns(N)
        denom = ops.sub(n_elems, correction)
        mean = ops.div(total_x, n_elems)

        diff = ops.sub(x, mean)
        scale = ops.div(ops.to_lns(2.0), denom)

        grad_x = grad_output
        if red_dims is None:
            grad_x = grad_x.expand(x.shape)

        else:
            if not ctx.keepdim:
                for d in sorted(red_dims):
                    grad_x = grad_x.unsqueeze(d)
            grad_x = grad_x.expand(x.shape)

        grad_x = ops.mul(grad_x, ops.mul(diff, scale))

        return grad_x, None, None, None

@implements(torch.var, _var, "default", default=True)
def var(x, dim=None, *, correction=1, keepdim=False, out=None):
    x, correction = format_lnstensor_operands(x, correction)
    result = LNSVarFunction.apply(x, correction, dim, keepdim)

    if out is not None:
        return out._inplace_copy(result)

    return result

def _matmul(ops, A, B):
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
                        dtype=torch.int64, device=A.device)

    # Perform matrix multiplication in log space
    for k in range(K_A):
        term = ops.mul(
            A[..., :, k].unsqueeze(-1), # (..., M, 1)
            B[..., k, :].unsqueeze(-2)  # (..., 1, N)
        )
        result = ops.add(result, term)

    if prepended_A:
        result = result.squeeze(-2) # Remove extra M dimension
    if appended_B:
        result = result.squeeze(-1) # Remove extra N dimension

    return result

class LNSMatmulFunction(LNSFunction):
    """
    Matrix multiplication uses the lns addition and
    multiplication functions to compute the result.

    Gradients are computed as follows:
    d/dA(A @ B) = B^T
    d/dB(A @ B) = A^T
    """

    @staticmethod
    def forward(ops, A, B):
        return _matmul(ops, A, B)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        # Here we must repeat the unsqueezing logic from forward
        # to ensure that we can correctly compute the gradients.
        # todo: This is a bit of a hack, we should ideally handle
        # the unsqueezing in a more elegant way. 
        A, B = inputs

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

        ctx.save_for_backward(A, B)

    @staticmethod
    def backward(ctx, ops, grad_output):
        A, B = ctx.saved_tensors

        #  Re-introduce squeezed dimensions
        if ctx.prepended_A and not ctx.appended_B:
            grad_output = grad_output.unsqueeze(-2)
        elif ctx.appended_B and not ctx.prepended_A:
            grad_output = grad_output.unsqueeze(-1)
        elif ctx.prepended_A and ctx.appended_B:
            grad_output = grad_output.unsqueeze(-1).unsqueeze(-1)

        # Compute gradients w.r.t A and B after broadcasting
        grad_A = ops.matmul(grad_output, B.transpose(-1, -2))
        grad_B = ops.matmul(A.transpose(-1, -2), grad_output)

        # Reduce gradients to match original shapes before broadcasting
        # We need to sum over dimensions that were broadcasted

        # For grad_A: reduce to shape before broadcasting
        A_shape_before_broadcast = ctx.A_shape_before_broadcast
        while grad_A.dim() > len(A_shape_before_broadcast):
            grad_A = ops.sum(grad_A, dim=0)

        # Sum over any dimensions that were size 1 and got broadcasted
        for i in range(len(A_shape_before_broadcast) - 2):  # Don't touch matrix dims
            if A_shape_before_broadcast[i] == 1 and grad_A.shape[i] > 1:
                grad_A = ops.sum(grad_A, dim=i, keepdim=True)

        # For grad_B: reduce to shape before broadcasting
        B_shape_before_broadcast = ctx.B_shape_before_broadcast
        while grad_B.dim() > len(B_shape_before_broadcast):
            grad_B = ops.sum(grad_B, dim=0)

        # Sum over any dimensions that were size 1 and got broadcasted
        for i in range(len(B_shape_before_broadcast) - 2):  # Don't touch matrix dims
            if B_shape_before_broadcast[i] == 1 and grad_B.shape[i] > 1:
                grad_B = ops.sum(grad_B, dim=i, keepdim=True)

        if ctx.prepended_A:
            grad_A = grad_A.squeeze(0) # Remove extra M dimension
        if ctx.appended_B:
            grad_B = grad_B.squeeze(-1) # Remove extra N dimension

        return grad_A, grad_B

@implements(torch.matmul, _matmul, "default", default=not CSRC_AVAILABLE)
def matmul(A, B, *, out=None):
    A, B = format_lnstensor_operands(A, B)
    result = LNSMatmulFunction.apply(A, B)

    if out is not None:
        return out._inplace_copy(result)

    return result

class LNSTransposeFunction(LNSFunction):
    """
    Transpose operation simply rearranges the dimensions
    of the input tensor. It doesn't change the underlying
    representations, so the forward pass isn't special.

    Gradients are computed as follows:
    d/dx(A.T) = 1
    """

    @staticmethod
    def forward(ops, A, dim0, dim1):
        return torch.transpose(A, dim0, dim1)

    @staticmethod
    def setup_context(ctx, ops, inputs, output):
        _, dim0, dim1 = inputs
        ctx.dim0 = dim0
        ctx.dim1 = dim1

    @staticmethod
    def backward(ctx, ops, grad_output):
        grad_x = torch.transpose(grad_output, ctx.dim0, ctx.dim1)
        return grad_x, None, None

@implements(torch.transpose, LNSTransposeFunction.forward, "default", default=True)
def transpose(A, dim0, dim1):
    return LNSTransposeFunction.apply(A, dim0, dim1)
