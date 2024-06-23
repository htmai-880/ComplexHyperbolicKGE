# Inherited from https://github.com/HazyResearch/KGEmb
"""Euclidean operations utils functions."""

import torch


def euc_sqdistance(x, y):
    """Compute euclidean squared distance between tensors.

    Args:
        x: torch.Tensor of shape (N1 x d)
        y: torch.Tensor of shape (N2 x d)
        eval_mode: boolean

    Returns:
        torch.Tensor of shape N1 x 1 with pairwise squared distances if eval_mode is false
        else torch.Tensor of shape N1 x N2 with all-pairs distances

    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    return x2 + y2 - 2 * xy


def givens_rotations(r, x, scale=None, inverse=False):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    if not scale is None:
        scaler = scale.view(*r.shape[:-1], -1)
        scaler = scaler /(torch.abs(scaler) + 1e-3) # size : (N * d/2 * 1)
    givens = r.view((*r.shape[:-1], -1, 2))   # givens size: (N * d/2 * 2)
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)   # normalize to make cos & sin

    x = x.view((*r.shape[:-1], -1, 2))   # x size: (N * d/2 * 2)
    if not scale is None:
        x_rot = torch.zeros_like(x)
        abs_scaler = torch.abs(scaler)
        if inverse:
            x_rot[..., 0] = (1/abs_scaler) * (givens[..., 0] * x[..., 0] + givens[..., 1] * x[..., 1])
            x_rot[..., 1] = (1/scaler) * (givens[..., 0] * x[..., 1] - givens[..., 1] * x[..., 0])
        else:
            x_rot[..., 0] = abs_scaler * givens[..., 0] * x[..., 0] - scaler * givens[..., 1] * x[..., 1]
            x_rot[..., 1] = abs_scaler * givens[..., 1] * x[..., 0] + scaler * givens[..., 0] * x[..., 1]
    else:
        if inverse:
            givens[..., 1].mul_(-1)
        x_rot = givens[..., 0:1] * x + givens[..., 1:] * torch.cat((-x[..., 1:], x[..., 0:1]), dim=-1)   # x_rot size: (N * d/2 * 2)

    return x_rot.view(r.size())   # size: (N * d)


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view((r.shape[0], -1, 2))
    x_ref = givens[..., 0:1] * torch.cat((x[..., 0:1], -x[..., :1]), dim=-1) + givens[..., 1:] * torch.cat(
        (x[..., 1:], x[..., 0:1]), dim=-1)
    return x_ref.view(r.size())


def givens_unitary(a, b, angle, z, lift=False):
    """Givens unitary transformation.

    Args:
        a: torch.Tensor of shape (N x d), unitary parameters (real, complex dimension d/2)
        b: torch.Tensor of shape (N x d), unitary parameters (real, complex dimension d/2)
        angle: torch.Tensor of shape (N x d), unitary determinant (real, complex dimension d/2)
        z : torch.Tensor of shape (N x d), points to rotate (complex, complex dimension d)


    Returns:
        torch.Tensor os shape (N x d) representing rotation of z by the unitary transformation
    """
    # z is already of complex dtype
    # aa* + bb* = 1
    # Matrix:
    # [a               b      ]
    # [-eitheta b*  eitheta a*]
    # a, b are of size (batch_size, 2d) (real)
    # angle is of size (batch_size, 2d) (real)
    # z is of size (batch_size, 2d) (complex)
    a_real, a_imag = torch.chunk(a, 2, dim=-1)
    b_real, b_imag = torch.chunk(b, 2, dim=-1)

    # Normalize a and b
    a_ = a_real + 1j * a_imag # (batch_size, ...,  d/2)
    b_ = b_real + 1j * b_imag # (batch_size, ..., d/2)

    norm = a_real**2 + a_imag**2 + b_real**2 + b_imag**2 # (batch_size, ..., d/2)
    norm = torch.sqrt(norm)

    a_ = a_ / norm
    b_ = b_ / norm
    if not angle is None:
        cos_theta, sin_theta = torch.chunk(angle, 2, dim=-1)
        eitheta = cos_theta + 1j * sin_theta # (batch_size, ..., d/2)
        eitheta = eitheta / torch.abs(eitheta) # Incidentally, this is also the determinant of each 2x2 unitary matrix.
        # If we apply a lift, the last component will be the inverse of the product of those.
        assert eitheta.size() == b_.size(), f"{eitheta.size()} != {b_.size()}"
    else:
        eitheta = 1

    z = z.view(*a_.shape, 2) # (batch_size, ..., d/2, 2)
    out = torch.zeros_like(z)

    out[..., 0] = a_ * z[..., 0] + b_ * z[..., 1]
    out[..., 1] = - eitheta * b_.conj() * z[..., 0] + eitheta * a_.conj() * z[..., 1]
    out = out.view(a.size())
    if not lift:
        return out
    # Get the determinant
    det = torch.prod(eitheta, dim=-1, keepdim=True).conj() # (batch_size, ... 1)
    det = det/ torch.abs(det)
    return out, det

    



def multi_bmm(input, mat2):
    """Batch matrix multiplication, where the inputs can have multiple batch dimensions.

    Args:
        input (tensor): Input tensor of size (*, n, m)
        mat2 (tensor): Input tensor of size (*, m, p)
    
    Returns:
        res (tensor): Result tensor of size (*, n, p)
    """
    input_shape = input.size()
    mat2_shape = mat2.size()
    input = input.view(-1, *input_shape[-2:])
    mat2 = mat2.view(-1, *mat2_shape[-2:])
    assert input_shape[:-2] == mat2_shape[:-2], f"Batch dimensions do not match: {input_shape[:-2]} != {mat2_shape[:-2]}"
    assert input_shape[-1] == mat2_shape[-2], f"Matrix dimensions do not match: {input_shape[-1]} != {mat2_shape[-2]}"
    res = torch.bmm(input, mat2)
    return res.view(*input_shape[:-1], mat2_shape[-1])


def multi_index_select(source, indices):
    ind_shape = indices.size()
    index = indices.view(-1)
    out = torch.index_select(source, 0, index)
    out = out.view(*ind_shape, *source.shape[1:])
    return out

def norm_clamp(source, min=None, max=None, p=2, dim=-1):
    assert (not min is None) or (not max is None), "at least min or max should be specified."
    norms = torch.norm(source, p=p, dim=dim, keepdim=True)
    min_ = None if min is None else min * (1 + 1e-3)
    max_ = None if max is None else max * (1 - 1e-3)
    norms_clamped = torch.clamp(norms, min=min_, max=max_)

    source = (source / norms) * norms_clamped
    return source 





