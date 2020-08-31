import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
)
from .tensor import Function

from numba import njit, prange
count = njit()(count)
index_to_position = njit()(index_to_position)
broadcast_index = njit()(broadcast_index)

#@njit(parallel=False)
def nxt(xs, shape, range_=None):
  range_ = range_ or len(xs) - 2
  for i in range(range_):
    xs[i] = (xs[i]+1) % shape[i]
    if xs[i]:
      return i, xs
  return range_ - 1, xs

#@njit(parallel=True)
def _matrix_multiply(
    out, out_shape, out_strides, a, a_shape, a_strides, b, b_shape, b_strides):
  oi = np.zeros_like(out_shape)
  ai = np.zeros_like(a_shape)
  bi = np.zeros_like(b_shape)

  for outer_a in range(np.prod(a_shape[:-2])):
    for outer_b in range(np.prod(b_shape[:-2])):
      for ra in range(a_shape[-2]):
        for cb in range(b_shape[-1]):
          ai[-2] = oi[-2] = ra
          bi[-1] = oi[-1] = cb
          opos = index_to_position(oi, out_strides)
          out[opos] = 0.0
          for i in range(a_shape[-1]):
            ai[-1] = i
            bi[-2] = i

            apos = index_to_position(ai, a_strides)
            bpos = index_to_position(bi, b_strides)
            out[opos] += a[apos]*b[bpos]

      ch_ix, bi = nxt(bi, b_shape)
      oi[ch_ix] = bi[ch_ix]

    ch_ix, ai = nxt(ai, a_shape)
    oi[ch_ix] = ai[ch_ix]

  return out


def matrix_multiply(a, b):
  ls = list(a.shape)
  assert a.shape[-1] == b.shape[-2]
  ls[-1] = b.shape[-1]
  out = a.zeros(tuple(ls))
  _matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
  return out


class MatMul(Function):
  @staticmethod
  def forward(ctx, t1, t2):
    ctx.save_for_backward(t1, t2)
    return matrix_multiply(t1, t2)

  @staticmethod
  def backward(ctx, grad_output):
    t1, t2 = ctx.saved_values
    return (
      matrix_multiply(grad_output, t2.permute(0, 2, 1)),
      matrix_multiply(t1.permute(0, 2, 1), grad_output),)


matmul = MatMul.apply
