import numpy as np
from .tensor_data import (
  count,
  index_to_position,
  broadcast_index,
  shape_broadcast,
  MAX_DIMS,
)


def tensor_map(fn):
  """
  Higher-order tensor map function.

  Args:
    fn: function mappings floats-to-floats to apply.
    out (array): storage for out tensor.
    out_shape (array): shape for out tensor.
    out_strides (array): strides for out tensor.
    in_storage (array): storage for in tensor.
    in_shape (array): shape for in tensor.
    in_strides (array): strides for in tensor.
  """
  def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
   oi = np.zeros_like(out_strides)
   ii = np.zeros_like(in_strides)

   def nxt(xs, shape):
    for i in range(len(xs)):
     xs[i] = (xs[i]+1) % shape[i]
     if xs[i] > 0: break
    return xs

   for pos in range(np.prod(out_shape)):
    incurr = in_storage[index_to_position(ii, in_strides)]
    out[index_to_position(oi, out_strides)] = fn(incurr)
    oi = nxt(oi, out_shape)
    ii = nxt(ii, in_shape)

  return _map


def map(fn):
  f = tensor_map(fn)

  def ret(a, out=None):
    if out is None:
      out = a.zeros(a.shape)
    f(*out.tuple(), *a.tuple())
    return out

  return ret


def tensor_zip(fn):
  """
  Higher-order tensor zipWith (or map2) function.

  Args:
    fn: function mappings two floats to float to apply.
    out (array): storage for `out` tensor.
    out_shape (array): shape for `out` tensor.
    out_strides (array): strides for `out` tensor.
    a_storage (array): storage for `a` tensor.
    a_shape (array): shape for `a` tensor.
    a_strides (array): strides for `a` tensor.
    b_storage (array): storage for `b` tensor.
    b_shape (array): shape for `b` tensor.
    b_strides (array): strides for `b` tensor.
  """

  def _zip(out, out_shape, out_strides, a, a_shape,
      a_strides, b, b_shape, b_strides):

    oi = np.zeros_like(out_strides)
    ai = np.zeros_like(a_strides)
    bi = np.zeros_like(b_strides)

    a1 = [i for i,ash in enumerate(a_shape) if ash==1]
    b1 = [i for i,bsh in enumerate(b_shape) if bsh==1]
    #print(a1,b1)

    def nxt(xs, shape):
      for i in range(len(xs)):
        xs[i] = (xs[i]+1) % shape[i]
        if xs[i]: break
      return xs

    for pos in range(np.prod(out_shape)):
      acurr = a[index_to_position(ai, a_strides)]
      bcurr = b[index_to_position(bi, b_strides)]
      out[index_to_position(oi, out_strides)] = fn(acurr, bcurr)
      oi = nxt(oi, out_shape)
      ai = np.copy(oi)
      bi = np.copy(oi)
      ai[a1] = 0
      bi[b1] = 0
      #print(ai,bi)

  return _zip


def zip(fn):
  f = tensor_zip(fn)

  def ret(a, b):
    if a.shape != b.shape:
      c_shape = shape_broadcast(a.shape, b.shape)
    else:
      c_shape = a.shape
    out = a.zeros(c_shape)
    f(*out.tuple(), *a.tuple(), *b.tuple())
    return out

  return ret


def tensor_reduce(fn):
  """
  Higher-order tensor reduce function.

  Args:
    fn: reduction function mapping two floats to float.
    out (array): storage for `out` tensor.
    out_shape (array): shape for `out` tensor.
    out_strides (array): strides for `out` tensor.
    a_storage (array): storage for `a` tensor.
    a_shape (array): shape for `a` tensor.
    a_strides (array): strides for `a` tensor.
    reduce_shape (array): shape of reduction (
     1 for dimension kept, shape value for dimensions summed out)
    reduce_size (int): size of reduce shape
  """
  def _reduce(
      out, out_shape, out_strides, a, a_shape, a_strides, reduce_shape,
      reduce_size):
    oi = np.zeros_like(out_strides)
    ai = np.zeros_like(a_strides)
    ch_ix = 0

    def nxt(xs, shape):
      for i in range(len(xs)):
        xs[i] = (xs[i]+1) % shape[i]
        if xs[i]: break
      return i, xs

    for pos in range(np.prod(a_shape)):
      oi_pos = index_to_position(oi, out_strides)
      ai_pos = index_to_position(ai, a_strides)
      out[oi_pos] = fn(out[oi_pos], a[ai_pos])
      ch_ix, ai = nxt(ai, a_shape)
      if reduce_shape[ch_ix] == 1:
        _, oi = nxt(oi, out_shape)

  return _reduce


def reduce(fn, start=0.0):
  f = tensor_reduce(fn)

  def ret(a, dims=None, out=None):
    if out is None:
      out_shape = list(a.shape)
      for d in dims:
        out_shape[d] = 1
      # Other values when not sum.
      out = a.zeros(tuple(out_shape))
      out._tensor._storage[:] = start

    # NOTE(tk) diff is always 0 ?? out_shape = list(a.shape)
    diff = len(a.shape) - len(out.shape)

    reduce_shape = []
    reduce_size = 1
    for i, s in enumerate(a.shape):
      if i < diff or out.shape[i - diff] == 1:
        reduce_shape.append(s)
        reduce_size *= s
      else:
        reduce_shape.append(1)
    #assert len(out.shape) == len(a.shape)
    f(*out.tuple(), *a.tuple(), reduce_shape, reduce_size)
    return out

  return ret


class TensorOps:
  map = map
  zip = zip
  reduce = reduce

