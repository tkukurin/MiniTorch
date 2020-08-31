import minitorch
import pytest
from .strategies import assert_close


def _matmul(m1, m2, expect, op=minitorch.matmul):
  a = minitorch.tensor_fromlist(m1)
  b = minitorch.tensor_fromlist(m2)
  c = op(a, b)
  c_expect = minitorch.tensor_fromlist(expect)

  print(f'{a.shape} * {b.shape} = {c_expect.shape}')
  print('result', c)
  print('expect', c_expect)

  for ind in c._tensor.indices():
    assert_close(c[ind], c_expect[ind])

  return c, c_expect


@pytest.mark.task3_1
def test_mm():
  _matmul(
    m1=[[1,2,3],
        [4,5,6]],
    m2=[[1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]],
    expect=[[ 38,  44,  50,  56],
            [ 83,  98, 113, 128]],
    op=minitorch.matmul
  )


@pytest.mark.task3_1
def test_broad_different_size0_mm():
  _matmul(m1=[
    [[1,2,3],
     [4,5,6]],
    [[1,2,3],
     [4,5,6]],],
  m2=[
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]],],
  expect=[
    [[ 38,  44,  50,  56],
     [ 83,  98, 113, 128]],
    [[ 38,  44,  50,  56],
     [ 83,  98, 113, 128]]],
  op=minitorch.matmul)


@pytest.mark.task3_1
def test_broad_same_size0_mm():
  _matmul(m1=[
    [[1,2,3],
     [4,5,6]],
    [[1,2,3],
     [4,5,6]],],
  m2=[
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]],
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]], ],
  expect=[
    [[ 38,  44,  50,  56],
     [ 83,  98, 113, 128]],
    [[ 38,  44,  50,  56],
     [ 83,  98, 113, 128]]],
  op=minitorch.matmul)


@pytest.mark.task3_4
def test_cuda_mm():
  _matmul(m1=[
    [[1,2,3],
     [4,5,6]],
    [[1,2,3],
     [4,5,6]],],
  m2=[
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]],
    [[1,2,3,4],
     [5,6,7,8],
     [9,10,11,12]], ],
  expect=[
    [[ 38,  44,  50,  56],
     [ 83,  98, 113, 128]],
    [[ 38,  44,  50,  56],
     [ 83,  98, 113, 128]]],
  op=minitorch.cuda_matmul)

