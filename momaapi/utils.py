from functools import wraps
from time import time


def flatten(x):
  if isinstance(x, tuple) or isinstance(x, list) or isinstance(x, set):
    return [z for y in x for z in flatten(y)]
  else:
    return [x]


def timeit(f):
  @wraps(f)
  def _timeit(*args, **kwargs):
      ts = time()
      result = f(*args, **kwargs)
      te = time()
      print(f'{f.__name__}() took {te-ts} sec')
      return result

  return _timeit
