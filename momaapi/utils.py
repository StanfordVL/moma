from functools import wraps
import time


def timeit(f):
  @wraps(f)
  def _timeit(*args, **kwargs):
      ts = time.time()
      result = f(*args, **kwargs)
      te = time.time()
      print(f'{f.__module__}.{f.__name__}() took {te-ts} sec')
      return result

  return _timeit
