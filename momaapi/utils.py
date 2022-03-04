from collections import Iterable


def flatten(x):
  if isinstance(x, Iterable):
    return [z for y in x for z in flatten(y)]
  else:
    return [x]
