def flatten(x):
  if isinstance(x, tuple) or isinstance(x, list) or isinstance(x, set):
    return [z for y in x for z in flatten(y)]
  else:
    return [x]
