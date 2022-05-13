import os
import pickle


def save_cache(dir_cache, named_variables):
  os.makedirs(dir_cache, exist_ok=True)

  for name, variable in named_variables.items():
    assert variable is not None
    with open(os.path.join(dir_cache, name), 'wb') as f:
      pickle.dump(variable, f)


def load_cache(dir_cache, names):
  if not all([os.path.exists(os.path.join(dir_cache, name)) for name in names]):
    raise FileNotFoundError

  variables = []
  for name in names:
    with open(os.path.join(dir_cache, name), 'rb') as f:
      variable = pickle.load(f)
      variables.append(variable)

  return variables
