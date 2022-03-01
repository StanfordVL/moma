import numpy as np
import os
import pickle

import momaapi


def main():
  dir_moma = '/home/alan/data/moma'

  with open(os.path.join(dir_moma, 'weights/model_final_571f7c.pkl'), 'rb') as f:
    weights = pickle.load(f)

  moma = momaapi.MOMA(dir_moma, toy=True)
  lvis_indices = np.array([moma.lvis_mapper[cname] for cname in ['actor']+moma.get_cnames('object')])

  w = weights['model']['roi_heads.mask_head.predictor.weight'][lvis_indices]
  b = weights['model']['roi_heads.mask_head.predictor.bias'][lvis_indices]
  weights['model']['roi_heads.mask_head.predictor.weight'] = w
  weights['model']['roi_heads.mask_head.predictor.bias'] = b

  with open(os.path.join(dir_moma, 'weights/model_final_571f7c_moma.pkl'), 'wb') as f:
    pickle.dump(weights, f)


if __name__ == '__main__':
  main()
