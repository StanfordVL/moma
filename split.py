from pprint import pprint
import random
from sklearn.metrics.pairwise import cosine_similarity

from moma_api import MOMA


def get_similarity(stats_per_class_train, stats_per_class_val):
  for key in stats_per_class_train:
    counts_train = stats_per_class_train[key]['counts']
    counts_val = stats_per_class_val[key]['counts']
    assert len(counts_train) == len(counts_val)


def main():
  dir_moma = '/home/alan/ssd/moma'
  moma = MOMA(dir_moma)
  ids_act = list(moma.anns_act.keys()).copy()

  random.shuffle(ids_act)

  ids_act_train = ids_act[:1100]
  ids_act_val = ids_act[1100:]

  stats_overall_train, stats_per_class_train = moma.get_stats(ids_act_train)
  stats_overall_val, stats_per_class_val = moma.get_stats(ids_act_val)

  pprint(stats_overall_train)
  pprint(stats_overall_val)


if __name__ == '__main__':
  main()
