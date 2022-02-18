import os
from pathlib import Path
from pprint import pprint

import momaapi


def analyze_split(moma):
  stats_overall_train, stats_per_class_train = momaapi.get_stats(moma, 'train')
  stats_overall_val, stats_per_class_val = momaapi.get_stats(moma, 'val')

  dists_overall = momaapi.get_dist_overall(stats_overall_train, stats_overall_val)
  dists_per_class = momaapi.get_dist_per_class(stats_per_class_train, stats_per_class_val)

  print('The quality of our split (cosine distance b/w train & val, the smaller the better):')
  pprint(dists_overall)
  pprint(dists_per_class)


def analyze_stats(moma):
  stats_overall, stats_per_class = momaapi.get_stats(moma)
  pprint(stats_overall)
  pprint(stats_per_class)


def main():
  dir_moma = os.path.join(Path.home(), 'data/moma')

  moma = momaapi.MOMA(dir_moma)

  analyze_split(moma)
  analyze_stats(moma)


if __name__ == '__main__':
  main()
