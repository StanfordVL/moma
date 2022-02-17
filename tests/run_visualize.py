import os
from pathlib import Path
import random

from momaapi import MOMA, AnnVisualizer, StatVisualizer


def visualize_anns(moma, dir_vis):
  visualizer = AnnVisualizer(moma, dir_vis)

  """ visualize 10 random sub-activities """
  ids_sact = moma.get_ids_sact()
  ids_sact = random.choices(ids_sact, k=10)
  for id_sact in ids_sact:
    visualizer.show_sact(id_sact, vstack=False)

  """ visualize a random higher-order interaction that contains an object basket """
  ids_hoi = moma.get_ids_hoi(cnames_object=['basket'])
  id_hoi = random.choice(ids_hoi)
  visualizer.show_hoi(id_hoi)


def visualize_stats(moma, dir_vis):
  visualizer = StatVisualizer(moma, dir_vis)
  visualizer.show(with_split=False)
  visualizer.show(with_split=True)


def main():
  dir_moma = os.path.join(Path.home(), 'data/moma')
  dir_vis = os.path.join(Path.home(), 'data/moma/vis')

  moma = MOMA(dir_moma)

  visualize_anns(moma, dir_vis)
  visualize_stats(moma, dir_vis)


if __name__ == '__main__':
  main()
