import random

from moma_api import MOMA, AnnVisualizer


def main(dir_moma='/home/alan/ssd/moma', dir_vis='/home/alan/ssd/moma/vis'):
  moma = MOMA(dir_moma)
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


if __name__ == '__main__':
  main()
