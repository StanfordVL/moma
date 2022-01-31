import random

from moma_api import MOMA, Visualizer


def main():
  dir_moma = '/home/alan/ssd/moma'
  moma = MOMA(dir_moma)
  visualizer = Visualizer(moma)

  ids_sact = moma.get_ids_sact()
  ids_sact = random.choices(ids_sact, k=10)

  for id_sact in ids_sact:
    visualizer.show_sact(id_sact, vstack=False)

  # ids_hoi = moma.get_ids_hoi(cnames_object=['basket'])
  # visualizer.show_hoi(ids_hoi[0])


if __name__ == '__main__':
  main()
