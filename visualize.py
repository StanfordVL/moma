from moma_api import MOMA, Visualizer


def main():
  dir_moma = '/home/alan/ssd/moma'
  moma = MOMA(dir_moma)
  visualizer = Visualizer(moma)

  # visualizer.show_hoi('13433_0018978556')
  # visualizer.show_sact('13433')

  ids_hoi = moma.get_ids_hoi(cnames_object=['basket'])
  visualizer.show_hoi(ids_hoi[0])


if __name__ == '__main__':
  main()
