import os

import matplotlib.pyplot as plt
from torchvision import io

from moma_api import MOMA


def main():
  dir_moma = '/home/alan/ssd/moma'
  moma = MOMA(dir_moma)

  id_hoi = '04016_0004971633'
  id_sact = moma.get_ids_sact(ids_hoi=[id_hoi])[0]
  id_act = moma.get_ids_act(ids_sact=[id_sact])[0]
  ann_hoi = moma.get_anns_hoi(ids_hoi=[id_hoi])[0]

  file_hoi = os.path.join(dir_moma, f'videos/higher_order_interaction/{id_hoi}.png')
  image = io.read_image(file_hoi).permute(1, 2, 0).numpy()

  print(image.shape)

  plt.figure(figsize=(12, 8))
  plt.title(id_hoi)
  plt.axis('off')
  plt.imshow(image)



if __name__ == '__main__':
  main()
