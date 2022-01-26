import random


def get_palette(num_colors):
  """
  Reference: https://qr.ae/pGBSKk
  """
  palette = []
  r = int(random.random()*256)
  g = int(random.random()*256)
  b = int(random.random()*256)
  step = 256/num_colors
  for i in range(num_colors):
    r += step
    g += step
    b += step
    r = int(r)%256
    g = int(g)%256
    b = int(b)%256
    palette.append((r, g, b))

  return palette