from distinctipy import distinctipy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from torchvision import io


class Visualizer:
  def __init__(self, moma):
    self.moma = moma

  @staticmethod
  def get_palette(ids):
    colors_box = distinctipy.get_colors(len(ids))
    colors_text = [distinctipy.get_text_color(color_box) for color_box in colors_box]
    colors_box = [tuple(int(x*255) for x in color_box) for color_box in colors_box]
    colors_text = [tuple(int(x*255) for x in color_text) for color_text in colors_text]

    palette = {id: (color_box, color_text) for id, color_box, color_text in zip(ids, colors_box, colors_text)}
    return palette

  @staticmethod
  def draw_entities(image, entities, palette):
    for entity in entities:
      draw = ImageDraw.Draw(image)
      y1, x1, y2, x2 = entity.bbox.y1, entity.bbox.x1, entity.bbox.y2, entity.bbox.x2
      width_line = int(max(image.size)*0.003)
      font = ImageFont.truetype('Ubuntu-R.ttf', int(max(image.size)*0.02))
      width_text, height_text = font.getsize(entity.cname)

      draw.rectangle(((x1, y1), (x2, y2)), width=width_line, outline=palette[entity.id][0])
      draw.rectangle(((x1, y1), (x1+width_text+2*width_line, y1+height_text+2*width_line)), fill=palette[entity.id][0])
      draw.text((x1+width_line, y1+width_line), entity.cname, fill=palette[entity.id][1], font=font)

    return image

  def draw_image(self, ann_hoi, palette):
    path_image = self.moma.get_path(id_hoi=ann_hoi.id)
    image = io.read_image(path_image).permute(1, 2, 0).numpy()
    image = Image.fromarray(image).convert('RGB')
    image = self.draw_entities(image, ann_hoi.actors, palette)
    image = self.draw_entities(image, ann_hoi.objects, palette)
    return image

  def show_hoi(self, id_hoi):
    ann_hoi = self.moma.get_anns_hoi(ids_hoi=[id_hoi])[0]
    palette = self.get_palette(ann_hoi.ids_actor+ann_hoi.ids_object)
    image = self.draw_image(ann_hoi, palette)

    plt.figure(figsize=(16, 9))
    plt.axis('off')
    plt.imshow(image)
    plt.show()

  def show_sact(self, id_sact):
    fig, ax = plt.subplots(figsize=(16, 9))

    ids_hoi = self.moma.get_ids_hoi(ids_sact=[id_sact])
    ann_sact = self.moma.get_anns_sact(ids_sact=[id_sact])[0]
    palette = self.get_palette(ann_sact.ids_actor+ann_sact.ids_object)

    artists = []
    for i, id_hoi in enumerate(ids_hoi):
      ann_hoi = self.moma.get_anns_hoi(ids_hoi=[id_hoi])[0]
      image = self.draw_image(ann_hoi, palette)

      artist = ax.imshow(image, animated=True)
      if i == 0:
        ax.imshow(image)
      artists.append([artist])

    _ = animation.ArtistAnimation(fig, artists, interval=250, blit=True, repeat_delay=1000)
    plt.axis('off')
    plt.show()
