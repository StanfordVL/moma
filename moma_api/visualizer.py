from collections import defaultdict
from distinctipy import distinctipy
import glob
import math
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import pygraphviz as pgv
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

  def show_sact(self, id_sact, vstack=True):
    os.makedirs(f'./figures/{id_sact}', exist_ok=True)

    ann_sact = self.moma.get_anns_sact(ids_sact=[id_sact])[0]
    ids_hoi = self.moma.get_ids_hoi(ids_sact=[id_sact])
    anns_hoi = self.moma.get_anns_hoi(ids_hoi=ids_hoi)
    palette = self.get_palette(ann_sact.ids_actor+ann_sact.ids_object)

    """ bbox """
    for i, id_hoi in enumerate(ids_hoi):
      ann_hoi = self.moma.get_anns_hoi(ids_hoi=[id_hoi])[0]
      image = self.draw_image(ann_hoi, palette)
      image.save(f'./figures/{id_sact}/bbox_{str(i).zfill(2)}.png')

    """ graph """
    # get node & edge positions
    info_nodes = []
    for id_entity, cname_entity in zip(ann_sact.ids_actor+ann_sact.ids_object,
                                       ann_sact.cnames_actor+ann_sact.cnames_object):
      info_nodes.append((id_entity, cname_entity))

    info_edges = []
    for ann_hoi in anns_hoi:
      edge_to_labels = defaultdict(list)
      for description in ann_hoi.tas+ann_hoi.rels:
        edge_to_labels[(description.id_src, description.id_trg)].append(description.cname)
      for description in ann_hoi.ias+ann_hoi.atts:
        edge_to_labels[(description.id_src, description.id_src)].append(description.cname)
      for edge, labels in edge_to_labels.items():
        info_edges.append((*edge, '\n'.join(labels)))
    info_edges = list(set(info_edges))

    G = pgv.AGraph(directed=True, strict=False)
    for node, label in info_nodes:
      G.add_node(node, label=node, xlabel=label, fontcolor='slategray', color='slategray', shape='circle')
    for node_src, node_trg, label in info_edges:
      G.add_edge((node_src, node_trg), label=label, color='slategray', fontcolor='slategray', fontsize='10', len=2)
    G.layout('neato')
    # G.draw('figures/graphs/tmp.png')

    pos_node = {node:node.attr['pos'] for node in G.nodes()}
    pos_edge = {(*edge, edge.attr['label']):edge.attr['pos'] for edge in G.edges()}

    G.remove_nodes_from([info_node[0] for info_node in info_nodes])

    # draw graphs
    for i, ann_hoi in enumerate(anns_hoi):
      # draw nodes
      data_node = []
      for info_node in info_nodes:
        if info_node in [(actor.id, actor.cname) for actor in ann_hoi.actors]:
          color = 'steelblue'
        elif info_node in [(object.id, object.cname) for object in ann_hoi.objects]:
          color = 'salmon3'
        else:
          color = 'grey'
        data_node.append((*info_node, color))

      for node, label, color in data_node:
        pos = pos_node[node]
        G.add_node(node, label=node, xlabel=label, pos=pos, fontcolor=color, color=color, shape='circle')

      # draw edges
      edge_to_labels = defaultdict(list)
      for description in ann_hoi.tas+ann_hoi.rels:
        edge_to_labels[(description.id_src, description.id_trg)].append(description.cname)
      for description in ann_hoi.ias+ann_hoi.atts:
        edge_to_labels[(description.id_src, description.id_src)].append(description.cname)
      edge_to_label = {edge:'\n'.join(labels) for edge, labels in edge_to_labels.items()}

      data_edge = []
      for info_edge in info_edges:
        if info_edge in [(*edge, label) for edge, label in edge_to_label.items()]:
          data_edge.append((*info_edge, 'slategray'))
        else:
          data_edge.append((*info_edge, '#00000000'))

      for node_src, node_trg, label, color in data_edge:
        pos = pos_edge[(node_src, node_trg, label)]
        G.add_edge((node_src, node_trg), label=label, pos=pos, color=color, fontcolor=color, fontsize='10', len=2)

      G.draw(f'./figures/{id_sact}/graph_{str(i).zfill(2)}.eps')
      G.remove_nodes_from([info_node[0] for info_node in info_nodes])

    """ gif """
    paths_bbox = sorted(glob.glob(f'./figures/{id_sact}/bbox_*.png'))
    paths_graph = sorted(glob.glob(f'./figures/{id_sact}/graph_*.eps'))

    images_bbox = [Image.open(path_bbox) for path_bbox in paths_bbox]
    images_graph = [Image.open(path_graph) for path_graph in paths_graph]

    assert (all(image_bbox.size == images_bbox[0].size for image_bbox in images_bbox))
    assert (all(image_graph.size == images_graph[0].size for image_graph in images_graph))

    width_bbox, height_bbox = images_bbox[0].size
    width_graph, height_graph = images_graph[0].size

    images_graph_resize = []
    if vstack:
      scale = math.ceil(width_bbox/width_graph)
      for image_graph in images_graph:
        image_graph.load(scale=scale)
        images_graph_resize.append(image_graph.resize((width_bbox, round(width_bbox*height_graph/width_graph))))
    else:  # hstack
      scale = math.ceil(height_bbox/height_graph)
      for image_graph in images_graph:
        image_graph.load(scale=scale)
        images_graph_resize.append(image_graph.resize((round(height_bbox*width_graph/height_graph), height_bbox)))
    images_graph = images_graph_resize

    images = []
    for image_bbox, image_graph in zip(images_bbox, images_graph):
      if vstack:
        image = Image.new('RGB', (image_bbox.width, image_bbox.height+image_graph.height))
        image.paste(image_bbox, (0, 0))
        image.paste(image_graph, (0, image_bbox.height))
      else:  # hstack
        image = Image.new('RGB', (image_bbox.width+image_graph.width, image_bbox.height))
        image.paste(image_bbox, (0, 0))
        image.paste(image_graph, (image_bbox.width, 0))
      images.append(image)

    image = images[0]
    image.save(f'./figures/{id_sact}.gif', format='GIF', append_images=images[1:], save_all=True, duration=250, loop=0)
