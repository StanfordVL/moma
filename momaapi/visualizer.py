from collections import defaultdict
from distinctipy import distinctipy
import glob
import math
from matplotlib import font_manager
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from pprint import pprint
import pygraphviz as pgv
import seaborn as sns
import shutil
from torchvision import io

from .utils import get_stats


class AnnVisualizer:
  def __init__(self, moma, dir_vis):
    self.moma = moma
    self.dir_vis = dir_vis

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
    draw = ImageDraw.Draw(image)
    width_line = int(max(image.size)*0.003)

    font = font_manager.FontProperties(family='sans-serif', stretch='extra-condensed', weight='light')
    path_font = font_manager.findfont(font)
    font = ImageFont.truetype(path_font, int(max(image.size)*0.02))

    for entity in entities:
      y1, x1, y2, x2 = entity.bbox.y1, entity.bbox.x1, entity.bbox.y2, entity.bbox.x2
      width_text, height_text = font.getsize(entity.cname)
      draw.rectangle(((x1, y1), (x2, y2)), width=width_line, outline=palette[entity.id][0])
      draw.rectangle(((x1, y1), (x1+width_text+2*width_line, y1+height_text+2*width_line)), fill=palette[entity.id][0])
      draw.text((x1+width_line, y1+width_line), entity.cname, fill=palette[entity.id][1], font=font)

    return image

  def draw_image(self, ann_hoi, palette):
    path_image = self.moma.get_paths(ids_hoi=[ann_hoi.id])[0]
    image = io.read_image(path_image).permute(1, 2, 0).numpy()
    image = Image.fromarray(image).convert('RGB')
    image = self.draw_entities(image, ann_hoi.actors, palette)
    image = self.draw_entities(image, ann_hoi.objects, palette)
    return image

  def show_hoi(self, id_hoi, vstack=True):
    if os.path.isfile(os.path.join(self.dir_vis, f'hoi/{id_hoi}.png')):
      return

    os.makedirs(os.path.join(self.dir_vis, 'hoi'), exist_ok=True)

    ann_hoi = self.moma.get_anns_hoi(ids_hoi=[id_hoi])[0]
    palette = self.get_palette(ann_hoi.ids_actor+ann_hoi.ids_object)

    """ bbox """
    image = self.draw_image(ann_hoi, palette)
    path_bbox = os.path.join(self.dir_vis, f'hoi/bbox_{id_hoi}.png')
    image.save(path_bbox)

    """ graph """
    G = pgv.AGraph(directed=True, strict=True)

    for actor in ann_hoi.actors:
      G.add_node(actor.id, label=actor.id, xlabel=actor.cname, fontcolor='steelblue', color='steelblue', shape='circle')
    for object in ann_hoi.objects:
      G.add_node(object.id, label=object.id, xlabel=object.cname, fontcolor='salmon3', color='salmon3', shape='circle')
    for description in ann_hoi.tas+ann_hoi.rels:
      G.add_edge((description.id_src, description.id_trg), label=description.cname,
                 color='slategray', fontcolor='slategray', fontsize='10', len=2)
    for description in ann_hoi.ias+ann_hoi.atts:
      G.add_edge((description.id_src, description.id_src), label=description.cname,
                 color='slategray', fontcolor='slategray', fontsize='10', len=2)

    G.layout('neato')
    G.node_attr['fontname'] = 'Arial'
    G.edge_attr['fontname'] = 'Arial'
    path_graph = os.path.join(self.dir_vis, f'hoi/graph_{id_hoi}.eps')
    G.draw(path_graph)

    """ save """
    image_bbox = Image.open(path_bbox)
    image_graph = Image.open(path_graph)

    width_bbox, height_bbox = image_bbox.size
    width_graph, height_graph = image_graph.size

    if vstack:
      scale = math.ceil(width_bbox/width_graph)
      image_graph.load(scale=scale)
      image_graph = image_graph.resize((width_bbox, round(width_bbox*height_graph/width_graph)))

      image = Image.new('RGB', (image_bbox.width, image_bbox.height+image_graph.height))
      image.paste(image_bbox, (0, 0))
      image.paste(image_graph, (0, image_bbox.height))

    else:  # hstack
      scale = math.ceil(height_bbox/height_graph)
      image_graph.load(scale=scale)
      image_graph = image_graph.resize((round(height_bbox*width_graph/height_graph), height_bbox))

      image = Image.new('RGB', (image_bbox.width+image_graph.width, image_bbox.height))
      image.paste(image_bbox, (0, 0))
      image.paste(image_graph, (image_bbox.width, 0))

    image.save(os.path.join(self.dir_vis, f'hoi/{id_hoi}.png'))
    os.remove(path_bbox)
    os.remove(path_graph)

  def show_sact(self, id_sact, vstack=True):
    if os.path.isfile(os.path.join(self.dir_vis, f'sact/{id_sact}.gif')):
      return

    os.makedirs(os.path.join(self.dir_vis, 'sact', id_sact), exist_ok=True)

    ann_sact = self.moma.get_anns_sact(ids_sact=[id_sact])[0]
    ids_hoi = self.moma.get_ids_hoi(ids_sact=[id_sact])
    anns_hoi = self.moma.get_anns_hoi(ids_hoi=ids_hoi)
    palette = self.get_palette(ann_sact.ids_actor+ann_sact.ids_object)

    """ bbox """
    for i, id_hoi in enumerate(ids_hoi):
      ann_hoi = self.moma.get_anns_hoi(ids_hoi=[id_hoi])[0]
      image = self.draw_image(ann_hoi, palette)
      image.save(os.path.join(self.dir_vis, f'sact/{id_sact}/bbox_{str(i).zfill(2)}.png'))

    """ graph """
    # get node & edge positions
    info_nodes = []
    for id_entity in ann_sact.ids_actor+ann_sact.ids_object:
      cname_entity = ann_sact.get_cname_entity(id_entity)
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
    G.node_attr['fontname'] = 'Arial'
    G.edge_attr['fontname'] = 'Arial'

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

      G.draw(os.path.join(self.dir_vis, f'sact/{id_sact}/graph_{str(i).zfill(2)}.eps'))
      G.remove_nodes_from([info_node[0] for info_node in info_nodes])

    """ save """
    paths_bbox = sorted(glob.glob(os.path.join(self.dir_vis, f'sact/{id_sact}/bbox_*.png')))
    paths_graph = sorted(glob.glob(os.path.join(self.dir_vis, f'sact/{id_sact}/graph_*.eps')))

    images_bbox = [Image.open(path_bbox) for path_bbox in paths_bbox]
    images_graph = [Image.open(path_graph) for path_graph in paths_graph]

    assert all(image_bbox.size == images_bbox[0].size for image_bbox in images_bbox)
    assert all(image_graph.size == images_graph[0].size for image_graph in images_graph)

    width_bbox, height_bbox = images_bbox[0].size
    width_graph, height_graph = images_graph[0].size

    images = []
    if vstack:
      scale = math.ceil(width_bbox/width_graph)
      for image_bbox, image_graph in zip(images_bbox, images_graph):
        image_graph.load(scale=scale)
        image_graph = image_graph.resize((width_bbox, round(width_bbox*height_graph/width_graph)))
        image = Image.new('RGB', (image_bbox.width, image_bbox.height+image_graph.height))
        image.paste(image_bbox, (0, 0))
        image.paste(image_graph, (0, image_bbox.height))
        images.append(image)

    else:  # hstack
      scale = math.ceil(height_bbox/height_graph)
      for image_bbox, image_graph in zip(images_bbox, images_graph):
        image_graph.load(scale=scale)
        image_graph = image_graph.resize((round(height_bbox*width_graph/height_graph), height_bbox))
        image = Image.new('RGB', (image_bbox.width+image_graph.width, image_bbox.height))
        image.paste(image_bbox, (0, 0))
        image.paste(image_graph, (image_bbox.width, 0))
        images.append(image)

    image = images[0]
    image.save(os.path.join(self.dir_vis, f'sact/{id_sact}.gif'), format='GIF', append_images=images[1:], save_all=True,
               duration=250, loop=0)
    shutil.rmtree(os.path.join(self.dir_vis, 'sact', id_sact))


class StatVisualizer:
  def __init__(self, moma, dir_vis):
    self.moma = moma
    self.dir_vis = dir_vis

  def show(self, with_split):
    os.makedirs(os.path.join(self.dir_vis, 'stats'), exist_ok=True)

    if with_split:
      stats_overall_train, stats_per_class_train = get_stats(self.moma, 'train')
      stats_overall_val, stats_per_class_val = get_stats(self.moma, 'val')
      stats_per_class = {}
      for key in stats_per_class_train:
        stats_per_class[key] = {
          'counts': stats_per_class_train[key]['counts']+stats_per_class_val[key]['counts'],
          'class_names': stats_per_class_train[key]['class_names']+stats_per_class_val[key]['class_names'],
          'hue': ['train']*len(stats_per_class_train[key]['counts'])+['val']*len(stats_per_class_val[key]['counts'])
        }
      pprint(stats_overall_train, sort_dicts=False)
      pprint(stats_overall_val, sort_dicts=False)

    else:
      stats_overall, stats_per_class = get_stats(self.moma)
      pprint(stats_overall, sort_dicts=False)

    for key in stats_per_class:
      counts = stats_per_class[key]['counts']
      cnames = stats_per_class[key]['class_names']
      hue = stats_per_class[key]['hue'] if with_split else None
      fname = f"{key}{'_split' if with_split else ''}.png"
      assert len(counts) == len(cnames), f'{key}: {len(counts)} vs {len(cnames)}'

      sns.set(style='darkgrid')
      width = max(20, int(0.25*len(counts)))
      height = int(0.5*width)
      fig, ax = plt.subplots(figsize=(width, height))
      sns.barplot(x=cnames, y=counts, hue=hue, ci=None, ax=ax, log=True,
                  color=None if with_split else 'seagreen', palette='dark' if with_split else None)
      ax.set(xlabel='class', ylabel='count')
      ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
      plt.tight_layout()
      plt.savefig(os.path.join(self.dir_vis, 'stats', fname))
