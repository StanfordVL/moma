import matplotlib.pyplot as plt
import os
from pprint import pprint
import seaborn as sns

from .stats import get_stats


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
