import matplotlib.pyplot as plt
from pprint import pprint
import seaborn as sns

from moma_api import MOMA


def main():
  dir_moma = '/home/alan/ssd/moma'
  moma = MOMA(dir_moma)

  stats_overall, stats_per_class = moma.get_stats()
  pprint(stats_overall, sort_dicts=False)

  for key in stats_per_class:
    print(key)
    counts = stats_per_class[key]['counts']
    cnames = stats_per_class[key]['class_names']
    assert len(counts) == len(cnames), f'{key}: {len(counts)} vs {len(cnames)}'

    sns.set(style='darkgrid')
    width = max(20, int(0.25*len(counts)))
    height = int(0.5*width)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.barplot(x=cnames, y=counts, ci=None, ax=ax, log=True, color='seagreen')
    ax.set(xlabel='class', ylabel='count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'./figures/{key}.png')


if __name__ == '__main__':
  main()
