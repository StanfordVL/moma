from collections import defaultdict
import itertools
import json
import os
import random


class SplitGenerator:
  def __init__(self, dir_moma):
    self.dir_moma = dir_moma

    with open(os.path.join(self.dir_moma, 'anns/anns.json'), 'r') as f:
      anns = json.load(f)

    ids_act = [ann['activity']['id'] for ann in anns]
    cnames_act = [ann['activity']['class_name'] for ann in anns]

    self.ids_act = sorted(ids_act)
    self.cname_to_ids = defaultdict(list)
    for id_act, cname_act in zip(ids_act, cnames_act):
      self.cname_to_ids[cname_act].append(id_act)

  def generate_regular_splits(self, ratio):
    # need all ids_act
    ids_act = random.sample(self.ids_act, len(self.ids_act))

    size_train = round(len(ids_act)*ratio)
    size_val = round(size_train*(1-ratio))
    size_train = size_train-size_val

    ids_act_train = ids_act[:size_train]
    ids_act_val = ids_act[size_train:(size_train+size_val)]
    ids_act_test = ids_act[(size_train+size_val):]

    path_split = os.path.join(self.dir_moma, 'anns/split_std.json')
    with open(path_split, 'w') as f:
      json.dump({'train': ids_act_train, 'val': ids_act_val, 'test': ids_act_test}, f, indent=2, sort_keys=False)

  def generate_few_shot_splits(self):
    with open(os.path.join(self.dir_moma, 'anns/taxonomy/few_shot.json'), 'r') as f:
      split_to_cnames = json.load(f)

    # need ids_act given cnames
    path_split = os.path.join(self.dir_moma, 'anns/split_fs.json')
    output = {}
    for split, cnames in split_to_cnames.items():
      output[split] = itertools.chain.from_iterable([self.cname_to_ids[cname] for cname in split_to_cnames[split]])
    with open(path_split, 'w') as f:
      json.dump(output, f, indent=2, sort_keys=False)
