import itertools
import json
import jsbeautifier
import numpy as np
import os
import os.path as osp


class Statistics(dict):
  def __init__(self, dir_moma, taxonomy, lookup, reset_cache):
    super().__init__()
    self.statistics = self.__read_statistics(dir_moma, taxonomy, lookup, reset_cache)

  def __read_statistics(self, dir_moma, taxonomy, lookup, reset_cache):
    paradigms = lookup.retrieve('paradigms')
    splits = lookup.retrieve('splits')

    path_statistics = osp.join(dir_moma, 'anns/cache/statistics.json')
    if reset_cache and osp.exists(path_statistics):
      os.remove(path_statistics)

    if osp.exists(path_statistics):
      with open(path_statistics, 'r') as f:
        statistics = json.load(f)
      print('Statistics: load cache')

    else:
      statistics = {'all': self.__get_statistics(taxonomy, lookup)}
      for paradigm, split in itertools.product(paradigms, splits):
        statistics[f'{paradigm}_{split}'] = self.__get_statistics(taxonomy, lookup, paradigm, split)
      with open(path_statistics, 'w') as f:
        options = jsbeautifier.default_options()
        options.indent_size = 4
        f.write(jsbeautifier.beautify(json.dumps(statistics), options))
      print('Statistics: save cache')

    return statistics

  @staticmethod
  def __get_duration(anns):
    duration_total = sum(ann.end-ann.start for ann in anns)
    duration_avg = duration_total/len(anns)
    duration_min = min(ann.end-ann.start for ann in anns)
    duration_max = max(ann.end-ann.start for ann in anns)
    return duration_total, duration_avg, duration_min, duration_max

  def __get_statistics(self, taxonomy, lookup, paradigm=None, split=None):
    # subsample metadata, anns_act, anns_sact, and anns_hoi
    if paradigm is None and split is None:
      metadata = lookup.retrieve('metadata')
      anns_act = lookup.retrieve('anns_act')
      anns_sact = lookup.retrieve('anns_sact')
      anns_hoi = lookup.retrieve('anns_hoi')
    else:
      assert paradigm is not None and split is not None
      ids_act = lookup.retrieve('ids_act', f'{paradigm}_{split}')
      metadata = [lookup.retrieve('metadatum', id_act) for id_act in ids_act]
      anns_act = [lookup.retrieve('ann_act', id_act) for id_act in ids_act]
      ids_sact = list(itertools.chain(*[lookup.trace('ids_sact', id_act=id_act) for id_act in ids_act]))
      anns_sact = [lookup.retrieve('ann_sact', id_sact) for id_sact in ids_sact]
      ids_hoi = list(itertools.chain(*[lookup.trace('ids_hoi', id_sact=id_sact) for id_sact in ids_sact]))
      anns_hoi = [lookup.retrieve('ann_hoi', id_hoi) for id_hoi in ids_hoi]

    # number of classes and instances
    num_acts = len(anns_act)
    num_classes_act = len(set([ann_act.cid for ann_act in anns_act]))
    num_sacts = len(anns_sact)
    num_classes_sact = len(set([ann_sact.cid for ann_sact in anns_sact]))
    num_hois = len(anns_hoi)

    num_actors_image = sum([len(ann_hoi.actors) for ann_hoi in anns_hoi])
    num_actors_video = sum([len(ann_sact.ids_actor) for ann_sact in anns_sact])
    num_classes_actor = len(set(itertools.chain(*[ann_sact.cids_actor for ann_sact in anns_sact])))
    num_objects_image = sum([len(ann_hoi.objects) for ann_hoi in anns_hoi])
    num_objects_video = sum([len(ann_sact.ids_object) for ann_sact in anns_sact])
    num_classes_object = len(set(itertools.chain(*[ann_sact.cids_object for ann_sact in anns_sact])))

    num_ias = sum([len(ann_hoi.ias) for ann_hoi in anns_hoi])
    num_classes_ia = len(set([ia.cid for ann_hoi in anns_hoi for ia in ann_hoi.ias]))
    num_tas = sum([len(ann_hoi.tas) for ann_hoi in anns_hoi])
    num_classes_ta = len(set([ta.cid for ann_hoi in anns_hoi for ta in ann_hoi.tas]))
    num_atts = sum([len(ann_hoi.atts) for ann_hoi in anns_hoi])
    num_classes_att = len(set([att.cid for ann_hoi in anns_hoi for att in ann_hoi.atts]))
    num_rels = sum([len(ann_hoi.rels) for ann_hoi in anns_hoi])
    num_classes_rel = len(set([rel.cid for ann_hoi in anns_hoi for rel in ann_hoi.rels]))

    # if paradigm != 'few-shot':
    #   print(paradigm, split)
    #   assert num_classes_actor == len(taxonomy['actor']), f"{num_classes_actor} vs {len(taxonomy['actor'])}"
    #   assert num_classes_object == len(taxonomy['object']), f"{num_classes_object} vs {len(taxonomy['object'])}"
    #   assert num_classes_ia == len(taxonomy['ia']), f"{num_classes_ia} vs {len(taxonomy['ia'])}"
    #   assert num_classes_ta == len(taxonomy['ta']), f"{num_classes_ta} vs {len(taxonomy['ta'])}"
    #   assert num_classes_att == len(taxonomy['att']), f"{num_classes_att} vs {len(taxonomy['att'])}"
    #   assert num_classes_rel == len(taxonomy['rel']), f"{num_classes_rel} vs {len(taxonomy['rel'])}"

    # durations
    duration_total_raw = sum(metadatum.duration for metadatum in metadata)
    duration_total_act, duration_avg_act, duration_min_act, duration_max_act = self.__get_duration(anns_act)
    duration_total_sact, duration_avg_sact, duration_min_sact, duration_max_sact = self.__get_duration(anns_sact)

    # class distributions
    bincount_act = np.bincount([ann_act.cid for ann_act in anns_act], minlength=num_classes_act).tolist()
    bincount_sact = np.bincount([ann_sact.cid for ann_sact in anns_sact], minlength=num_classes_sact).tolist()
    bincount_actor, bincount_object, bincount_ia, bincount_ta, bincount_att, bincount_rel = [], [], [], [], [], []
    for ann_hoi in anns_hoi:
      bincount_actor += [actor.cid for actor in ann_hoi.actors]
      bincount_object += [object.cid for object in ann_hoi.objects]
      bincount_ia += [ia.cid for ia in ann_hoi.ias]
      bincount_ta += [ta.cid for ta in ann_hoi.tas]
      bincount_att += [att.cid for att in ann_hoi.atts]
      bincount_rel += [rel.cid for rel in ann_hoi.rels]
    bincount_actor = np.bincount(bincount_actor, minlength=num_classes_actor).tolist()
    bincount_object = np.bincount(bincount_object, minlength=num_classes_object).tolist()
    bincount_ia = np.bincount(bincount_ia, minlength=num_classes_ia).tolist()
    bincount_ta = np.bincount(bincount_ta, minlength=num_classes_ta).tolist()
    bincount_att = np.bincount(bincount_att, minlength=num_classes_att).tolist()
    bincount_rel = np.bincount(bincount_rel, minlength=num_classes_rel).tolist()

    # curate statistics
    statistics = {
      'raw': {
        'duration_total': duration_total_raw
      },
      'act': {
        'num_instances': num_acts,
        'num_classes': num_classes_act,
        'duration_avg': duration_avg_act,
        'duration_min': duration_min_act,
        'duration_max': duration_max_act,
        'duration_total': duration_total_act,
        'distribution': bincount_act
      },
      'sact': {
        'num_instances': num_sacts,
        'num_classes': num_classes_sact,
        'duration_avg': duration_avg_sact,
        'duration_min': duration_min_sact,
        'duration_max': duration_max_sact,
        'duration_total': duration_total_sact,
        'distribution': bincount_sact
      },
      'hoi': {
        'num_instances': num_hois,
      },
      'actor': {
        'num_instances_image': num_actors_image,
        'num_instances_video': num_actors_video,
        'num_classes': num_classes_actor,
        'distribution': bincount_actor
      },
      'object': {
        'num_instances_image': num_objects_image,
        'num_instances_video': num_objects_video,
        'num_classes': num_classes_object,
        'distribution': bincount_object
      },
      'ia': {
        'num_instances': num_ias,
        'num_classes': num_classes_ia,
        'distribution': bincount_ia
      },
      'ta': {
        'num_instances': num_tas,
        'num_classes': num_classes_ta,
        'distribution': bincount_ta
      },
      'att': {
        'num_instances': num_atts,
        'num_classes': num_classes_att,
        'distribution': bincount_att
      },
      'rel': {
        'num_instances': num_rels,
        'num_classes': num_classes_rel,
        'distribution': bincount_rel
      },
    }

    return statistics

  def keys(self):
    return self.statistics.keys()

  def values(self):
    raise NotImplementedError

  def items(self):
    raise NotImplementedError

  def __getitem__(self, key):
    return self.statistics[key]

  def __len__(self):
    return len(self.statistics.keys())

  def __repr__(self):
    return repr(self.statistics)
