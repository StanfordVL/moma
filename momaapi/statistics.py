import numpy as np


class Statistics(dict):
  def __init__(self, moma):
    super().__init__()
    self.statistics = self.get_statistics(moma)

  def get_statistics(self, moma):
    statistics = {'all': self.get_statistic(moma)}

    for split in moma.split_to_ids_act:
      statistics[split] = self.get_statistic(moma, split)

    return statistics

  @staticmethod
  def get_duration(anns):
    duration_total = sum(ann.end-ann.start for ann in anns)
    duration_avg = duration_total/len(anns)
    duration_min = min(ann.end-ann.start for ann in anns)
    duration_max = max(ann.end-ann.start for ann in anns)
    return duration_total, duration_avg, duration_min, duration_max

  def get_statistic(self, moma, split=None):
    if split is None:
      metadata = moma.metadata.values()
      anns_act = moma.id_act_to_ann_act.values()
      anns_sact = moma.id_sact_to_ann_sact.values()
      anns_hoi = moma.id_hoi_to_ann_hoi.values()
    else:
      assert split in moma.split_to_ids_act
      ids_act = moma.split_to_ids_act[split]
      metadata = moma.get_metadata(ids_act=ids_act)
      anns_act = moma.get_anns_act(ids_act=ids_act)
      ids_sact = moma.get_ids_sact(ids_act=ids_act)
      anns_sact = moma.get_anns_sact(ids_sact=ids_sact)
      ids_hoi = moma.get_ids_hoi(ids_act=ids_act)
      anns_hoi = moma.get_anns_hoi(ids_hoi=ids_hoi)

    num_acts = len(anns_act)
    num_classes_act = len(moma.taxonomy['act'])
    num_sacts = len(anns_sact)
    num_classes_sact = len(moma.taxonomy['sact'])
    num_hois = len(anns_hoi)

    num_actors_image = sum([len(ann_hoi.actors) for ann_hoi in anns_hoi])
    num_actors_video = sum([len(ann_sact.ids_actor) for ann_sact in anns_sact])
    num_classes_actor = len(moma.taxonomy['actor'])
    num_objects_image = sum([len(ann_hoi.objects) for ann_hoi in anns_hoi])
    num_objects_video = sum([len(ann_sact.ids_object) for ann_sact in anns_sact])
    num_classes_object = len(moma.taxonomy['object'])

    num_ias = sum([len(ann_hoi.ias) for ann_hoi in anns_hoi])
    num_classes_ia = len(moma.taxonomy['ia'])
    num_tas = sum([len(ann_hoi.tas) for ann_hoi in anns_hoi])
    num_classes_ta = len(moma.taxonomy['ta'])
    num_atts = sum([len(ann_hoi.atts) for ann_hoi in anns_hoi])
    num_classes_att = len(moma.taxonomy['att'])
    num_rels = sum([len(ann_hoi.rels) for ann_hoi in anns_hoi])
    num_classes_rel = len(moma.taxonomy['rel'])

    duration_total_raw = sum(metadatum.duration for metadatum in metadata)
    duration_total_act, duration_avg_act, duration_min_act, duration_max_act = self.get_duration(anns_act)
    duration_total_sact, duration_avg_sact, duration_min_sact, duration_max_sact = self.get_duration(anns_sact)

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

    statistic = {
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

    return statistic

  def keys(self):
    return self.statistics.keys()

  def __getitem__(self, item):
    return self.statistics[item]

  def __len__(self):
    return len(self.statistics.keys())

  def __repr__(self):
    return repr(self.statistics)
