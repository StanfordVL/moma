import numpy as np
import random
from scipy.spatial import distance


def get_stats(moma, split=None):
  if split is None:
    anns_act = moma.id_act_to_ann_act.values()
    anns_sact = moma.id_sact_to_ann_sact.values()
    anns_hoi = moma.id_hoi_to_ann_hoi.values()
  elif split == 'train' or split == 'val':
    ids_act = moma.ids_act_train if split == 'train' else moma.ids_act_val
    anns_act = moma.get_anns_act(ids_act=ids_act)
    ids_sact = moma.get_ids_sact(ids_act=ids_act)
    anns_sact = moma.get_anns_sact(ids_sact=ids_sact)
    ids_hoi = moma.get_ids_hoi(ids_act=ids_act)
    anns_hoi = moma.get_anns_hoi(ids_hoi=ids_hoi)
  else:
    assert False

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

  duration_avg_act = sum(ann_act.end-ann_act.start for ann_act in anns_act)/len(anns_act)
  duration_min_act = min(ann_act.end-ann_act.start for ann_act in anns_act)
  duration_max_act = max(ann_act.end-ann_act.start for ann_act in anns_act)
  duration_avg_sact = sum(ann_sact.end-ann_sact.start for ann_sact in anns_sact)/len(anns_act)
  duration_min_sact = min(ann_sact.end-ann_sact.start for ann_sact in anns_sact)
  duration_max_sact = max(ann_sact.end-ann_sact.start for ann_sact in anns_sact)

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

  stats_overall = {
    'activity': {
      'num_instances': num_acts,
      'num_classes': num_classes_act,
      'duration_avg': duration_avg_act,
      'duration_min': duration_min_act,
      'duration_max': duration_max_act
    },
    'sub_activity': {
      'num_instances': num_sacts,
      'num_classes': num_classes_sact,
      'duration_avg': duration_avg_sact,
      'duration_min': duration_min_sact,
      'duration_max': duration_max_sact
    },
    'higher_order_interaction': {
      'num_instances': num_hois,
    },
    'actor': {
      'num_instances_image': num_actors_image,
      'num_instances_video': num_actors_video,
      'num_classes': num_classes_actor
    },
    'object': {
      'num_instances_image': num_objects_image,
      'num_instances_video': num_objects_video,
      'num_classes': num_classes_object
    },
    'intransitive_action': {
      'num_instances': num_ias,
      'num_classes': num_classes_ia
    },
    'transitive_action': {
      'num_instances': num_tas,
      'num_classes': num_classes_ta
    },
    'attribute': {
      'num_instances': num_atts,
      'num_classes': num_classes_att
    },
    'relationship': {
      'num_instances': num_rels,
      'num_classes': num_classes_rel
    },
  }

  stats_per_class = {
    'activity': {
      'counts': bincount_act,
      'class_names': moma.taxonomy['act']
    },
    'sub_activity': {
      'counts': bincount_sact,
      'class_names': moma.taxonomy['sact']
    },
    'actor': {
      'counts': bincount_actor,
      'class_names': moma.taxonomy['actor']
    },
    'object': {
      'counts': bincount_object,
      'class_names': moma.taxonomy['object']
    },
    'intransitive_action': {
      'counts': bincount_ia,
      'class_names': [x[0] for x in moma.taxonomy['ia']]
    },
    'transitive_action': {
      'counts': bincount_ta,
      'class_names': [x[0] for x in moma.taxonomy['ta']]
    },
    'attribute': {
      'counts': bincount_att,
      'class_names': [x[0] for x in moma.taxonomy['att']]
    },
    'relationship': {
      'counts': bincount_rel,
      'class_names': [x[0] for x in moma.taxonomy['rel']]
    }
  }

  return stats_overall, stats_per_class


def get_dist_per_class(stats_per_class_train, stats_per_class_val):
  dists = {}

  for kind in stats_per_class_train:
    counts_train = stats_per_class_train[kind]['counts']
    counts_val = stats_per_class_val[kind]['counts']
    assert len(counts_train) == len(counts_val) == len(stats_per_class_train[kind]['class_names'])
    dist = distance.cosine(counts_train, counts_val)
    dists[f'{kind}_counts'] = dist

  return dists


def get_dist_overall(stats_overall_train, stats_overall_val):
  dists = {}

  ratio_best = stats_overall_train['activity']['num_instances']/ \
               (stats_overall_train['activity']['num_instances']+stats_overall_val['activity']['num_instances'])

  for kind in stats_overall_train:
    for stat in stats_overall_train[kind]:
      if stat == 'num_classes':
        continue

      num_instances_train = stats_overall_train[kind][stat]
      num_instances_val = stats_overall_val[kind][stat]
      ratio = num_instances_train/(num_instances_train+num_instances_val)
      dist = abs(ratio-ratio_best)/ratio_best
      dists[f'{kind}_{stat}'] = dist

  return dists


def split_ids_act(ids_act, ratio_train=0.80):
  random.shuffle(ids_act)

  size_train = round(len(ids_act)*ratio_train)
  ids_act_train = ids_act[:size_train]
  ids_act_val = ids_act[size_train:]

  return ids_act_train, ids_act_val
