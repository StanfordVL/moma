import json
import os

from .data import *
from .utils import *


class AnnMerger:
  def __init__(self, dir_moma, ann_phase1, ann_phase2):
    self.dir_moma = dir_moma
    self.ann_phase1 = ann_phase1
    self.ann_phase2 = ann_phase2
    self.anns = self.__merge()

  def dump(self):
    with open(os.path.join(self.dir_moma, 'anns/anns.json'), 'w') as f:
      json.dump(self.anns, f, ensure_ascii=False, indent=2, sort_keys=False)

    with open(os.path.join(self.dir_moma, 'anns/anns_toy.json'), 'w') as f:
      json.dump(self.anns[:10], f, ensure_ascii=False, indent=2, sort_keys=False)

  def __get_anns(self):
    for id_act in self.ann_phase1.id_act_to_ids_sact:
      ids_sact_phase1 = self.ann_phase1.id_act_to_ids_sact[id_act]
      ids_sact_phase2 = self.ann_phase2.ids_sact
      ids_sact = list(set(ids_sact_phase1)&set(ids_sact_phase2))

      if len(ids_sact) == 0:
        continue

      ann_act = self.ann_phase1.anns_act[id_act]
      anns_sact_phase1 = [ann_sact for ann_sact in ann_act['subactivity']
                          if self.ann_phase1.get_id_sact(ann_sact) in ids_sact]
      anns_sact_phase1 = sorted(anns_sact_phase1, key=lambda x: hms2s(x['start']))
      ids_sact = [self.ann_phase1.get_id_sact(ann_sact) for ann_sact in anns_sact_phase1]
      anns_sact_phase2 = [self.ann_phase2.anns_sact_raw[id_sact] for id_sact in ids_sact]

      yield id_act, ann_act, ids_sact, anns_sact_phase1, anns_sact_phase2

  def __merge(self):
    cn2en = self.ann_phase1.cn2en

    anns = []
    for id_act, ann_act, ids_sact, anns_sact_phase1, anns_sact_phase2 in self.__get_anns():
      assert len(ids_sact) == len(anns_sact_phase1) == len(anns_sact_phase2)
      sact = []
      for id_sact, ann_sact_phase1, ann_sact_phase2 in zip(ids_sact, anns_sact_phase1, anns_sact_phase2):
        hoi = []
        record = ann_sact_phase2[0]['task']['task_params']['record']
        id_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
        ids_hoi = sorted(id_hoi_to_timestamp.keys(), key=int)
        assert len(ids_hoi) == len(ann_sact_phase2)
        for id_hoi, ann_hoi in zip(ids_hoi, ann_sact_phase2):
          id_ent_remove = []

          anns_actor = ann_hoi['task_result']['annotations'][0]['slotsChildren']
          anns_actor = [Ent(ann_actor, cn2en) for ann_actor in anns_actor]
          actor = []
          for ann_actor in anns_actor:
            if ann_actor.cname == 'REMOVE':
              id_ent_remove.append(ann_actor.id)
            else:
              actor.append({
                'id': ann_actor.id,
                'class_name': ann_actor.cname,
                'bbox': [ann_actor.bbox.x, ann_actor.bbox.y, ann_actor.bbox.width, ann_actor.bbox.height]
              })
          
          anns_object = ann_hoi['task_result']['annotations'][1]['slotsChildren']
          anns_object = [Ent(ann_object, cn2en) for ann_object in anns_object]
          object = []
          for ann_object in anns_object:
            if ann_object.cname == 'REMOVE':
              id_ent_remove.append(ann_object.id)
            else:
              object.append({
                'id': ann_object.id,
                'class_name': ann_object.cname,
                'bbox': [ann_object.bbox.x, ann_object.bbox.y, ann_object.bbox.width, ann_object.bbox.height]
              })

          anns_binary = ann_hoi['task_result']['annotations'][2]['slotsChildren']
          anns_binary = [Predicate(ann_binary, cn2en) for ann_binary in anns_binary]
          rel, ta = [], []
          for ann_binary in anns_binary:
            for id_src, id_trg, cname in ann_binary.breakdown():
              if not (id_src in id_ent_remove or id_trg in id_ent_remove or cname == 'REMOVE'):
                if ann_binary.cname in [x[0] for x in self.ann_phase2.taxonomy_rel]:
                  rel.append({
                    'class_name': cname,
                    'source_id': id_src,
                    'target_id': id_trg
                  })
                elif ann_binary.cname in [x[0] for x in self.ann_phase2.taxonomy_ta]:
                  ta.append({
                    'class_name': cname,
                    'source_id': id_src,
                    'target_id': id_trg
                  })
                else:
                  assert False, ann_binary.cname

          anns_unary = ann_hoi['task_result']['annotations'][3]['slotsChildren']
          anns_unary = [Predicate(ann_unary, cn2en) for ann_unary in anns_unary]
          att, ia = [], []
          for ann_unary in anns_unary:
            for id_src, cname in ann_unary.breakdown():
              if not (id_src in id_ent_remove or cname == 'REMOVE'):
                if ann_unary.cname in [x[0] for x in self.ann_phase2.taxonomy_att]:
                  att.append({
                    'class_name': cname,
                    'source_id': id_src
                  })
                elif ann_unary.cname in [x[0] for x in self.ann_phase2.taxonomy_ia]:
                  ia.append({
                    'class_name': cname,
                    'source_id': id_src
                  })
                else:
                  assert False, ann_unary.cname

          record = ann_hoi['task']['task_params']['record']
          hoi.append({
            'id': f"{id_sact.zfill(5)}_{record['attachment'].split('_')[-1][:-4].split('/')[1]}",
            'time': hms2s(ann_sact_phase1['start'])+float(id_hoi_to_timestamp[id_hoi]),
            'actors': actor,
            'objects': object,
            'attributes': att,
            'relationships': rel,
            'intransitive_actions': ia,
            'transitive_actions': ta
          })

        sact.append({
          'id': id_sact.zfill(5),
          'class_name': cn2en[ann_sact_phase1['filename'].split('_')[0]],
          'start_time': hms2s(ann_sact_phase1['start']),
          'end_time': hms2s(ann_sact_phase1['end']),
          'higher_order_interactions': hoi
        })

      act = {
        'id': id_act,
        'class_name': cn2en[ann_act['subactivity'][0]['orig_vid'].split('_')[0]],
        'start_time': hms2s(ann_act['crop_start']),
        'end_time': hms2s(ann_act['crop_end']),
        'sub_activities': sact
      }

      metadata = self.ann_phase1.metadata[id_act]
      anns.append({
        'file_name': anns_sact_phase1[0]['orig_vid'].split('_', 1)[1],
        'num_frames': int(metadata['nb_frames']),
        'width': int(metadata['width']),
        'height': int(metadata['height']),
        'duration': float(metadata['duration']),
        'activity': act
      })

    return anns
