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

  def __get_anns(self):
    for iid_act in self.ann_phase1.iid_act_to_iids_sact:
      iids_sact_phase1 = self.ann_phase1.iid_act_to_iids_sact[iid_act]
      iids_sact_phase2 = self.ann_phase2.iids_sact
      iids_sact = list(set(iids_sact_phase1)&set(iids_sact_phase2))

      if len(iids_sact) == 0:
        continue

      ann_act = self.ann_phase1.anns_act[iid_act]
      anns_sact_phase1 = [ann_sact for ann_sact in ann_act['subactivity']
                          if self.ann_phase1.get_iid_sact(ann_sact) in iids_sact]
      anns_sact_phase1 = sorted(anns_sact_phase1, key=lambda x: hms2s(x['start']))
      iids_sact = [self.ann_phase1.get_iid_sact(ann_sact) for ann_sact in anns_sact_phase1]
      anns_sact_phase2 = [self.ann_phase2.anns_sact_raw[iid_sact] for iid_sact in iids_sact]

      yield iid_act, ann_act, iids_sact, anns_sact_phase1, anns_sact_phase2

  def __merge(self):
    """ annotation syntax
    [
      {
        file_name: str,
        num_frames: int,
        width: int,
        height: int,
        duration: float,

        # an activity
        activity: {
          key: str,
          class_name: str,
          start_time: int,
          end_time: int,
          sub_activities: [
            # a sub-activity
            {
              key: str,
              class_name: str,
              start_time: int,
              end_time: int,
              higher_order_interactions: [
                # a higher-order interaction
                {
                  key: str,
                  time: int,
                  actors: [
                    # an actor
                    {
                      instance_id,
                      class_name: str,
                      bbox: [x, y, width, height]
                    },
                    ...
                  ],
                  objects: [
                    # an object
                    {
                      instance_id: str,
                      class_name: str,
                      bbox: [x, y, width, height]
                    },
                    ...
                  ],
                  relationships: [
                    # a relationship
                    {
                      class_name: str,
                      source_instance_id: str,
                      target_instance_id: str
                    },
                    ...
                  ],
                  attributes: [
                    # an attribute
                    {
                      class_name: str,
                      source_instance_id: str
                    },
                    ...
                  ],
                  transitive_actions: [
                    # a transitive action
                    {
                      class_name: str,
                      source_instance_id: str,
                      target_instance_id: str
                    },
                    ...
                  ],
                  intransitive_actions: [
                    # an intransitive action
                    {
                      class_name: str,
                      source_instance_id: str
                    },
                    ...
                  ]
                }
              ]
            },
            ...
          ]
        }
      },
      ...
    ]
    """

    anns = []
    for iid_act, ann_act, iids_sact, anns_sact_phase1, anns_sact_phase2 in self.__get_anns():
      assert len(iids_sact) == len(anns_sact_phase1) == len(anns_sact_phase2)
      sact = []
      for iid_sact, ann_sact_phase1, ann_sact_phase2 in zip(iids_sact, anns_sact_phase1, anns_sact_phase2):
        hoi = []
        record = ann_sact_phase2[0]['task']['task_params']['record']
        iid_hoi_to_timestamp = record['metadata']['additionalInfo']['framesTimestamp']
        iids_hoi = sorted(iid_hoi_to_timestamp.keys(), key=int)
        assert len(iids_hoi) == len(ann_sact_phase2)
        for iid_hoi, ann_hoi in zip(iids_hoi, ann_sact_phase2):
          anns_actor = ann_hoi['task_result']['annotations'][0]['slotsChildren']
          anns_actor = [Entity(ann_actor, self.ann_phase2.cn2en) for ann_actor in anns_actor]
          actor = []
          for ann_actor in anns_actor:
            actor.append({
              'instance_id': ann_actor.iid,
              'class_name': ann_actor.cname,
              'bbox': [ann_actor.bbox.x, ann_actor.bbox.y, ann_actor.bbox.width, ann_actor.bbox.height]
            })
          
          anns_object = ann_hoi['task_result']['annotations'][1]['slotsChildren']
          anns_object = [Entity(ann_object, self.ann_phase2.cn2en) for ann_object in anns_object]
          object = []
          for ann_object in anns_object:
            object.append({
              'instance_id': ann_object.iid,
              'class_name': ann_object.cname,
              'bbox': [ann_object.bbox.x, ann_object.bbox.y, ann_object.bbox.width, ann_object.bbox.height]
            })

          anns_binary = ann_hoi['task_result']['annotations'][2]['slotsChildren']
          anns_binary = [Description(ann_binary, self.ann_phase2.cn2en) for ann_binary in anns_binary]
          rel, ta = [], []
          for ann_binary in anns_binary:
            if ann_binary.cname in [x[0] for x in self.ann_phase2.taxonomy_rel]:
              for iid_src, iid_trg, cname in ann_binary.breakdown():
                rel.append({
                  'class_name': cname,
                  'source_instance_id': iid_src,
                  'target_instance_id': iid_trg,
                })
            elif ann_binary.cname in [x[0] for x in self.ann_phase2.taxonomy_ta]:
              for iid_src, iid_trg, cname in ann_binary.breakdown():
                ta.append({
                  'class_name': cname,
                  'source_instance_id': iid_src,
                  'target_instance_id': iid_trg,
                })
            else:
              assert False, ann_binary.cname

          anns_unary = ann_hoi['task_result']['annotations'][3]['slotsChildren']
          anns_unary = [Description(ann_unary, self.ann_phase2.cn2en) for ann_unary in anns_unary]
          att, ia = [], []
          for ann_unary in anns_unary:
            if ann_unary.cname in [x[0] for x in self.ann_phase2.taxonomy_att]:
              for iid_src, cname in ann_unary.breakdown():
                att.append({
                  'class_name': cname,
                  'source_instance_id': iid_src
                })
            elif ann_unary.cname in [x[0] for x in self.ann_phase2.taxonomy_ia]:
              for iid_src, cname in ann_unary.breakdown():
                ia.append({
                  'class_name': cname,
                  'source_instance_id': iid_src
                })
            else:
              assert False, ann_unary.cname

          record = ann_hoi['task']['task_params']['record']
          hoi.append({
            'key': f"{iid_sact}_{record['attachment'].split('_')[-1][:-4].split('/')[1]}",
            'time': hms2s(ann_sact_phase1['start'])+float(iid_hoi_to_timestamp[iid_hoi]),
            'actors': actor,
            'objects': object,
            'attributes': att,
            'relationships': rel,
            'intransitive_actions': ia,
            'transitive_actions': ta
          })

        sact.append({
          'key': iid_sact,
          'class_name': ann_sact_phase1['class'],
          'start_time': hms2s(ann_sact_phase1['start']),
          'end_time': hms2s(ann_sact_phase1['end']),
          'higher_order_interactions': hoi
        })

      act = {
        'key': iid_act,
        'class_name': ann_act['class'],
        'start_time': hms2s(ann_act['crop_start']),
        'end_time': hms2s(ann_act['crop_end']),
        'sub_activities': sact
      }

      metadata = self.ann_phase1.metadata[iid_act]
      anns.append({
        'file_name': anns_sact_phase1[0]['orig_vid'].split('_', 1)[1],
        'num_frames': int(metadata['nb_frames']),
        'width': int(metadata['width']),
        'height': int(metadata['height']),
        'duration': float(metadata['duration']),
        'activity': act
      })

    return anns
