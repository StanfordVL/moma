from .data import *
from .utils import *


class AnnMerger:
  def __init__(self, ann_phase1, ann_phase2):
    self.ann_phase1 = ann_phase1
    self.ann_phase2 = ann_phase2

  def __merge(self):
    act = {}
    for iid_act, iids_sact in self.ann_phase1.iid_act_to_iids_sact.items():
      ann_act = self.ann_phase1.anns_act[iid_act]
      iids_sact = [iid_sact for iid_sact in iids_sact
                   if iid_sact in self.ann_phase1.iid_act_to_iids_sact[iid_act]
                   and iid_sact in self.ann_phase2.iids_sact]
      anns_sact = [ann_sact for ann_sact in ann_act['subactivity']
                   if self.ann_phase1.get_iid_sact(ann_sact) in iids_sact]
      anns_sact = sorted(anns_sact, key=lambda x: hms2s(x['start']))
      assert len(iids_sact) == len(anns_sact)

      sact = {}
      for ann_sact1 in anns_sact:
        iid_sact = self.ann_phase1.get_iid_sact(ann_sact1)
        ann_sact2 = self.ann_phase2.anns_sact_raw[iid_sact]

        hoi = {}
        for ann_hoi in ann_sact2:
          anns_actor = ann_hoi['task_result']['annotations'][0]['slotsChildren']
          anns_actor = [Entity(ann_actor, self.ann_phase2.cn2en) for ann_actor in anns_actor]
          actor = {}
          for ann_actor in anns_actor:
            actor[ann_actor.iid] = {
              'class': ann_actor.cname,
              'bbox': [ann_actor.bbox.x, ann_actor.bbox.y, ann_actor.bbox.width, ann_actor.bbox.height]
            }
          
          anns_object = ann_hoi['task_result']['annotations'][1]['slotsChildren']
          anns_object = [Entity(ann_object, self.ann_phase2.cn2en) for ann_object in anns_object]
          object = {}
          for ann_object in anns_object:
            object[ann_object.iid] = {
              'class': ann_object.cname,
              'bbox': [ann_object.bbox.x, ann_object.bbox.y, ann_object.bbox.width, ann_object.bbox.height]
            }

          anns_binary = ann_hoi['task_result']['annotations'][2]['slotsChildren']
          anns_binary = [Description(ann_binary, self.ann_phase2.cn2en) for ann_binary in anns_binary]
          rel, ta = [], []
          for ann_binary in anns_binary:
            if ann_binary.cname in [x[0] for x in self.ann_phase2.taxonomy_rel]:
              rel.append({'class': ann_binary.cname})
            elif ann_binary.cname in [x[0] for x in self.ann_phase2.taxonomy_ta]:
              ta.append({'class': ann_binary.cname})
            else:
              assert False

          anns_unary = ann_hoi['task_result']['annotations'][3]['slotsChildren']
          anns_unary = [Description(ann_unary, self.ann_phase2.cn2en) for ann_unary in anns_unary]
          att, ia = [], []
          for ann_unary in anns_unary:
            if ann_unary.cname in [x[0] for x in self.ann_phase2.taxonomy_att]:
              att.append({'class': ann_unary.cname})
            elif ann_unary.cname in [x[0] for x in self.ann_phase2.taxonomy_ia]:
              ia.append({'class': ann_unary.cname})
            else:
              assert False

          hoi[0] = {
            'actor': actor,
            'object': object,
            'attribute': att,
            'relationship': rel,
            'intransitive action': ia,
            'transitive action': ta
          }

        sact[iid_sact] = {
          'class': ann_sact1['class'],
          'start': hms2s(ann_sact1['start']),
          'end': hms2s(ann_sact1['end']),
          'higher-order interaction': hoi
        }

      act[iid_act] = {
        'class': ann_act['class'],
        'start': hms2s(ann_act['crop_start']),
        'end': hms2s(ann_act['crop_end']),
        'sub-activity': sact
      }

    return act
