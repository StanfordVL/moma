import numpy as np


class Metadatum:
  def __init__(self, ann):
    self.id = ann['activity']['id']
    self.fname = ann['file_name']
    self.num_frames = ann['num_frames']
    self.width = ann['width']
    self.height = ann['height']
    self.duration = ann['duration']

  def get_fid(self, time):
    """ Get the frame ID given a timestamp in seconds
    """
    fps = (self.num_frames-1)/self.duration
    fid = time*fps
    return fid

  def get_time(self, fid):
    raise NotImplementedError

  @property
  def scale_factor(self):
    return min(self.width, self.height)/320

  def __repr__(self):
    return f'Metadatum(id={self.id}, fname={self.fname}, size=({self.num_frames}, {self.height}, {self.width}, 3), ' \
           f'duration={self.duration}'


class Act:
  def __init__(self, ann, taxonomy):
    self.id = ann['id']
    self.cname = ann['class_name']
    self.cid = taxonomy.index(ann['class_name'])
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.ids_sact = [x['id'] for x in ann['sub_activities']]

  def __repr__(self):
    return f'Act(id={self.id}, cname={self.cname}, time=[{self.start}, end={self.end}), num_sacts={len(self.ids_sact)}'


class SAct:
  def __init__(self, ann, scale_factor, taxonomy_sact, taxonomy_actor, taxonomy_object,
               taxonomy_ia, taxonomy_ta, taxonomy_att, taxonomy_rel):
    self.id = ann['id']
    self.cname = ann['class_name']
    self.cid = taxonomy_sact.index(ann['class_name'])
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.ids_hoi = [x['id'] for x in ann['higher_order_interactions']]
    self.times = [x['time'] for x in ann['higher_order_interactions']]

    # find unique entity instances
    ids_actor = sorted(set([y['id'] for x in ann['higher_order_interactions'] for y in x['actors']]))
    ids_object = sorted(set([y['id'] for x in ann['higher_order_interactions'] for y in x['objects']]))

    # group annotations by entity ID and frame ID
    actors = {id_actor:[None for _ in self.ids_hoi] for id_actor in ids_actor}
    objects = {id_object:[None for _ in self.ids_hoi] for id_object in ids_object}
    ias = {id_entity:[[] for _ in self.ids_hoi] for id_entity in ids_actor+ids_object}
    tas = {id_entity: [[] for _ in self.ids_hoi] for id_entity in ids_actor+ids_object}
    atts = {id_entity: [[] for _ in self.ids_hoi] for id_entity in ids_actor+ids_object}
    rels = {id_entity: [[] for _ in self.ids_hoi] for id_entity in ids_actor+ids_object}
    for i, ann_hoi_raw in enumerate(ann['higher_order_interactions']):
      for x in ann_hoi_raw['actors']:
        assert x['id'] not in actors or actors[x['id']][i] is None
        actors[x['id']][i] = Entity(x, 'actor', taxonomy_actor)
      for x in ann_hoi_raw['objects']:
        assert x['id'] not in objects or objects[x['id']][i] is None
        objects[x['id']][i] = Entity(x, 'object', taxonomy_object)
      for x in ann_hoi_raw['intransitive_actions']:
        ias[x['source_id']][i].append(Predicate(x, 'ia', taxonomy_ia))
      for x in ann_hoi_raw['transitive_actions']:
        tas[x['source_id']][i].append(Predicate(x, 'ta', taxonomy_ta))
      for x in ann_hoi_raw['attributes']:
        atts[x['source_id']][i].append(Predicate(x, 'att', taxonomy_att))
      for x in ann_hoi_raw['relationships']:
        rels[x['source_id']][i].append(Predicate(x, 'rel', taxonomy_rel))

    # create aacts
    info = {'start_time': self.start, 'end_time': self.end, 'times': self.times, 'scale_factor': scale_factor,
            'num_classes_ia': len(taxonomy_ia), 'num_classes_ta': len(taxonomy_ta), 
            'num_classes_att': len(taxonomy_att), 'num_classes_rel': len(taxonomy_rel)}
    self.aacts_actor = [AAct(info, actors[i], ias[i], tas[i], atts[i], rels[i]) for i in ids_actor]
    self.aacts_object = [AAct(info, objects[i], ias[i], tas[i], atts[i], rels[i]) for i in ids_object]

  @property
  def ids_actor(self):
    return [aact_actor.id_entity for aact_actor in self.aacts_actor]

  @property
  def ids_object(self):
    return [aact_object.id_entity for aact_object in self.aacts_object]

  @property
  def length(self):
    return len(self.times)

  def __repr__(self):
    return f'SAct(id={self.id}, cname={self.cname}, time=[{self.start}, end={self.end}), length={self.length})'


class AAct:
  def __init__(self, info, entities, ias, tas, atts, rels):
    entity = next(entity for entity in entities if entity is not None)
    self.id_entity = entity.id
    self.kind_entity = entity.kind
    self.cname_entity = entity.cname
    self.cid_entity = entity.cid

    self.start = info['start_time']
    self.end = info['end_time']
    self.times = info['times']

    self._scale_factor = info['scale_factor']
    self._entities = entities
    self._ias = ias
    self._tas = tas
    self._atts = atts
    self._rels = rels
    self._num_classes_ia = info['num_classes_ia']
    self._num_classes_ta = info['num_classes_ta']
    self._num_classes_att = info['num_classes_att']
    self._num_classes_rel = info['num_classes_rel']

  def get_bboxes(self, full_res=False):
    bboxes = []
    for entity in self._entities:
      if entity is None:
        bbox = None
      elif not full_res:
        bbox = BBox.scale(entity.bbox, self._scale_factor)
      else:
        bbox = entity.bbox
      bboxes.append(bbox)
    return bboxes

  @property
  def cids_predicate(self):  # binary
    indices_ia = np.array([[t, ia.cid] for t, ias in enumerate(self._ias) for ia in ias]).T
    cids_ia = np.zeros((self.length, self._num_classes_ia))
    if len(indices_ia) > 0:
      cids_ia[indices_ia[0], indices_ia[1]] = 1
    
    indices_ta = np.array([[t, ta.cid] for t, tas in enumerate(self._tas) for ta in tas]).T
    cids_ta = np.zeros((self.length, self._num_classes_ta))
    if len(indices_ta) > 0:
      cids_ta[indices_ta[0], indices_ta[1]] = 1
    
    indices_att = np.array([[t, att.cid] for t, atts in enumerate(self._atts) for att in atts]).T
    cids_att = np.zeros((self.length, self._num_classes_att))
    if len(indices_att) > 0:
      cids_att[indices_att[0], indices_att[1]] = 1
    
    indices_rel = np.array([[t, rel.cid] for t, rels in enumerate(self._rels) for rel in rels]).T
    cids_rel = np.zeros((self.length, self._num_classes_rel))
    if len(indices_rel) > 0:
      cids_rel[indices_rel[0], indices_rel[1]] = 1

    cids_predicate = np.concatenate((cids_ia, cids_ta, cids_att, cids_rel), axis=1)
    return cids_predicate

  @property
  def length(self):
    return len(self.times)

  def __repr__(self):
    return f'AAct_{self.kind}(id={self.id}, cname={self.cname}, time=[{self.start}, end={self.end}), ' \
           f'length={self.length})'


class HOI:
  def __init__(self, ann, taxonomy_actor, taxonomy_object, taxonomy_ia, taxonomy_ta, taxonomy_att, taxonomy_rel):
    self.id = ann['id']
    self.time = ann['time']
    self.actors = [Entity(x, 'actor', taxonomy_actor) for x in ann['actors']]
    self.objects = [Entity(x, 'object', taxonomy_object) for x in ann['objects']]
    self.ias = [Predicate(x, 'ia', taxonomy_ia) for x in ann['intransitive_actions']]
    self.tas = [Predicate(x, 'ta', taxonomy_ta) for x in ann['transitive_actions']]
    self.atts = [Predicate(x, 'att', taxonomy_att) for x in ann['attributes']]
    self.rels = [Predicate(x, 'rel', taxonomy_rel) for x in ann['relationships']]

  @property
  def ids_actor(self):
    return sorted([actor.id for actor in self.actors])

  @property
  def ids_object(self):
    return sorted([object.id for object in self.objects], key=int)

  def __repr__(self):
    return f'HOI(id={self.id}, time={self.time}, ' \
           f'num_actors={len(self.actors)}, num_objects={len(self.objects)}, ' \
           f'num_ias={len(self.ias)}, num_tas={len(self.tas)}, ' \
           f'num_atts={len(self.atts)}, num_rels={len(self.rels)}, ' \
           f'ids_actor={self.ids_actor}, ids_object={self.ids_object})'


class Clip:
  """ A clip corresponds to a 1 second/5 frames video clip centered at the higher-order interaction
   - <1 second/5 frames if exceeds the raw video boundary
   - Currently, only clips from the test set have been generated
  """
  def __init__(self, ann, neighbors):
    self.id = ann['id']
    self.time = ann['time']
    self.neighbors = neighbors


class BBox:
  def __init__(self, ann):
    self.x, self.y, self.width, self.height = ann
    
  @classmethod
  def scale(cls, bbox, scale_factor):
    return cls((round(bbox.x/scale_factor), round(bbox.y/scale_factor),
                round(bbox.width/scale_factor), round(bbox.height/scale_factor)))

  @property
  def x1(self):
      return self.x

  @property
  def y1(self):
      return self.y

  @property
  def x2(self):
      return self.x+self.width

  @property
  def y2(self):
      return self.y+self.height

  def __repr__(self):
    return f'BBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})'


class Entity:
  def __init__(self, ann, kind, taxonomy):
    self.id = ann['id']  # local instance ID
    self.kind = kind
    self.cname = ann['class_name']
    self.cid = taxonomy.index(self.cname)
    self.bbox = BBox(ann['bbox'])

  def __repr__(self):
    name = ''.join(x.capitalize() for x in self.kind.split('_'))
    return f'{name}(id={self.id}, cname={self.cname})'


class Predicate:
  def __init__(self, ann, kind, taxonomy):
    is_binary = 'target_id' in ann
    self.kind = kind
    self.signature = {x[0]:(x[1:] if is_binary else x[1]) for x in taxonomy}[ann['class_name']]
    self.cname = ann['class_name']
    self.cid = [x[0] for x in taxonomy].index(self.cname)
    self.id_src = ann['source_id']
    self.id_trg = ann['target_id'] if is_binary else None

  def __repr__(self):
    name = ''.join(x.capitalize() for x in self.kind.split('_'))
    id = f'{self.id_src}' if self.id_trg is None else f'{self.id_src} -> {self.id_trg}'
    return f'{name}(id={id}, cname={self.cname})'
