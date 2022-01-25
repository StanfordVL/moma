class bidict(dict):
  """
  A many-to-one bidirectional dictionary
  Reference: https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
  """
  def __init__(self, *args, **kwargs):
    super(bidict, self).__init__(*args, **kwargs)
    self.inverse = {}
    for key, value in self.items():
      self.inverse.setdefault(value, set()).add(key)

  def __setitem__(self, key, value):
    if key in self:
      self.inverse[self[key]].remove(key)
    super(bidict, self).__setitem__(key, value)
    self.inverse.setdefault(value, set()).add(key)

  def __delitem__(self, key):
    self.inverse[self[key]].remove(key)
    if len(self.inverse[self[key]]) == 0:
      del self.inverse[self[key]]
    super(bidict, self).__delitem__(key)


class Act:
  def __init__(self, ann, taxonomy):
    self.id = ann['id']
    self.cid = taxonomy.index(ann['class_name'])
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.ids_sact = [x['id'] for x in ann['sub_activities']]

  def __repr__(self):
    pass


class SAct:
  def __init__(self, ann, taxonomy):
    self.id = ann['id']
    self.cid = taxonomy.index(ann['class_name'])
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.ids_hoi = [x['id'] for x in ann['higher_order_interactions']]

  def __repr__(self):
    pass


class HOI:
  def __init__(self, ann, taxonomy):
    self.id = ann['id']
    self.time = ann['time']
    self.actors = [Entity(x, 'actors') for x in ann['actors']]
    self.objects = [Entity(x, 'objects') for x in ann['objects']]
    descriptions = []
    for kind in ['intransitive_actions', 'transitive_actions', 'attributes', 'relationships']:
      descriptions.append([Description(x, kind, taxonomy[kind]) for x in ann[kind]])
    self.ias, self.tas, self.atts, self.rels = descriptions

  def __repr__(self):
    pass


class Entity:
  def __init__(self, ann, kind):
    self.id = ann['id']
    self.kind = kind
    self.cname = ann['class_name']
    self.bbox = BBox(ann['bbox'])

  def __repr__(self):
    return f"{''.join(x.capitalize() for x in self.kind.split('_'))}(id={self.id}, cname={self.cname})"


class BBox:
  def __init__(self, ann):
    self.x, self.y, self.width, self.height = ann

  def __repr__(self):
    return f'BBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})'


class Description:
  def __init__(self, ann, kind, taxonomy):
    is_binary = 'target_id' in ann
    self.kind = kind
    self.signature = {x[0]:(x[1:] if is_binary else x[1]) for x in taxonomy}[ann['class_name']]
    self.cid = taxonomy.index(ann['class_name'])
    self.src_iid = ann['source_id']
    self.trg_iid = ann['target_id'] if is_binary else None

  def __repr__(self):
    return f"{''.join(x.capitalize() for x in self.kind.split('_'))}(cid={self.cid})"
