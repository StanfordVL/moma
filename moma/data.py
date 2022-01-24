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
  def __init__(self, ann):
    self.cname = ann['class_name']
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.sact = [SAct(x) for x in ann['sub_activities']]

  def __repr__(self):
    pass


class SAct:
  def __init__(self, ann):
    self.cname = ann['class_name']
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.hoi = [HOI(x) for x in ann['higher_order_interactions']]

  def __repr__(self):
    pass


class HOI:
  def __init__(self, ann):
    self.time = ann['time']
    self.actors = [Entity(x, 'actors') for x in ann['actors']]
    self.objects = [Entity(x, 'objects') for x in ann['objects']]
    self.atts = [Description(x, 'attributes') for x in ann['attributes']]
    self.rels = [Description(x, 'relationships') for x in ann['relationships']]
    self.ias = [Description(x, 'intransitive_actions') for x in ann['intransitive_actions']]
    self.tas = [Description(x, 'transitive_actions') for x in ann['transitive_actions']]

  def __repr__(self):
    pass


class Entity:
  def __init__(self, ann, kind):
    self.kind = kind
    self.iid = ann['instance_id']
    self.cname = ann['class_name']
    self.bbox = BBox(ann['bbox'])

  def __repr__(self):
    return f"{''.join(x.capitalize() for x in self.kind.split('_'))}(iid={self.iid}, cname={self.cname})"


class BBox:
  def __init__(self, ann):
    self.x, self.y, self.width, self.height = ann

  def __repr__(self):
    return f'BBox(x={self.x}, y={self.y}, width={self.width}, height={self.height})'


class Description:
  def __init__(self, ann, kind):
    self.kind = kind
    self.cname = ann['class_name']
    self.src_iid = ann['source_instance_id']
    self.trg_iid = ann['target_instance_id'] if 'target_instance_id' in ann else None

  def __repr__(self):
    return f"{''.join(x.capitalize() for x in self.kind.split('_'))}(cname={self.cname})"
