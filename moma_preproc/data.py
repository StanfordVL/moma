import itertools
from collections import defaultdict


def fix_cname(cname):
  cname = cname.replace('(', '（').replace(')', '）').replace(',', '，')
  cname = cname.replace(' ', '').replace('·', '').replace('蓝球', '篮球')
  return cname


def fix_id(id):
  id = id.replace('（', '(').replace('）', ')').replace('，', ',')
  id = id.replace(' ', '').replace('\n', '').upper()
  id = id.replace('.', ',').replace(',,', ',')
  return id


class BBox:
  def __init__(self, bbox_raw):
    x_tl = round(bbox_raw['topLeft']['x'])
    y_tl = round(bbox_raw['topLeft']['y'])
    x_tr = round(bbox_raw['topRight']['x'])
    y_tr = round(bbox_raw['topRight']['y'])
    x_bl = round(bbox_raw['bottomLeft']['x'])
    y_bl = round(bbox_raw['bottomLeft']['y'])
    x_br = round(bbox_raw['bottomRight']['x'])
    y_br = round(bbox_raw['bottomRight']['y'])

    assert x_tl == x_bl and x_tr == x_br and y_tl == y_tr and y_bl == y_br, \
        '[BBox] inconsistent coordinates {}'.format(bbox_raw)

    # ignore negative size errors
    self.x = min(x_tl, x_tr)
    self.y = min(y_tl, y_bl)
    self.width = abs(x_tr-x_tl)
    self.height = abs(y_bl-y_tl)

  def __repr__(self):
    return f'({self.x}, {self.y}, {self.width}, {self.height})'


class Entity:
  def __init__(self, entity_raw, cn2en):
    """ kind """
    kind = cn2en[entity_raw['slot']['label']]

    """ cname """
    cname = entity_raw['children'][0]['input']['value']
    cname = fix_cname(cname)
    assert cname in cn2en, '[Entity] unseen cname {}'.format(cname)

    """ id """
    id = entity_raw['children'][1]['input']['value']
    id = fix_id(id)

    """ bbox """
    bbox = BBox(entity_raw['slot']['plane'])

    self.kind = kind
    self.cname = cn2en[cname]
    self.id = id
    self.bbox = bbox


class Description:
  def __init__(self, description_raw, cn2en):
    """ kind """
    kind = cn2en[description_raw['slot']['label']]

    """ cname """
    cname = description_raw['children'][0]['input']['value']
    cname = fix_cname(cname)
    assert cname in cn2en, '[Description] unseen cname {}'.format(cname)

    """ ids_associated """
    ids_associated = description_raw['children'][1]['input']['value']
    ids_associated = fix_id(ids_associated)

    self.kind = kind
    self.cname = cn2en[cname]
    self.ids_associated = ids_associated

  def breakdown(self):
    if self.kind == 'binary description':
      ids_src = self.ids_associated[1:-1].split('),(')[0].split(',')
      ids_trg = self.ids_associated[1:-1].split('),(')[1].split(',')
      return list(itertools.product(ids_src, ids_trg, [self.cname]))
    elif self.kind == 'unary description':
      ids_src = self.ids_associated.split(',')
      return list(itertools.product(ids_src, [self.cname]))
    else:
      assert False


def defaultdict_to_dict(d):
  if isinstance(d, defaultdict):
    d = {k: defaultdict_to_dict(v) for k, v in d.items()}
  return d
