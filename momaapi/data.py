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

    # FIXME: fix negative size error
    self.x = min(x_tl, x_tr)
    self.y = min(y_tl, y_bl)
    self.width = abs(x_tr-x_tl)
    self.height = abs(y_bl-y_tl)

  def __repr__(self):
    return f'({self.x}, {self.y}, {self.width}, {self.height})'


class Entity:
  def __init__(self, entity_raw, cn2en):
    """ type """
    type = cn2en[entity_raw['slot']['label']]

    """ cname """
    cname = entity_raw['children'][0]['input']['value']
    # FIXME: fix type error
    if cname == '服务员':
      type = 'actor'
    if cname == '花洒' or cname == '飞盘' or cname == '钢琴' or cname == '足球门' or cname == '车' or cname == '纸' or \
       cname == '剃刀' or cname == '梳子' or cname == '吹风机' or cname == '相机' or \
       cname == '无法识别但确实和正在进行的动作相关的物体':
      type = 'object'
    # FIXME: fix typos
    cname = cname.replace(',', '，').replace(' ', '').replace('·', '')\
                 .replace('蓝球', '篮球').replace('蓝球框', '篮球框').replace('篮子', '篮球框').replace('篮筐', '篮球框')
    assert cname in cn2en, '[Entity] unseen cname {}'.format(cname)

    """ iid """
    iid = entity_raw['children'][1]['input']['value']
    # FIXME: fix typos
    iid = iid.replace(' ', '').replace('\n', '').upper()

    """ bbox """
    bbox = BBox(entity_raw['slot']['plane'])

    self.type = type
    self.cname = cn2en[cname]
    self.iid = iid
    self.bbox = bbox


class Description:
  def __init__(self, description_raw, cn2en):
    """ type """
    type = cn2en[description_raw['slot']['label']]

    """ cname """
    cname = description_raw['children'][0]['input']['value']
    # FIXME: fix typos
    cname = cname.replace('(', '（').replace(')', '）').replace(',', '，').replace(' ', '')\
                 .replace('保持低头的姿势', '保持低头的姿势且身体没有移动')\
                 .replace('保持蹲着的姿势', '保持蹲着的姿势且身体没有移动')\
                 .replace('保持跪姿并且没有移动', '保持跪姿并且身体没有移动')
    assert cname in cn2en, '[Description] unseen cname {}'.format(cname)

    """ iids_associated """
    iids_associated = description_raw['children'][1]['input']['value']
    # FIXME: fix typos
    iids_associated = iids_associated.replace('（', '(').replace('）', ')').replace('，', ',')\
                                     .replace(' ', '').replace('\n', '').replace('.', ',').replace(')(', '),(')

    self.type = type
    self.cname = cn2en[cname]
    self.iids_associated = iids_associated
