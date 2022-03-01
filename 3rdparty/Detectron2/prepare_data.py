from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from distinctipy import distinctipy
from momaapi import MOMA


def create_dataset(moma, ids_hoi, cname_to_cid):
  anns_hoi = moma.get_anns_hoi(ids_hoi)

  records = []
  for id_hoi, ann_hoi in zip(ids_hoi, anns_hoi):
    image_path = moma.get_paths(ids_hoi=[id_hoi])[0]
    id_act = moma.get_ids_act(ids_hoi=[id_hoi])[0]
    metadatum = moma.get_metadata(ids_act=[id_act])[0]

    annotations = []
    for entity in ann_hoi.actors+ann_hoi.objects:
      annotation = {
        'bbox': [entity.bbox.x, entity.bbox.y, entity.bbox.width, entity.bbox.height],
        'bbox_mode': BoxMode.XYWH_ABS,
        'category_id': cname_to_cid[entity.cname],
        'iscrowd': 1 if entity.cname == 'crowd' else 0
      }
      annotations.append(annotation)

    record = {
      'file_name': image_path,
      'image_id': ann_hoi.id,
      'width': metadatum.width,
      'height': metadatum.height,
      'annotations': annotations
    }
    records.append(record)

  return records


def register_dataset(dir_moma, threshold_train=None, threshold_val=None):
  moma = MOMA(dir_moma)

  if threshold_train is None and threshold_val is None:
    """
     - Train on all classes
     - Validate on all classes
    """
    ids_hoi_train = moma.get_ids_hoi(split='train')
    ids_hoi_val = moma.get_ids_hoi(split='val')
    cnames = moma.get_cnames('actor')+moma.get_cnames('object')

  elif threshold_train is None:
    """
     - Train on all classes
     - Validate on the classes with more than threshold_val instances in the validation set
    """
    cnames_actor = moma.get_cnames(concept='actor', num_instances=threshold_val, split='val')
    cnames_object = moma.get_cnames(concept='object', num_instances=threshold_val, split='val')
    ids_hoi_train = moma.get_ids_hoi(split='train')
    ids_hoi_val = moma.get_ids_hoi(split='val', cnames_actor=cnames_actor, cnames_object=cnames_object)
    cnames = moma.get_cnames('actor')+moma.get_cnames('object')

  elif threshold_val is None:
    """ Train and validate on the classes with:
     - more than threshold_train instances in the training set
    """
    cnames_actor = moma.get_cnames(concept='actor', num_instances=threshold_train, split='train')
    cnames_object = moma.get_cnames(concept='object', num_instances=threshold_train, split='train')
    ids_hoi_train = moma.get_ids_hoi(split='train', cnames_actor=cnames_actor, cnames_object=cnames_object)
    ids_hoi_val = moma.get_ids_hoi(split='val', cnames_actor=cnames_actor, cnames_object=cnames_object)
    cnames = cnames_actor+cnames_object

  else:
    """ Train and val on the classes with:
     - more than threshold_train instances in the training set AND
     - more than threshold_val instances in the validation set
    """
    cnames_actor_train = moma.get_cnames(concept='actor', num_instances=threshold_train, split='train')
    cnames_object_train = moma.get_cnames(concept='object', num_instances=threshold_train, split='train')
    cnames_actor_val = moma.get_cnames(concept='actor', num_instances=threshold_val, split='val')
    cnames_object_val = moma.get_cnames(concept='object', num_instances=threshold_val, split='val')
    cnames_actor = list(set(cnames_actor_train).intersection(cnames_actor_val))
    cnames_object = list(set(cnames_object_train).intersection(cnames_object_val))
    ids_hoi_train = moma.get_ids_hoi(split='train', cnames_actor=cnames_actor, cnames_object=cnames_object)
    ids_hoi_val = moma.get_ids_hoi(split='val', cnames_actor=cnames_actor, cnames_object=cnames_object)
    cnames = cnames_actor+cnames_object

  cname_to_cid = {cname: i for i, cname in enumerate(cnames)}
  records_train = create_dataset(moma, ids_hoi_train, cname_to_cid)
  records_val = create_dataset(moma, ids_hoi_val, cname_to_cid)
  colors = distinctipy.get_colors(len(cnames))
  colors = [tuple(int(x*255) for x in color) for color in colors]

  DatasetCatalog.register('moma_train', records_train)
  DatasetCatalog.register('moma_val', records_val)
  MetadataCatalog.get('moma_train').thing_classes = cnames
  MetadataCatalog.get('moma_val').thing_classes = cnames
  MetadataCatalog.get('moma_train').thing_colors = colors
  MetadataCatalog.get('moma_val').thing_colors = colors
