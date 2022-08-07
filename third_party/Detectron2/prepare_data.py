from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from distinctipy import distinctipy


def create_dataset(moma, ids_hoi, kind, cname_to_cid):
    records = []

    for id_hoi in ids_hoi:
        ann_hoi = moma.get_anns_hoi([id_hoi])[0]
        image_path = moma.get_paths(ids_hoi=[id_hoi])[0]
        id_act = moma.get_ids_act(ids_hoi=[id_hoi])[0]
        metadatum = moma.get_metadata(ids_act=[id_act])[0]

        if kind is None:
            entities = ann_hoi.actors + ann_hoi.objects
        elif kind == "actor":
            entities = ann_hoi.actors
        else:  # kind == 'object'
            entities = ann_hoi.objects

        annotations = []
        for entity in entities:
            if entity.cname in cname_to_cid:
                annotation = {
                    "bbox": [
                        entity.bbox.x,
                        entity.bbox.y,
                        entity.bbox.width,
                        entity.bbox.height,
                    ],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": cname_to_cid[entity.cname],
                }
                annotations.append(annotation)

        record = {
            "file_name": image_path,
            "image_id": ann_hoi.id,
            "width": metadatum.width,
            "height": metadatum.height,
            "annotations": annotations,
        }
        records.append(record)

    return records


def register_datasets(moma, threshold=25, kind=None):
    """
    - kind: 'actor' or 'object' or None (both)
    """
    if kind is None:
        cnames = moma.get_cnames("actor", threshold, "either") + moma.get_cnames(
            "object", threshold, "either"
        )
    else:
        cnames = moma.get_cnames(kind, threshold, "either")

    # remove 'crowd'
    if "crowd" in cnames:
        cnames.remove("crowd")

    cname_to_cid = {cname: i for i, cname in enumerate(cnames)}
    ids_hoi_train = moma.get_ids_hoi(split="train")
    ids_hoi_val = moma.get_ids_hoi(split="val")

    colors = distinctipy.get_colors(len(cnames))
    colors = [tuple(int(x * 255) for x in color) for color in colors]

    DatasetCatalog.register(
        "moma_train", lambda: create_dataset(moma, ids_hoi_train, kind, cname_to_cid)
    )
    DatasetCatalog.register(
        "moma_val", lambda: create_dataset(moma, ids_hoi_val, kind, cname_to_cid)
    )
    MetadataCatalog.get("moma_train").thing_classes = cnames
    MetadataCatalog.get("moma_val").thing_classes = cnames
    MetadataCatalog.get("moma_train").thing_colors = colors
    MetadataCatalog.get("moma_val").thing_colors = colors
