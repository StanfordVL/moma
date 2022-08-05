import csv
import os
from pathlib import Path

from momaapi import MOMA


# Reference: https://github.com/JonathonLuiten/TrackEval/blob/master/docs/MOTChallenge-Official/Readme.md
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z> (1-based)
# <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, -1, -1, -1, -1 (1-based)
def main():
    dir_moma = os.path.join(Path.home(), "data/moma")
    dir_actor = os.path.join(dir_moma, "tracking/gt_actor")
    dir_object = os.path.join(dir_moma, "tracking/gt_object")
    os.makedirs(dir_actor, exist_ok=True)
    os.makedirs(dir_object, exist_ok=True)

    moma = MOMA(dir_moma)
    ids_sact = moma.get_ids_sact(split="test")
    for id_sact in ids_sact:
        ids_hoi = moma.get_ids_hoi(ids_sact=[id_sact])
        ids_hoi = moma.sort(ids_hoi=ids_hoi)
        anns_hoi = moma.get_anns_hoi(ids_hoi=ids_hoi)

        lines_actor, iids_actor = [], []
        for i, ann_hoi in enumerate(anns_hoi):
            for actor in ann_hoi.actors:
                lines_actor.append(
                    [
                        i + 1,
                        actor.id,
                        actor.bbox.x,
                        actor.bbox.y,
                        actor.bbox.width,
                        actor.bbox.height,
                        0,
                        actor.cid + 1,
                        1,
                    ]
                )
                iids_actor.append(actor.id)
        iids_actor = sorted(set(iids_actor))
        map_iid_actor = {iid: j + 1 for j, iid in enumerate(iids_actor)}
        for k in range(len(lines_actor)):
            lines_actor[k][1] = map_iid_actor[lines_actor[k][1]]

        lines_object, iids_object = [], []
        for i, ann_hoi in enumerate(anns_hoi):
            for object in ann_hoi.objects:
                lines_object.append(
                    [
                        i + 1,
                        object.id,
                        object.bbox.x,
                        object.bbox.y,
                        object.bbox.width,
                        object.bbox.height,
                        0,
                        object.cid + 1,
                        1,
                    ]
                )
                iids_object.append(object.id)
        iids_object = sorted(set(iids_object))
        map_iid_object = {iid: j + 1 for j, iid in enumerate(iids_object)}
        for k in range(len(lines_object)):
            lines_object[k][1] = map_iid_object[lines_object[k][1]]

        with open(os.path.join(dir_actor, f"{id_sact}.txt"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(lines_actor)

        with open(os.path.join(dir_object, f"{id_sact}.txt"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(lines_object)


if __name__ == "__main__":
    main()
