import numpy as np
import os
import pickle

import momaapi


def main():
    dir_moma = "/home/alan/data/moma"
    threshold = 25

    with open(os.path.join(dir_moma, "weights/model_final_571f7c.pkl"), "rb") as f:
        weights = pickle.load(f)

    moma = momaapi.MOMA(dir_moma)
    cnames = moma.get_cnames("object", threshold, "either")
    indices_cls = np.array([moma.lvis_mapper[cname] - 1 for cname in cnames])
    indices_bbox = np.stack(
        [4 * indices_cls, 4 * indices_cls + 1, 4 * indices_cls + 2, 4 * indices_cls + 3]
    ).flatten(order="F")
    indices_cls = np.append(
        indices_cls,
        weights["model"]["roi_heads.box_predictor.cls_score.weight"].shape[0] - 1,
    )

    print("Old dimensions:")
    print(weights["model"]["roi_heads.box_predictor.cls_score.weight"].shape)
    print(weights["model"]["roi_heads.box_predictor.cls_score.bias"].shape)
    print(weights["model"]["roi_heads.box_predictor.bbox_pred.weight"].shape)
    print(weights["model"]["roi_heads.box_predictor.bbox_pred.bias"].shape)

    w1 = weights["model"]["roi_heads.box_predictor.cls_score.weight"][indices_cls]
    b1 = weights["model"]["roi_heads.box_predictor.cls_score.bias"][indices_cls]
    w2 = weights["model"]["roi_heads.box_predictor.bbox_pred.weight"][indices_bbox]
    b2 = weights["model"]["roi_heads.box_predictor.bbox_pred.bias"][indices_bbox]

    weights["model"]["roi_heads.box_predictor.cls_score.weight"] = w1
    weights["model"]["roi_heads.box_predictor.cls_score.bias"] = b1
    weights["model"]["roi_heads.box_predictor.bbox_pred.weight"] = w2
    weights["model"]["roi_heads.box_predictor.bbox_pred.bias"] = b2

    print("\nNew dimensions:")
    print(weights["model"]["roi_heads.box_predictor.cls_score.weight"].shape)
    print(weights["model"]["roi_heads.box_predictor.cls_score.bias"].shape)
    print(weights["model"]["roi_heads.box_predictor.bbox_pred.weight"].shape)
    print(weights["model"]["roi_heads.box_predictor.bbox_pred.bias"].shape)

    with open(os.path.join(dir_moma, "weights/model_final_571f7c_moma.pkl"), "wb") as f:
        pickle.dump(weights, f)


if __name__ == "__main__":
    main()
