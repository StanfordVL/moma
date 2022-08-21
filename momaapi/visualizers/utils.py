from scipy.spatial import distance


def get_dist_per_class(stats_per_class_train, stats_per_class_val):
    dists = {}

    for kind in stats_per_class_train:
        counts_train = stats_per_class_train[kind]["counts"]
        counts_val = stats_per_class_val[kind]["counts"]
        assert (
            len(counts_train)
            == len(counts_val)
            == len(stats_per_class_train[kind]["class_names"])
        )
        dist = distance.cosine(counts_train, counts_val)
        dists[f"{kind}_counts"] = dist

    return dists


def get_dist_overall(stats_overall_train, stats_overall_val):
    dists = {}

    ratio_best = stats_overall_train["activity"]["num_instances"] / (
        stats_overall_train["activity"]["num_instances"]
        + stats_overall_val["activity"]["num_instances"]
    )

    for kind in stats_overall_train:
        for stat in stats_overall_train[kind]:
            if stat == "num_classes":
                continue

            num_instances_train = stats_overall_train[kind][stat]
            num_instances_val = stats_overall_val[kind][stat]
            ratio = num_instances_train / (num_instances_train + num_instances_val)
            dist = abs(ratio - ratio_best) / ratio_best
            dists[f"{kind}_{stat}"] = dist

    return dists
