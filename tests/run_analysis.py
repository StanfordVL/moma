import os
from pathlib import Path
from pprint import pprint

import momaapi


def analyze_split(moma):
    dists_overall = momaapi.get_dist_overall(moma.statistics_train, moma.statistics_val)
    dists_per_class = momaapi.get_dist_per_class(
        moma.distributions_train, moma.distributions_val
    )

    print(
        "The quality of our split (cosine distance b/w train & val, the smaller the better):"
    )
    pprint(dists_overall)
    pprint(dists_per_class)


def analyze_all(moma):
    pprint(moma.statistics)
    pprint(moma.distributions)


def main():
    dir_moma = os.path.join(Path.home(), "data/moma")
    moma = momaapi.MOMA(dir_moma)

    analyze_split(moma)
    analyze_all(moma)


if __name__ == "__main__":
    main()
