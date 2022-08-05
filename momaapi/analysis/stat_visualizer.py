import matplotlib.pyplot as plt
import os
import os.path as osp
from pprint import pprint
import seaborn as sns


class StatVisualizer:
    def __init__(self, moma, dir_vis):
        self.moma = moma
        self.dir_vis = dir_vis

    def show(self, with_split):
        os.makedirs(osp.join(self.dir_vis, "stats"), exist_ok=True)

        if with_split:
            distributions, hues = {}, {}
            for key in self.moma.distributions_train:
                distributions[key] = (
                    self.moma.distributions_train[key]
                    + self.moma.distributions_val[key]
                )
                hues[key] = ["train"] * len(self.moma.distributions_train[key]) + [
                    "val"
                ] * len(self.moma.distributions_val[key])
            pprint(self.moma.statistics_train, sort_dicts=False)
            pprint(self.moma.statistics_val, sort_dicts=False)

        else:
            distributions = self.moma.distributions
            hues = {key: None for key in self.moma.distributions}
            pprint(self.moma.statistics, sort_dicts=False)

        for key in distributions:
            counts = distributions[key]
            cnames = (
                self.moma.get_taxonomy(key) + self.moma.get_taxonomy(key)
                if with_split
                else self.moma.get_taxonomy(key)
            )
            if isinstance(cnames[0], tuple):
                cnames = [cname[0] for cname in cnames]
            hue = hues[key]
            fname = f"{key}{'_split' if with_split else ''}.png"
            color = None if with_split else "seagreen"
            palette = "dark" if with_split else None
            assert len(counts) == len(cnames), f"{key}: {len(counts)} vs {len(cnames)}"

            sns.set(style="darkgrid")
            width = max(20, int(0.25 * len(counts)))
            height = int(0.5 * width)
            fig, ax = plt.subplots(figsize=(width, height))
            sns.barplot(
                x=cnames,
                y=counts,
                hue=hue,
                ci=None,
                ax=ax,
                log=True,
                color=color,
                palette=palette,
            )
            ax.set(xlabel="class", ylabel="count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_ylim(bottom=1)
            plt.tight_layout()
            plt.savefig(osp.join(self.dir_vis, "stats", fname))
