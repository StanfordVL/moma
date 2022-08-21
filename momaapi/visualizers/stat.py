import matplotlib.pyplot as plt
import os
import os.path as osp
from pprint import pprint
import seaborn as sns
import tempfile


class StatVisualizer:
    def __init__(self, moma, dir_vis=None):
        if dir_vis is None:
            dir_vis = tempfile.mkdtemp()

        self.moma = moma
        self.dir_vis = dir_vis

    def show(self, with_split):
        os.makedirs(osp.join(self.dir_vis, "stats"), exist_ok=True)

        keys = [
            x for x in self.moma.statistics["all"].keys() if x != "raw" and x != "hoi"
        ]
        if with_split:
            distributions, hues = {}, {}
            for key in keys:
                distributions[key] = (
                    self.moma.statistics["standard_train"][key]["distribution"]
                    + self.moma.statistics["standard_val"][key]["distribution"]
                    + self.moma.statistics["standard_test"][key]["distribution"]
                )
                hues[key] = (
                    ["train"]
                    * len(self.moma.statistics["standard_train"][key]["distribution"])
                    + ["val"]
                    * len(self.moma.statistics["standard_val"][key]["distribution"])
                    + ["test"]
                    * len(self.moma.statistics["standard_test"][key]["distribution"])
                )

        else:
            distributions = {
                key: self.moma.statistics["all"][key]["distribution"] for key in keys
            }
            hues = {key: None for key in keys}

        paths = {}
        for key in distributions:
            counts = distributions[key]
            cnames = (
                self.moma.taxonomy[key] * 3 if with_split else self.moma.taxonomy[key]
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

            path = osp.join(self.dir_vis, "stats", fname)
            paths[key] = path
            plt.savefig(path)
            plt.close(fig)

        return paths
