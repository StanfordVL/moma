from collections import defaultdict
from datetime import datetime, timedelta
from distinctipy import distinctipy
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
import os.path as osp
import tempfile

from ..utils import supress_stdout


def to_dt(sec):
    s = datetime(2000, 1, 1)  # does not matter
    return s + timedelta(seconds=sec)


def lighten(color, factor=0.5):
    color = tuple(x + (1 - x) * factor for x in color)
    return color


class TimelineVisualizer:
    def __init__(self, moma, dir_vis=None):
        if dir_vis is None:
            dir_vis = tempfile.mkdtemp()

        self.moma = moma
        self.dir_vis = dir_vis
        self.palette = {}

    def _get_palette(self, num_colors):
        if num_colors not in self.palette:
            self.palette[num_colors] = distinctipy.get_colors(
                num_colors, pastel_factor=0.5
            )
        return self.palette[num_colors]

    @supress_stdout
    def show(self, id_act, id_sact=None, id_hoi=None, path=None):
        os.makedirs(osp.join(self.dir_vis, "timeline"), exist_ok=True)

        metadatum = self.moma.get_metadata(ids_act=[id_act])[0]
        ann_act = self.moma.get_anns_act(ids_act=[id_act])[0]

        ids_sact = self.moma.get_ids_sact(ids_act=[id_act])
        index_sact = ids_sact.index(id_sact) if id_sact in ids_sact else None
        anns_sact = self.moma.get_anns_sact(ids_sact=ids_sact)
        anns_sact = sorted(anns_sact, key=lambda x: x.start)

        ids_hoi = self.moma.get_ids_hoi(ids_sact=ids_sact)
        index_hoi = ids_hoi.index(id_hoi) if id_hoi in ids_hoi else None
        anns_hoi = self.moma.get_anns_hoi(ids_hoi=ids_hoi)
        anns_hoi = sorted(anns_hoi, key=lambda x: x.time)

        interval_video = [0, metadatum.duration]
        interval_act = [ann_act.start, ann_act.end]
        label_act = ann_act.cname
        intervals_sact = [[ann_sact.start, ann_sact.end] for ann_sact in anns_sact]
        labels_sact = [ann_sact.cname for ann_sact in anns_sact]
        x = defaultdict(lambda: len(x))
        ids_color = [x[label_sact] for label_sact in labels_sact]
        colors_sact = self._get_palette(len(ids_color))
        times_hoi = [ann_hoi.time for ann_hoi in anns_hoi]

        interval_video = [to_dt(x) for x in interval_video]
        time_act = to_dt((interval_act[0] + interval_act[1]) / 2)
        interval_act = [to_dt(x) for x in interval_act]
        intervals_sact = [[to_dt(x[0]), to_dt(x[1])] for x in intervals_sact]
        times_hoi = [to_dt(x) for x in times_hoi]

        fig, ax = plt.subplots(figsize=(20, 2), constrained_layout=True)

        # draw video
        ax.plot(interval_video, [0, 0], "o-", color="lightgray", markerfacecolor="w")

        # draw act
        ax.fill_between(interval_act, -0.1, 1.2, color="black", alpha=0.1, linewidth=0)
        ax.plot(interval_act, [0, 0], "-", color="black")
        ax.text(time_act, 1.2, label_act, size="large", ha="center", va="bottom")

        # draw sact
        for i, (interval_sact, label_sact, id_color) in enumerate(
            zip(intervals_sact, labels_sact, ids_color)
        ):
            ax.fill_between(
                interval_sact, 0, 1, color=lighten(colors_sact[id_color]), linewidth=0
            )
            ax.vlines(
                interval_sact, 0, 1.1, color=colors_sact[id_color], linestyles="dashed"
            )
            ax.plot(
                interval_sact,
                [1, 1],
                "-",
                color=colors_sact[id_color],
                label=label_sact,
            )
            ax.plot(interval_sact[0], 1, marker=4, color=colors_sact[id_color])
            ax.plot(interval_sact[1], 1, marker=5, color=colors_sact[id_color])
        if index_sact is not None:
            interval = intervals_sact[index_sact]
            ax.fill_between(
                interval, -0.1, 1.2, color="firebrick", linewidth=0, alpha=0.3
            )

        # draw hoi
        ax.plot(
            times_hoi, [0] * len(times_hoi), "|", color="black", markerfacecolor="w"
        )
        if index_hoi is not None:
            time = times_hoi[index_hoi]
            ax.plot(
                interval_video,
                [-0.2, -0.2],
                "-",
                color="lightgray",
                linewidth=4,
                solid_capstyle="round",
            )
            ax.plot(
                [interval_video[0], time],
                [-0.2, -0.2],
                "-",
                color="firebrick",
                linewidth=4,
                solid_capstyle="round",
            )
            ax.plot(time, -0.2, marker="o", color="tab:red", markersize=10)
            ax.text(
                interval_video[1],
                -0.2,
                time.strftime("  %M:%S"),
                color="firebrick",
                va="center",
            )

        locator = mdates.SecondLocator(bysecond=[0, 30])
        format = mdates.DateFormatter("%M:%S")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(format)
        plt.setp(ax.get_xticklabels())

        ax.yaxis.set_visible(False)
        ax.spines[["left", "top", "right"]].set_visible(False)
        ax.margins(y=0.1)

        handles, labels = ax.get_legend_handles_labels()
        unique = [
            (h, l)
            for i, (h, l) in enumerate(zip(handles, labels))
            if l not in labels[:i]
        ]
        ax.legend(*zip(*unique))

        if path is None:
            fname = (
                f"{id_act}"
                + ("" if index_sact is None else f"{id_sact}")
                + ("" if index_hoi is None else f"{id_hoi}")
                + ".png"
            )
            path = osp.join(self.dir_vis, f"timeline/{fname}")
        plt.savefig(path)
        plt.close(fig)
        return path
