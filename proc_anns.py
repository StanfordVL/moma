import itertools

import moma


def main():
  dir_moma = '/home/alan/ssd/moma'
  fname_ann_phase1 = 'video_anns_phase1_processed.json'
  fname_ann_phase2 = 'MOMA-videos-0121-all.jsonl'

  # ann_phase1 = moma.AnnPhase1(dir_moma, fname_ann_phase1)
  # ann_phase1.inspect(verbose=False)
  # distribution = ann_phase1.get_distribution()
  # pprint(distribution)

  ann_phase2 = moma.AnnPhase2(dir_moma, fname_ann_phase2)
  ann_phase2.inspect(verbose=False)

  # iids_sact_phase1 = set(itertools.chain(*ann_phase1.iid_act_to_iids_sact.values()))
  # iids_sact_phase2 = set(ann_phase2.iids_sact)
  # print(iids_sact_phase2-iids_sact_phase1)


if __name__ == '__main__':
  main()
