import preproc


def main():
  dir_moma = '/home/alan/ssd/moma'
  fname_ann_phase1 = 'video_anns_phase1_processed.json'
  fname_ann_phase2 = 'MOMA-videos-0121-all.jsonl'

  taxonomy_parser = preproc.TaxonomyParser(dir_moma)
  taxonomy_parser.dump(verbose=False)

  ann_phase1 = preproc.AnnPhase1(dir_moma, fname_ann_phase1)
  ann_phase1.inspect(verbose=False)

  ann_phase2 = preproc.AnnPhase2(dir_moma, fname_ann_phase2)
  ann_phase2.inspect(verbose=False)

  ann_merger = preproc.AnnMerger(dir_moma, ann_phase1, ann_phase2)
  ann_merger.dump()


if __name__ == '__main__':
  main()
