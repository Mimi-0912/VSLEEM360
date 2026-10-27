# VSLEEM360 staging data card

- Status: local staging; no public URL/version/date is asserted.
- Planned coverage: all 421 sequences, approximately78.3h, dual ERP video and sensor logs.
- Selected data license: CC BY-NC4.0; see LICENSE.txt. Code/third-party weights are separate.
- Video input: one explicitly selected HD/Preview pair per sequence. The six repaired videos for LDH/XYR/VMY are selected from release_redacted_20261027; original process/process2 alternatives remain internal archives.
- Privacy: automated processing is imperfect; 60 historical audit frames included four failures. The four JPEGs were reprocessed and user-reviewed. Six full-video copies were reprocessed with localized redaction; 50 known/nearby/uniform frames were sampled and all frame presentation timestamps/frame counts preserved. No full-corpus per-frame human review or privacy certification is claimed.
- Annotations: 421 author-provided attribute rows, without an independent second annotator or published agreement statistic.
- GPS: original validity labels419/421 are not accuracy certificates. Raw logs have rows in409 sequences, while camera trajectory files exist for421. Median reference geotag interval is2.09s; interpolation and unfiltered outliers affect VO metrics. No independent RTK accuracy reference.
- Splits:252 train/84 validation/85 frozen test, no overlap; five historical recovered training IDs are explicitly included. No assertion that a seed reconstructs the test list.
- VPR: use the separately frozen78386 HD/Preview JPEG inputs and metadata, not frames decoded anew from privacy-reencoded videos. benchmark_inputs.jsonl/VPR_INPUT_SHA256SUMS record exact identities. Positive threshold <=25m;10622 answerable queries in the reported geographic split.
- VSR:255 clips over85 sequences; image-derived diagnostics on all255 are in the revision report.246 peak within one frame,8 at2--3, one at the search boundary. These are not physical clock-error measurements.
- Public indexes:dataset_manifest.csv, file_manifest.json and SHA256SUMS use relative paths. release_files.json is an internal assembly map containing local source paths and should not be mistaken for a downloader path configuration.
- Checksums: unchanged original digests are retained from previous full-corpus hashing; edited videos and small metadata are freshly hashed. Checksum validity is distinct from GPS, annotation or privacy quality.
