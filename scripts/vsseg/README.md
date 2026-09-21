# VS-Seg registration_v0_manual_v2 data pipeline

The pipeline keeps source WSI and annotations read-only. It separates dataset
orchestration from generic DeeperHistReg code.

Stages:

1. audit
   - map existing paired manual landmark files;
   - require configured correspondence count/name consistency;
   - fit HE-to-IHC affine in level-0 WSI pixels;
   - write public/private pair inventories and transform QC.

2. plan
   - find HE tissue on an appropriate pyramid level;
   - generate deterministic patch coordinates;
   - write geometry-specific patch plans.

3. materialize
   - read a larger HE context;
   - affine-resample IHC into the exact HE grid;
   - propagate the affine source-validity mask through DHR;
   - center-crop the final patch;
   - write image pairs atomically;
   - record tissue, affine validity, DHR validity, displacement and Jacobian QC;
   - support resume, retry and deterministic GPU shards.

4. merge
   - combine shard manifests into one canonical manifest;
   - reject duplicate/missing/unexpected sample IDs when completeness is required.

Default config:

    configs/vsseg/registration_v0_manual.json

Default geometry:

    final patch       1024 x 1024
    DHR context       2048 x 2048
    stride            1024
    minimum tissue    0.50
    minimum validity  0.98

Core commands:

    scripts/vsseg/run_registration_v0_manual.sh --stage audit
    scripts/vsseg/run_registration_v0_manual.sh --stage plan

Single-sample DHR smoke:

    scripts/vsseg/run_registration_v0_manual.sh --stage materialize --registration dhr --max-samples 1 --device cuda:0 --force

Full single-GPU resume:

    scripts/vsseg/run_registration_v0_manual.sh --stage materialize --registration dhr --device cuda:0

Eight-GPU mode uses one process per GPU with matching --num-shards 8 and
--shard-index 0 through 7. Each process writes its own manifest.

After all eight shards finish:

    scripts/vsseg/run_registration_v0_manual.sh --stage merge --registration dhr --num-shards 8

The canonical manifest can be consumed with:

    --dataset_mode vsseg_paired
    --dataroot /NAS145/linboliao/Data/VS-Seg/derived/registration_v0_manual_v2/patch_manifest_dhr_ps1024_ctx2048.csv

Resume semantics:

- ready schema-v2 samples with both files present are skipped;
- failed rows are skipped unless --retry-failed is set;
- tissue/validity/topology rejections are skipped unless --retry-rejected is set;
- --force recomputes selected rows;
- detailed failure messages are written only under the private derived directory.

DHR topology thresholds currently default to null. The pipeline records
deformation statistics first so thresholds can be chosen from the empirical
distribution rather than from an unvalidated hard-coded cutoff.
