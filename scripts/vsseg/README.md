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
distribution rather than from an unvalidated hard-coded cutoff.\n\n## registration_v1_component

registration_v1_component is the active VS-Seg landmark-free registration
pipeline. It replaces the earlier SIFT-first prototypes.

Pipeline:

    low-resolution tissue detection
      -> merge fragments into block-level tissue components
      -> HE/IHC component matching
      -> KFB cross-level read-integrity QC
      -> common trusted pyramid level per component pair
      -> bbox / PCA-shape / SIFT candidate generation
      -> signed-distance tissue-shape refinement
      -> multimodal mutual-information refinement
      -> candidate scoring by tissue Dice, boundary distance, NMI,
         and internal gradient correlation
      -> medium/high confidence gate
      -> DHR review using only the already-read trusted component arrays

Reader rules:

- KFB read_region uses coordinates in the selected pyramid level when level > 0;
  registration v1 converts level-0 coordinates by level_downsample.
- read_fixed_region is never used by registration v1.
- suspected KFB corruption is handled by cross-level integrity QC and fallback
  to the first trusted level, not by tile mosaicing.
- SIFT is only one candidate source. Match count/inlier ratio cannot by itself
  promote a registration.

Formal files:

    configs/vsseg/registration_v1_component.json
    scripts/vsseg/registration_utils.py
    scripts/vsseg/registration_v1_component.py
    scripts/vsseg/test_registration_v1_component.py

Run:

    export PYTHONPATH=/NAS3/lbliao/Code-138/pytorch-CycleGAN-and-pix2pix/scripts/vsseg:/NAS3/lbliao/Code-138:/NAS3/lbliao/Code-138/aslide:$PYTHONPATH
    export LD_LIBRARY_PATH=/usr/local/lib/aslide-lib/lib:$LD_LIBRARY_PATH

    /data12/jing/anaconda3/envs/DHR/bin/python \
      scripts/vsseg/registration_v1_component.py \
      --pair-id PAIR_9CDF1C3DBCCB \
      --dhr-device cuda:0

Formal outputs:

    /NAS145/linboliao/Data/VS-Seg/derived/registration_v1_component
    /NAS145/linboliao/Data/VS-Seg/reports/dataset_audit/registration_v1_component

Per-component reports:

    02_component_XX_read_integrity.csv
    03_component_XX_candidates.csv
    04_component_XX_structure_affine.png
    05_component_XX_dhr_review.png

The DHR image is a trusted-level registration review artifact. High-resolution
training-patch materialization still requires a separate WSI ROI integrity gate;
a lower trusted registration level must never be upsampled and presented as a
real high-magnification training patch.

Retired experimental implementations remain available through Git history and
are intentionally absent from the active work tree.\n
## registration_v1_component paired patch pilot

The active small-scale materializer is:

    configs/vsseg/patches_registration_v1_pilot.json
    scripts/vsseg/materialize_registration_v1_patches.py
    scripts/vsseg/test_materialize_registration_v1_patches.py

Default pilot output:

    /NAS145/linboliao/Data/VS-Seg/derived/patches/
      registration_v1_component/ps1024_ctx2048/pilot_v0

The planner samples component-aware HE locations from the trusted registration
level. Materialization always reads the actual level-0 HE context and the actual
level-0 IHC affine source ROI, then compares each against its component-specific
trusted pyramid level. A failed integrity check is recorded as
roi_integrity_rejected and never reaches DHR or the training image tree.

Ready images are stored once, independent of split:

    images/he/<PAIR_ID>/<PATCH_ID>.png
    images/ihc/<PAIR_ID>/<PATCH_ID>.png

Split membership and all QC/provenance are manifest-driven. Raw WSI paths are
kept only under manifests/private.

Typical pilot workflow:

    /data12/jing/anaconda3/envs/DHR/bin/python       scripts/vsseg/materialize_registration_v1_patches.py --stage plan

    /data12/jing/anaconda3/envs/DHR/bin/python       scripts/vsseg/materialize_registration_v1_patches.py       --stage materialize --device cuda:0

The canonical manifests/patch_manifest.csv can be passed directly to the
vsseg_paired Dataset adapter. Do not promote pilot_v0 into the full dataset
without reviewing the contact sheets and representative per-patch registration
QC.
