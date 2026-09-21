import csv
from pathlib import Path

from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class VssegPairedDataset(BaseDataset):
    """Manifest-driven paired HE/IHC dataset."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--vsseg_stain",
            type=str,
            default="",
            help="optional exact stain filter, e.g. CKpan or p63+CKpan",
        )
        parser.add_argument(
            "--vsseg_registration_stage",
            type=str,
            default="",
            choices=["", "affine", "dhr"],
            help="optional registration_stage filter",
        )
        parser.set_defaults(preprocess="none")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        manifest_path = Path(opt.dataroot)
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "--dataroot must point to a VS-Seg patch manifest CSV, got %s"
                % manifest_path
            )

        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

        phase = getattr(opt, "phase", "train")
        self.rows = [
            row
            for row in rows
            if row.get("status") == "ready"
            and row.get("split") == phase
            and (
                not getattr(opt, "vsseg_stain", "")
                or row.get("stain") == opt.vsseg_stain
            )
            and (
                not getattr(opt, "vsseg_registration_stage", "")
                or row.get("registration_stage") == opt.vsseg_registration_stage
            )
        ]
        if not self.rows:
            raise RuntimeError(
                "No ready VS-Seg samples for phase=%s stain=%s stage=%s in %s"
                % (
                    phase,
                    getattr(opt, "vsseg_stain", "") or "*",
                    getattr(opt, "vsseg_registration_stage", "") or "*",
                    manifest_path,
                )
            )

        self.input_nc = (
            opt.output_nc if opt.direction == "BtoA" else opt.input_nc
        )
        self.output_nc = (
            opt.input_nc if opt.direction == "BtoA" else opt.output_nc
        )

    def __getitem__(self, index):
        row = self.rows[index]
        a_path = row["he_patch_path"]
        b_path = row["ihc_patch_path"]
        if not a_path or not b_path:
            raise RuntimeError("Ready manifest row has empty patch path")

        A = Image.open(a_path).convert("RGB")
        B = Image.open(b_path).convert("RGB")
        if A.size != B.size:
            raise ValueError(
                "Paired patch size mismatch for %s: %s vs %s"
                % (row["sample_id"], A.size, B.size)
            )

        manifest_patch_size = int(row.get("patch_size", A.size[0]))
        if A.size != (manifest_patch_size, manifest_patch_size):
            raise ValueError(
                "Patch size disagrees with manifest for %s: image=%s manifest=%d"
                % (row["sample_id"], A.size, manifest_patch_size)
            )

        params = get_params(self.opt, A.size)
        A_transform = get_transform(
            self.opt,
            params,
            grayscale=(self.input_nc == 1),
        )
        B_transform = get_transform(
            self.opt,
            params,
            grayscale=(self.output_nc == 1),
        )
        A = A_transform(A)
        B = B_transform(B)

        return {
            "A": A,
            "B": B,
            "A_paths": a_path,
            "B_paths": b_path,
            "sample_id": row["sample_id"],
            "pair_id": row["pair_id"],
            "case_id": row["case_id"],
            "patient_group_id": row["patient_group_id"],
            "stain": row["stain"],
            "registration_version": row["registration_version"],
            "registration_stage": row["registration_stage"],
            "dataset_variant": row["dataset_variant"],
            "valid_fraction": float(row["valid_fraction"]),
        }

    def __len__(self):
        return len(self.rows)
