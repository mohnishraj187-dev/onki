import os
import numpy as np
import nibabel as nib
import torch

from models.tumor_model import load_model


class TumorSegmentationService:

    def __init__(self, checkpoint_path):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = load_model(
            checkpoint_path,
            self.device
        )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def normalize_volume(volume):

        volume = np.asarray(
            volume,
            dtype=np.float32
        )

        volume = np.nan_to_num(
            volume,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        minimum = np.min(volume)
        maximum = np.max(volume)

        denominator = (
            maximum - minimum
        )

        if denominator <= 0:

            return np.zeros_like(
                volume,
                dtype=np.float32
            )

        return (
            (volume - minimum)
            / denominator
        ).astype(np.float32)

    # ============================================================
    # LOAD NIFTI
    # ============================================================

    @staticmethod
    def load_nifti(path):

        nii = nib.load(path)

        data = nii.get_fdata(
            dtype=np.float32
        )

        if data.ndim != 3:

            raise ValueError(
                f"MRI must be 3D. "
                f"Received shape {data.shape}"
            )

        return nii, data

    # ============================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        t1n_path,
        t1c_path,
        t2w_path,
        t2f_path,
        output_path
    ):

        print()
        print("=" * 70)
        print("STARTING TUMOUR SEGMENTATION")
        print("=" * 70)

        paths = {
            "T1n": t1n_path,
            "T1c": t1c_path,
            "T2w": t2w_path,
            "T2f": t2f_path
        }

        # --------------------------------------------------------
        # Load all four modalities
        # --------------------------------------------------------

        volumes = {}
        reference_nii = None

        for modality, path in paths.items():

            if not os.path.exists(path):

                raise FileNotFoundError(
                    f"{modality} file not found: {path}"
                )

            nii, data = self.load_nifti(path)

            if reference_nii is None:
                reference_nii = nii
                reference_shape = data.shape

            if data.shape != reference_shape:

                raise ValueError(
                    f"Dimension mismatch.\n"
                    f"Expected: {reference_shape}\n"
                    f"{modality}: {data.shape}"
                )

            volumes[modality] = self.normalize_volume(
                data
            )

            print(
                f"{modality}: "
                f"{data.shape}"
            )

        # --------------------------------------------------------
        # Training uses:
        #
        # [T1n, T1c, T2w, T2f]
        #
        # as four channels.
        # --------------------------------------------------------

        t1n = volumes["T1n"]
        t1c = volumes["T1c"]
        t2w = volumes["T2w"]
        t2f = volumes["T2f"]

        x_dim, y_dim, z_dim = t1n.shape

        print()
        print(
            f"Volume dimensions: "
            f"{x_dim} x {y_dim} x {z_dim}"
        )

        print(
            f"Running inference on {z_dim} slices..."
        )

        predicted_mask = np.zeros(
            (x_dim, y_dim, z_dim),
            dtype=np.uint8
        )

        # --------------------------------------------------------
        # Process each axial slice
        #
        # This exactly follows the training dataset:
        #
        # image_stack =
        # np.stack([t1n,t1c,t2w,t2f], axis=0)
        # --------------------------------------------------------

        with torch.no_grad():

            for z in range(z_dim):

                slice_stack = np.stack(
                    [
                        t1n[:, :, z],
                        t1c[:, :, z],
                        t2w[:, :, z],
                        t2f[:, :, z]
                    ],
                    axis=0
                ).astype(np.float32)

                tensor = torch.from_numpy(
                    slice_stack
                )

                tensor = tensor.unsqueeze(0)

                tensor = tensor.to(
                    self.device
                )

                logits = self.model(
                    tensor
                )

                probabilities = torch.sigmoid(
                    logits
                )

                prediction = (
                    probabilities[0, 0]
                    > 0.5
                )

                predicted_mask[:, :, z] = (
                    prediction
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                if (
                    z % 10 == 0
                    or z == z_dim - 1
                ):

                    print(
                        f"Processed slice "
                        f"{z + 1}/{z_dim}"
                    )

        # --------------------------------------------------------
        # Save mask
        # --------------------------------------------------------

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        mask_nii = nib.Nifti1Image(
            predicted_mask.astype(np.uint8),
            reference_nii.affine,
            reference_nii.header
        )

        nib.save(
            mask_nii,
            output_path
        )

        tumour_voxels = int(
            np.sum(predicted_mask > 0)
        )

        print()
        print(
            f"Tumour voxels: "
            f"{tumour_voxels}"
        )

        print(
            f"Tumour mask saved to:"
            f"\n{output_path}"
        )

        print("=" * 70)

        return {
            "mask_path": output_path,
            "mask": predicted_mask,
            "dimensions": [
                x_dim,
                y_dim,
                z_dim
            ],
            "tumour_voxels": tumour_voxels
        }