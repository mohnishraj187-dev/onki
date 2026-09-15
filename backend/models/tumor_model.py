import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    EXACT architecture used during training.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(nn.Module):
    """
    EXACT SimpleUNet used by the training code.

    Input:
        [B, 4, H, W]

    Channels:
        0 = T1n
        1 = T1c
        2 = T2w
        3 = T2f

    Output:
        [B, 1, H, W]
    """

    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(64, 128)

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )

        self.conv4 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )

        self.conv5 = DoubleConv(64, 32)

        self.out = nn.Conv2d(
            32,
            out_channels,
            1
        )

    def forward(self, x):

        c1 = self.conv1(x)

        c2 = self.conv2(
            self.pool1(c1)
        )

        c3 = self.conv3(
            self.pool2(c2)
        )

        u1 = self.conv4(
            torch.cat(
                [
                    self.up1(c3),
                    c2
                ],
                dim=1
            )
        )

        u2 = self.conv5(
            torch.cat(
                [
                    self.up2(u1),
                    c1
                ],
                dim=1
            )
        )

        return self.out(u2)


def create_model():

    return SimpleUNet(
        in_channels=4,
        out_channels=1
    )


def load_model(checkpoint_path, device):

    print("=" * 70)
    print("LOADING TUMOUR SEGMENTATION MODEL")
    print("=" * 70)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    model = create_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    # Your training code uses:
    #
    # torch.save(model.state_dict(), checkpoint_path)
    #
    # so the checkpoint itself is the state_dict.

    if isinstance(checkpoint, dict):

        # Normal state_dict
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            state_dict = checkpoint

        # In case a future checkpoint wraps it
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            raise RuntimeError(
                "Checkpoint dictionary does not contain "
                "a recognizable state_dict."
            )

    else:

        raise RuntimeError(
            "Checkpoint is not a PyTorch state_dict."
        )

    # Remove DataParallel prefix if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        raise RuntimeError(
            "Missing model weights:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected model weights:\n"
            + "\n".join(unexpected)
        )

    model.to(device)
    model.eval()

    print("Tumour model loaded successfully.")
    print("=" * 70)

    return model