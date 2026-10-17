from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

class EuroSATMSDataset(Dataset):
    def __init__(self, DATA_ROOT, transform=None):
        super().__init__()
        self.data_root = Path(DATA_ROOT)
        self.root = self.data_root
        self.transform = transform

        cls_to_idx = {}
        class_dirs = [
            dir for dir in sorted(self.data_root.iterdir()) if dir.is_dir()
        ]
        for i, dir in enumerate(class_dirs):
            cls_to_idx[dir.name] = i

        self.classes = [directory.name for directory in class_dirs]
        self.class_to_idx = cls_to_idx

        samples = []
        for dir in class_dirs:
            # we sort for determinism
            for hsi in sorted(dir.iterdir()): 
                if hsi.suffix.lower() in {".tif", ".tiff"}:
                    samples.append((hsi, cls_to_idx[dir.name]))

        self.samples = samples
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        hsi, cls = self.samples[index]

        hsi_numpy = tifffile.imread(hsi).astype(np.float32)
        hsi_torch = torch.from_numpy(hsi_numpy)

        hsi_torch = hsi_torch.permute(2, 0, 1)
        if self.transform is not None:
            hsi_torch = self.transform(hsi_torch)

        return hsi_torch, cls

        