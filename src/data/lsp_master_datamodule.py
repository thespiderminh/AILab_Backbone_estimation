import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import DataLoader, Dataset, random_split
import hydra
import rootutils
from omegaconf import DictConfig
# from src.data.components.dataset import Customed_Dataset
from components.dataset import Customed_Dataset


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class DataModule(LightningDataModule):
    def __init__(
        self,
        train_val_test_split: Tuple[float, float] = (0.6, 0.2, 0.2),
        batch_size: int = 32,
        num_workers: int = 3,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_transform = A.Compose(
            [
                A.Resize(128, 128),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        self.val_transform = A.Compose(
            [
                A.Resize(128, 128),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    # stage: Optional[str] = None
    def setup(self):
        dataset = Customed_Dataset(transform=self.train_transform)

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            persistent_workers=True,
        )

    def teardown(self, stage: Optional[str] = None):
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        pass


@hydra.main(version_base="1.3", config_path="../../configs/data", config_name="lsp")
def main(cfg: DictConfig) -> Optional[float]:
    datamodule: LightningDataModule = hydra.utils.instantiate(config=cfg)
    datamodule.setup

    for batch in datamodule.train_dataloader():
        image, keypoints = batch
        # print(image.shape, keypoints.shape)
        # print(type(image), type(keypoints))
        break

    print(len(datamodule.train_dataloader()))
    print(len(datamodule.val_dataloader()))
    print(len(datamodule.test_dataloader()))


if __name__ == "__main__":
    main()
