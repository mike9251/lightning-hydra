import os
import glob
import random
from PIL import Image

from typing import Optional, Tuple

import cv2
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor


class SimSwapDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 8,
            num_workers: int = 1,
            pin_memory: bool = False):
        super().__init__()

        # this line allows access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = Compose(
            [ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load dataset only if it's not loaded already
        if not self.train_set:
            self.train_set = SwappingDataset(self.hparams.data_dir,
                                             self.transforms,
                                             subfix='jpg',
                                             random_seed=1234,
                                             debug=False)
            self.val_set = SwappingDataset(self.hparams.data_dir,
                                           self.transforms,
                                           subfix='jpg',
                                           random_seed=1234,
                                           debug=False,
                                           batch_size=self.hparams.batch_size)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    # def val_dataloader(self):
    #     return DataLoader(
    #         dataset=self.val_set,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.num_workers,
    #         pin_memory=self.hparams.pin_memory,
    #         shuffle=False,
    #     )


class SwappingDataset(Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                 image_dir,
                 transform,
                 subfix: str = 'jpg',
                 random_seed: int = 1234,
                 debug: bool = False,
                 batch_size: Optional[int] = None):
        """Initialize and preprocess the Swapping dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.subfix = subfix
        self.dataset = []
        self.random_seed = random_seed
        self.debug = debug
        self.batch_size = batch_size
        self.preprocess()
        self.num_dirs = len(self.dataset)

    def preprocess(self) -> None:
        """Preprocess the Swapping dataset."""
        print("processing Swapping dataset images...")
        temp_path = os.path.join(self.image_dir, '*/')
        paths = glob.glob(temp_path)
        self.dataset = []
        for dir_item in paths:
            join_path = glob.glob(os.path.join(dir_item, f'*.{self.subfix}'))
            self.dataset.append(join_path)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        print('Finished preprocessing the Swapping dataset, total dirs number: %d...' % len(self.dataset))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two src domain images and two dst domain images."""
        dir_tmp1 = self.dataset[index]
        dir_tmp1_len = len(dir_tmp1)

        filename1 = dir_tmp1[random.randint(0, dir_tmp1_len - 1)]
        filename2 = dir_tmp1[random.randint(0, dir_tmp1_len - 1)]

        image1 = self.transform(Image.open(filename1))
        image2 = self.transform(Image.open(filename2))

        if self.debug:
            # Make sure images loaded correctly
            img_np = cv2.imread(filename1)
            cv2.imshow("img", img_np)
            cv2.waitKey(0)

            def denorm(x):
                mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
                std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
                out = std * x + mean
                return out.clamp_(0, 1)

            img_np = denorm(image1).numpy()
            img_np = img_np[0]
            img_np = img_np.transpose(1, 2, 0)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            cv2.imshow("img_denorm", img_np)
            cv2.waitKey(0)

        return image1, image2

    def __len__(self) -> int:
        """Return the number of image dirs (identities)."""
        if self.batch_size:
            return self.batch_size
        return self.num_dirs


if __name__ == '__main__':
    transforms = Compose(
        [ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )
    data = SwappingDataset(r"C:\Users\petrush\Downloads\SimSwap\content\TrainingData\vggface2_crop_arcfacealign_224",
                           transforms)

    loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    print(f"Num of batches: {len(loader)}")

    for batch in loader:
        batch = batch
        b1, b2 = batch
        b1 = b1.cuda(non_blocking=True)
        b2 = b2.cuda(non_blocking=True)
        print(f"B1: {b1.shape} B2: {b2.shape} device: {b2.device}")
