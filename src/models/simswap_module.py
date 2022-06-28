from typing import Any, List
import os
import random

import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.simswap_generator import Generator_Adain_Upsample
from src.models.components.pg_modules.discriminator import ProjectedDiscriminator

from src.losses.cos_dist_loss import cosine_dist_loss
from src.losses.adv_hinge_loss import adv_hinge_gen_loss, adv_hinge_disc_loss

from src.utils.plot import plot_batch


class SimSwapLitModule(LightningModule):
    """LightningModule for SimSwap model.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, opt_gen, opt_disc, loss, arcnet_path):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.generator = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.discriminator = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        self.discriminator.feature_network.requires_grad_(False)

        self.idnet = torch.load(self.hparams.arcnet_path, map_location=torch.device("cpu"))
        self.idnet.eval()
        for m in self.idnet.parameters():
            print(m.requires_grad)

        # loss functions
        self.adv_gen_loss = adv_hinge_gen_loss
        self.adv_disc_loss = adv_hinge_disc_loss
        self.rec_loss = F.l1_loss
        self.feat_matching_loss = F.l1_loss
        self.cos_dist_loss = cosine_dist_loss

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int, optimizer_idx: int):
        target_img, source_img = batch

        rand_index = list(range(target_img.shape[0]))
        random.shuffle(rand_index)

        if batch_idx % 2 == 0:
            source_img = source_img
        else:
            source_img = source_img[rand_index]

        source_img_112 = torch.clamp(F.interpolate(source_img, size=(112, 112), mode='bicubic'), min=0.0)
        # source_img_112 = F.interpolate(source_img, size=(112, 112), mode='bilinear')
        latent_id = self.idnet(source_img_112)
        latent_id = F.normalize(latent_id, p=2, dim=1)
        loss = 0.0
        # print(f'Batch_ids: {batch_idx} optimizer_idx: {optimizer_idx}')
        if optimizer_idx == 0:
            img_fake = self.generator(target_img, latent_id)

            gen_logits, feat = self.discriminator(img_fake, None)

            adv_gen_loss = self.adv_gen_loss(gen_logits)

            img_fake_112 = torch.clamp(F.interpolate(img_fake, size=(112, 112), mode='bicubic'), min=0.0)
            # img_fake_112 = F.interpolate(img_fake, size=(112, 112), mode='bilinear')
            latent_fake = self.idnet(img_fake_112)
            latent_fake = F.normalize(latent_fake, p=2, dim=1)
            id_loss = self.cos_dist_loss(latent_fake, latent_id)

            # Attributes of the target image should be preserved
            real_feat = self.discriminator.get_feature(target_img)
            feat_match_loss = self.feat_matching_loss(feat["3"], real_feat["3"])

            loss = adv_gen_loss + id_loss * self.hparams.loss.id_loss_weight + feat_match_loss * self.hparams.loss.fm_loss_weight

            self.log(f"train/id_loss", id_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/adv_gen_loss", adv_gen_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/feat_match_loss", feat_match_loss, on_step=False, on_epoch=True, prog_bar=True)

            # source id == target id, so fake should be identical (close) to the target
            if batch_idx % 2 == 0:
                rec_loss = self.rec_loss(img_fake, target_img) * self.hparams.loss.rec_loss_weight
                loss += rec_loss
                self.log(f"train/rec_loss", rec_loss, on_step=False, on_epoch=True, prog_bar=True)

            self.log(f"train/gen_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        else:
            img_fake = self.generator(target_img, latent_id)
            gen_logits, _ = self.discriminator(img_fake.detach(), None)
            real_logits, _ = self.discriminator(source_img, None)
            adv_disc_fake_loss, adv_disc_real_loss = self.adv_disc_loss(gen_logits, real_logits)
            loss = adv_disc_fake_loss + adv_disc_real_loss
            self.log(f"train/adv_disc_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/adv_disc_fake_loss", adv_disc_fake_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"train/adv_disc_real_loss", adv_disc_real_loss, on_step=False, on_epoch=True, prog_bar=True)

        # loss, preds, targets = self.step(batch)

        # log train metrics
        # acc = self.train_acc(preds, targets)

        # self.log(f"train/loss_{model}", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}  # , "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        def tensor2img(tensor) -> np.ndarray:
            imagenet_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            imagenet_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            return (tensor.cpu() * imagenet_std + imagenet_mean).numpy()

        target_img, source_img = batch

        source_img_112 = torch.clamp(F.interpolate(source_img, size=(112, 112), mode='bicubic'), min=0.0)
        latent_id = self.idnet(source_img_112)
        latent_id = F.normalize(latent_id, p=2, dim=1)

        batch_size = source_img.shape[0]

        tiles: List[np.ndarray] = [tensor2img(target_img[i]) for i in range(batch_size)]
        tiles.insert(0, np.zeros_like(tensor2img(source_img[0])))

        for i in range(batch_size):
            ident = latent_id[i].repeat(batch_size, 1)
            img_fake = self.generator(target_img, ident)

            tile = [tensor2img(img_fake[i]) for i in range(batch_size)]
            tile.insert(0, tensor2img(source_img[i]))
            tiles = tiles + tile

        tiles = np.stack(tiles, axis=0).transpose(0, 2, 3, 1)

        plot_batch(tiles, os.path.join(os.getcwd(), 'epoch_' + str(self.current_epoch) + '.jpg'))

        loss = torch.tensor([0.0], requires_grad=False)
        return {"loss": loss}  # , "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass
        # acc = self.val_acc.compute()  # get val accuracy from current epoch
        # self.val_acc_best.update(acc)
        # self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.opt_gen.lr,
                                         betas=(self.hparams.opt_gen.beta1, 0.99), eps=1e-8)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.opt_disc.lr,
                                          betas=(self.hparams.opt_disc.beta1, 0.99), eps=1e-8)

        return [optimizer_gen, optimizer_disc]
