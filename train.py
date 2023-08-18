import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from module.protomaml import ProtoMAML

def train_model(
    train_loader,
    val_loader,
    device,
    epoch,
    exp,
    test_loader,
    proto_dim=3964,
    lr=1e-3,
    lr_inner=0.1,
    lr_output=0.1,
    num_inner_steps=4,
    ):
    # region PATH
    checkpoint_path = os.path.join(exp, 'ckpt')
    # endregion


    trainer = pl.Trainer(
      default_root_dir= exp,
      accelerator="gpu" if str(device).startswith("cuda") else "cpu",
      devices=1,
      max_epochs=epoch,
      # val_check_interval= 20,
      enable_progress_bar= True,
      callbacks=[
          ModelCheckpoint(
              dirpath = checkpoint_path,
              filename = 'best_model',
              verbose = True,
              save_last = True,
              save_weights_only=False,
              # every_n_train_steps = 20, # Sau 2500 bước thì lưu 1 checkpoint
              mode="max",
              monitor="val_acc"),
          LearningRateMonitor(logging_interval='step'),
          ],
      )

    model = ProtoMAML(
        proto_dim= proto_dim,
        lr = lr,
        lr_inner= lr_inner,
        lr_output= lr_output,
        num_inner_steps= num_inner_steps,
    )
    trainer.fit(model, train_loader, val_loader)
    try:
        trainer.test(test_loader)
    except Exception as e:
        print(e)

    return model