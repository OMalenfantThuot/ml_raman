from schnetpack.train.hooks import Hook
import schnetpack as spk
from loss import get_loss_fn
from torch.optim import Adam
from utils.models import SmoothTrainer
import os
import torch


def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr, weight_decay=args.l2reg)

    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))

    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)
    hooks.append(PatchingHook())

    if args.logger == "csv":
        logger = spk.train.CSVHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)
    elif args.logger == "tensorboard":
        logger = spk.train.TensorboardHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    if args.save_n_steps > 0:
        saving_hook = SavingHook(args.save_n_steps)
        hooks.append(saving_hook)

    # setup loss function
    loss_fn = get_loss_fn(args)

    # setup trainer
    if args.loss == "smooth":
        trainer = SmoothTrainer(
            args.modelpath,
            model,
            loss_fn,
            optimizer,
            train_loader,
            val_loader,
            checkpoint_interval=args.checkpoint_interval,
            keep_n_checkpoints=args.keep_n_checkpoints,
            hooks=hooks,
        )
    else:
        trainer = spk.train.Trainer(
            args.modelpath,
            model,
            loss_fn,
            optimizer,
            train_loader,
            val_loader,
            checkpoint_interval=args.checkpoint_interval,
            keep_n_checkpoints=args.keep_n_checkpoints,
            hooks=hooks,
        )
    return trainer


class SavingHook(Hook):
    def __init__(self, n_steps):
        self._n_steps = n_steps
        self._save_counter = 0
        self._epoch_counter = 0

    def on_train_begin(self, trainer):
        self._save_dir = os.path.join(trainer.model_path, "saved_models")
        os.makedirs(self._save_dir, exist_ok=True)

    def on_epoch_end(self, trainer):
        self._epoch_counter += 1
        if self._epoch_counter == self._n_steps:
            torch.save(
                trainer._model,
                os.path.join(self._save_dir, "model_{:05d}".format(self._save_counter)),
            )
            self._save_counter += 1
            self._epoch_counter = 0

class PatchingHook(Hook):
    def __init__(self):
        pass

    def on_validation_begin(self, trainer):
        for mod in trainer._model.modules():
            mod.dump_patches = True
