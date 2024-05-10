from schnetpack.train import Trainer
from copy import deepcopy
import torch
import sys


class SmoothTrainer(Trainer):
    def __init__(
        self,
        model_path,
        model,
        loss_fn,
        optimizer,
        train_loader,
        validation_loader,
        keep_n_checkpoints=3,
        checkpoint_interval=10,
        hooks=[],
    ):
        super().__init__(
            model_path=model_path,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            validation_loader=validation_loader,
            keep_n_checkpoints=keep_n_checkpoints,
            checkpoint_interval=checkpoint_interval,
            hooks=hooks,
        )

    def train(self, device, n_epochs=sys.maxsize):
        """Train the model for the given number of epochs on a specified device.

        Args:
            device (torch.torch.Device): device on which training takes place.
            n_epochs (int): number of training epochs.

        Note: Depending on the `hooks`, training can stop earlier than `n_epochs`.

        """
        self._model.to(device)
        self._optimizer_to(device)
        self._stop = False

        for h in self.hooks:
            h.on_train_begin(self)

        try:
            for _ in range(n_epochs):
                # increase number of epochs by 1
                self.epoch += 1

                for h in self.hooks:
                    h.on_epoch_begin(self)

                if self._stop:
                    # decrease self.epoch if training is aborted on epoch begin
                    self.epoch -= 1
                    break

                # perform training epoch
                train_iter = self.train_loader

                for train_batch in train_iter:
                    self.optimizer.zero_grad()

                    for h in self.hooks:
                        h.on_batch_begin(self, train_batch)

                    noise_batch = deepcopy(train_batch)
                    noise_batch["_positions"] += 0.01 * (
                        2 * torch.rand(noise_batch["_positions"].shape) - 1
                    )

                    # move input to gpu, if needed
                    train_batch = {k: v.to(device) for k, v in train_batch.items()}
                    noise_batch = {k: v.to(device) for k, v in noise_batch.items()}

                    result = self._model(train_batch)
                    noise_result = self._model(noise_batch)

                    loss = self.loss_fn(train_batch, result, noise_result)
                    loss.backward()
                    self.optimizer.step()
                    self.step += 1

                    for h in self.hooks:
                        h.on_batch_end(self, train_batch, result, loss)

                    if self._stop:
                        break

                if self.epoch % self.checkpoint_interval == 0:
                    self.store_checkpoint()

                # validation
                if self.epoch % self.validation_interval == 0 or self._stop:
                    for h in self.hooks:
                        h.on_validation_begin(self)

                    val_loss = 0.0
                    n_val = 0
                    for val_batch in self.validation_loader:
                        # append batch_size
                        vsize = list(val_batch.values())[0].size(0)
                        n_val += vsize

                        for h in self.hooks:
                            h.on_validation_batch_begin(self)

                        noise_val_batch = deepcopy(val_batch)
                        noise_val_batch["_positions"] += 0.01 * (
                            2 * torch.rand(noise_val_batch["_positions"].shape) - 1
                        )

                        # move input to gpu, if needed
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}
                        noise_val_batch = {
                            k: v.to(device) for k, v in noise_val_batch.items()
                        }

                        val_result = self._model(val_batch)
                        noise_val_result = self._model(noise_val_batch)

                        val_batch_loss = (
                            self.loss_fn(val_batch, val_result, noise_val_result)
                            .data.cpu()
                            .numpy()
                        )
                        if self.loss_is_normalized:
                            val_loss += val_batch_loss * vsize
                        else:
                            val_loss += val_batch_loss

                        for h in self.hooks:
                            h.on_validation_batch_end(self, val_batch, val_result)

                    # weighted average over batches
                    if self.loss_is_normalized:
                        val_loss /= n_val

                    if self.best_loss > val_loss:
                        self.best_loss = val_loss
                        torch.save(self._model, self.best_model)

                    for h in self.hooks:
                        h.on_validation_end(self, val_loss)

                for h in self.hooks:
                    h.on_epoch_end(self)

                if self._stop:
                    break
            #
            # Training Ends
            #
            # run hooks & store checkpoint
            for h in self.hooks:
                h.on_train_ends(self)
            self.store_checkpoint()

        except Exception as e:
            for h in self.hooks:
                h.on_train_failed(self)

            raise e
