import os
import torch
from tqdm import tqdm


class Trainer:

    def __init__(

        self,

        model,

        optimizer,

        scheduler,

        criterion,

        device

    ):

        self.model = model

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.criterion = criterion

        self.device = device

        self.best_acc = 0

        ############################################################
        # Mixed Precision
        ############################################################

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=torch.cuda.is_available()
        )

        ############################################################
        # Training History
        ############################################################

        self.history = {

            "train_loss": [],

            "train_acc": [],

            "val_loss": [],

            "val_acc": []

        }

    ############################################################

    def train_one_epoch(

        self,

        loader

    ):

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        loop = tqdm(loader)

        for eeg, label in loop:

            eeg = eeg.to(self.device)

            label = label.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            ####################################################
            # Mixed Precision Forward
            ####################################################

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):

                output = self.model(eeg)

                loss = self.criterion(

                    output,

                    label

                )

            ####################################################
            # Backward
            ####################################################

            self.scaler.scale(loss).backward()

            ####################################################
            # Gradient Clipping
            ####################################################

            self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(

                self.model.parameters(),

                max_norm=1.0

            )

            ####################################################
            # Optimizer
            ####################################################

            self.scaler.step(

                self.optimizer

            )

            self.scaler.update()

            ####################################################

            pred = output.argmax(1)

            correct += (pred == label).sum().item()

            total += label.size(0)

            running_loss += loss.item()

            current_loss = running_loss / (total / label.size(0))

            current_acc = 100 * correct / total

            current_lr = self.optimizer.param_groups[0]["lr"]

            loop.set_postfix(

                Loss=f"{current_loss:.4f}",

                Acc=f"{current_acc:.2f}",

                LR=f"{current_lr:.6f}"

            )

        if self.scheduler is not None:

            self.scheduler.step()

        epoch_loss = running_loss / len(loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    ############################################################

    @torch.no_grad()

    def validate(

        self,

        loader

    ):

        self.model.eval()

        running_loss = 0.0

        correct = 0

        total = 0

        for eeg, label in loader:

            eeg = eeg.to(self.device)

            label = label.to(self.device)

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):

                output = self.model(eeg)

                loss = self.criterion(

                    output,

                    label

                )

            pred = output.argmax(1)

            correct += (pred == label).sum().item()

            total += label.size(0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    ############################################################

    def fit(

        self,

        train_loader,

        val_loader,

        epochs,

        save_path

    ):

        os.makedirs(

            os.path.dirname(save_path),

            exist_ok=True

        )

        for epoch in range(epochs):

            train_loss, train_acc = self.train_one_epoch(

                train_loader

            )

            val_loss, val_acc = self.validate(

                val_loader

            )

            ####################################################
            # Save History
            ####################################################

            self.history["train_loss"].append(train_loss)

            self.history["train_acc"].append(train_acc)

            self.history["val_loss"].append(val_loss)

            self.history["val_acc"].append(val_acc)

            ####################################################

            print("\n" + "=" * 60)

            print(f"Epoch {epoch + 1}/{epochs}")

            print("=" * 60)

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Train Acc  : {train_acc:.2f}%")

            print(f"Val Loss   : {val_loss:.4f}")

            print(f"Val Acc    : {val_acc:.2f}%")

            print(f"Learning Rate : {self.optimizer.param_groups[0]['lr']:.6f}")

            ####################################################

            if val_acc > self.best_acc:

                self.best_acc = val_acc

                torch.save(

                    {

                        "epoch": epoch + 1,

                        "model_state_dict": self.model.state_dict(),

                        "optimizer_state_dict": self.optimizer.state_dict(),

                        "scheduler_state_dict": self.scheduler.state_dict()

                        if self.scheduler is not None

                        else None,

                        "best_accuracy": self.best_acc

                    },

                    save_path

                )

                print("\nBest model saved.")

        print("\n" + "=" * 60)

        print("Training Finished")

        print(f"Best Accuracy : {self.best_acc:.2f}%")

        print("=" * 60)

        return self.history
