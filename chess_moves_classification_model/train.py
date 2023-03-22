import argparse
import os
import re
import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chess_moves_classification_model.loaders.moves import ChessMovesDataset
from chess_moves_classification_model.models.moves import ChessMovesModel


class Train:

    DEFAULT_MODEL_NAME = "model"
    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH_SIZE = 256

    def __init__(self, name: str, epochs: int, batch_size: int, cuda: bool = None):
        """
        Args:
            name (str): Model name to be saved
            epochs (int): For how many epochs train the model
            batch_size (int): Batch size for training
            cuda (bool): Whether to use CUDA to train. If unspecified then CUDA if available will be used
        """
        self._name = name or self.DEFAULT_MODEL_NAME
        self._epochs = epochs or self.DEFAULT_EPOCHS
        self._batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self._cuda = cuda if cuda is not None else torch.cuda.is_available()

        self._train_set = ChessMovesDataset(train=True)
        self._test_set = ChessMovesDataset(train=False)
        self._train_loader = DataLoader(self._train_set, batch_size=batch_size, shuffle=True)
        self._test_loader = DataLoader(self._test_set, batch_size=batch_size, shuffle=True)

        self.model = ChessMovesModel()
        if self._cuda:
            self.model = self.model.cuda()

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def run_training(self):
        """Runs training consisting of a given number of epochs.
        Each epoch consists of a training part and model validation."""
        for epoch in range(self._epochs):
            self._train(epoch)
            self._validate()

    def _train(self, epoch):
        """Performs one cycle of training on training dataset and writes out the results"""
        running_loss = 0.0
        running_items = 0
        epoch_str = "   "[:-len(str(epoch + 1))] + str(epoch + 1)
        self.model.train()
        for data in tqdm(self._train_loader, bar_format=(
                "Epoch: {}   ".format(epoch_str) + "{l_bar}{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]"),
                         unit='item', unit_scale=True, unit_divisor=1, position=0, leave=True):
            inputs, labels = data
            if self._cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            self._optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()
            running_loss += loss.item()
            running_items += len(labels)
        print('Train loss: %.6f' % (running_loss / running_items))

    def _validate(self):
        """Validates current model on validation data"""
        val_correct = 0
        val_loss = 0.
        total = 0
        self.model.eval()
        with torch.inference_mode():
            for data in self._test_loader:
                inputs, labels = data
                if self._cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = self.model(inputs)
                loss = self._criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()
                total += len(labels)
        val_loss_total = val_loss / total
        val_acc = 100.0 * val_correct / total
        print('Val loss:   %.6f, val accuracy: %.2f' % (val_loss_total, val_acc) + "%")
        time.sleep(0.1)  # To avoid wrong printing of progress

    def save(self):
        """Saves the trained model to a file"""
        model_number_regex = r"^{}".format(self._name) + r"(\d*).pt$"
        existing_models_numbers = [
            int(re.search(model_number_regex, model_name).group(1))
            for model_name in os.listdir("runs") if re.match(model_number_regex, model_name)]
        next_number = max(existing_models_numbers) if len(existing_models_numbers) > 0 else ""
        torch.save(self.model.state_dict(), f"chess_moves_classification_model/runs/{self._name}{next_number}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help=f"Name of saved model, default '{Train.DEFAULT_MODEL_NAME}'")
    parser.add_argument("--epochs", type=int, help=f"Number of epochs, default {Train.DEFAULT_EPOCHS}")
    parser.add_argument("--batch_size", type=int, help=f"Batch size, default {Train.DEFAULT_BATCH_SIZE}")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA, default use CUDA if available")
    opt = parser.parse_args()

    model_trainer = Train(opt.name, opt.epochs, opt.batch_size, opt.cuda)
    model_trainer.run_training()
    model_trainer.save()
