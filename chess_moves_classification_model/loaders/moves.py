import os
import re
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transform() -> transforms.Compose:
    """Creates image transformation to {-1., 1.} torch tensor - model input format.

    Returns:
         transforms.Compose: Image transformation to model input format.
    """
    return transforms.Compose([
        transforms.Resize((80, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)])


class ChessMovesDataset(Dataset):

    """Custom PyTorch dataset for loading images of different chess moves.

    Properties:
        labels_to_moves (Dict[int, str]): A dictionary mapping integer class labels to their corresponding move names.
    """

    FILES_PATH = "images"  # Path to directory with moves images sub folders.

    def __init__(self, train: bool = True, split: float = 0.75):
        """
        Args:
            train (bool, optional):
                If True, returns the training set; otherwise, returns the test set. Defaults to True.
            split (float, optional): TThe fraction of the dataset to use for training. Defaults to 0.75.
        """
        self._moves = os.listdir(self.FILES_PATH)
        self._moves_to_labels = {m: e for e, m in enumerate(sorted(self._moves))}
        self._moves_cnt = {m: len(os.listdir(self.files_path + "/" + m)) for m in self._moves}
        self.labels_to_moves = {v: k for k, v in self._moves_to_labels.items()}
        self._train = train
        self._split = split
        self._image_list = self._load_image_list()
        self._transform = get_transform()

    def _load_image_list(self) -> List[str]:
        """ Loads the list of image file paths for the specified split.

        Returns:
            List[str]: A list of image file paths.
        """
        return [
            self.FILES_PATH + "/{}/img_{}.png".format(m, n)
            for m in self._moves for n in self._get_image_number_range_for_move(m)]

    def _get_image_number_range_for_move(self, move: str) -> range:
        """ Returns the range of image numbers for the specified move.

        Args:
            move (str): Algebraic notation move.

        Returns:
            range: The number of images for the specified move.
        """
        n_moves = self._moves_cnt[move]
        return range(int(n_moves * self._split)) if self._train else range(int(n_moves * self._split), self.N_IMG)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The number of image files in the dataset.
        """
        return len(self._image_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns the idx-th item in the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the transformed image and its corresponding label.
        """
        img_path = self._image_list[idx]
        img = Image.open(img_path)
        cls = self._moves_to_labels[re.search(r"images/(.*)/img_", img_path).group(1)]
        return self._transform(img), torch.tensor(cls)
