import argparse
import json
import re

import torch
from PIL import Image

from chess_moves_classification_model.loaders.moves import get_transform
from chess_moves_classification_model.models.moves import ChessMovesModel


class Classifier:

    """Classifies images containing handwritten notations of chess moves"""

    def __init__(self, model: str, cuda: bool = None):
        """
        Args:
            model (str): Path to model
            cuda (bool): If use cuda
        """
        self._model = ChessMovesModel()
        self._cuda = cuda if cuda is not None else torch.cuda.is_available()
        if self._cuda:
            self._model.load_state_dict(torch.load(model))
        else:
            self._model.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        self._model.eval()
        with open("data/labels_to_moves.json", "r") as f:
            self._labels_to_moves = json.load(f)
        self._transform = get_transform()

    def classify(self, img_path: str, save_results: bool = False) -> str:
        """Classifies image containing handwritten notations of chess moves

        Args:
            img_path (str): Path to the classified image
            save_results (bool, optional): If save result, then create file with classified class

        Returns:
            str: The class into which the image was classified
        """
        img = Image.open(img_path)
        torch_img = self._transform(img)
        torch_img = torch_img.unsqueeze(0)
        if self._cuda:
            torch_img = torch_img.cuda()
        with torch.inference_mode():
            output = self._model(torch_img)
        _, predicted = torch.max(output.data, 1)
        predicted_move = self._labels_to_moves[int(predicted[0])]
        print(f"Image {image_path} classified as {predicted_move}")
        if save_results:
            ext = img_path.split(".")[-1]
            with open(re.sub(r"{}$".format(ext), "txt", img_path), "w") as f:
                f.write(predicted_move)
        return predicted_move


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--images_paths", nargs="+", required=True, help="Paths to images to classify")
    parser.add_argument("--save_results", action="store_true", help="If save results")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA, default use CUDA if available")
    opt = parser.parse_args()

    classifier = Classifier(opt.model, opt.cuda)
    for image_path in opt.images_paths:
        classifier.classify(image_path, opt.save_results)
