#! /usr/bin/env python

import os
import torch
import argparse
from mlcalcdriver.calculators.schnetpack import load_model


class Splitter:
    def __init__(self, path, save=False):
        self.path = path
        self.save = save

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if isinstance(path, str):
            self._path = path
        else:
            raise TypeError("The path should be given as a str.")

    @property
    def save(self):
        return self._save

    @save.setter
    def save(self, save):
        self._save = save

    def split(self):
        try:
            model = load_model(self.path, device="cpu")
        except:
            model = load_model(os.environ["MODELDIR"] + self.path, device="cpu")
        self.representation = model.representation
        self.output_modules = model.output_modules
        if self.save:
            torch.save(self.representation, "representation_file")


def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("path", help="Path to the model to split.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    sp = Splitter(path=args.path, save=True)
    sp.split()
