#!/usr/bin/env python
import os
import torch
import logging
import torch.nn as nn
import schnetpack as spk
from schnetpack.utils import (
    get_dataset,
    get_metrics,
    get_loaders,
    get_statistics,
    get_trainer,
    ScriptError,
    evaluate,
    setup_run,
    get_divide_by_atoms,
)
from schnetpack.utils.script_utils.settings import get_environment_provider
from schnetpack.utils.script_utils.parsing import build_parser
from schnetpack.utils.script_utils.model import get_output_module


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def main(args):

    # setup
    train_args = setup_run(args)
    device = torch.device("cuda" if args.cuda else "cpu")

    # get dataset
    environment_provider = get_environment_provider(train_args, device=device)
    dataset = get_dataset(train_args, environment_provider=environment_provider)

    # get dataloaders
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=dataset, split_path=split_path, logging=logging
    )

    # define metrics
    metrics = get_metrics(train_args)

    # train or evaluate
    if args.mode == "train":

        # get statistics
        atomref = dataset.get_atomref(args.property)
        mean, stddev = get_statistics(
            args=args,
            split_path=split_path,
            train_loader=train_loader,
            atomref=atomref,
            divide_by_atoms=get_divide_by_atoms(args),
            logging=logging,
        )

        # build model
        model = get_model(args, train_loader, mean, stddev, atomref, logging=logging)

        # build trainer
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)

        # run training
        trainer.train(device, n_epochs=args.n_epochs)
        logging.info("...training done!")

    else:
        raise ("Use the original SchnetPack script instead.")


def get_model(args, train_loader, mean, stddev, atomref, logging=logging):
    if logging:
        logging.info("building model...")
    representation = get_representation(args, train_loader)
    output_module = get_output_module(
        args, representation=representation, mean=mean, stddev=stddev, atomref=atomref,
    )
    model = spk.AtomisticModel(representation, [output_module])

    if args.parallel:
        model = nn.DataParallel(model)
    if logging:
        logging.info(
            "The model you built has: %d parameters" % spk.utils.count_params(model)
        )
    return model


def get_representation(args, train_loader):
    if args.rep is not None:
        try:
            representation = torch.load(args.rep)
        except:
            representation = torch.load(os.environ["MODELDIR"] + args.rep)
    else:
        raise ("Representation path should be given.")
    return representation


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from_json":
        args = spk.utils.read_from_json(args.json_path)
    main(args)
