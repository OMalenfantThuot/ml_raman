import logging
import schnetpack as spk
import torch.nn as nn
from utils.models import DropoutSchNet, DropoutAtomwise
from schnetpack.utils import (
    get_derivative,
    get_negative_dr,
    get_contributions,
    get_stress,
)


def get_model(args, train_loader, mean, stddev, atomref, logging=None):
    if args.mode == "train":
        if logging:
            logging.info("building model...")
        if args.dropout == 0 and args.n_layers == 2:
            from schnetpack.utils import get_representation, get_output_module

            representation = get_representation(args, train_loader)
            output_module = get_output_module(
                args,
                representation=representation,
                mean=mean,
                stddev=stddev,
                atomref=atomref,
            )
        else:
            from schnetpack.utils import get_representation #get_output_module# get_representation

            representation = get_representation(args, train_loader)
#            representation = get_rep_with_dropout(args, train_loader)
#            output_module = get_output_module(
            output_module = get_outmod_with_dropout(
                args,
                representation=representation,
                mean=mean,
                stddev=stddev,
                atomref=atomref,
            )
        model = spk.AtomisticModel(representation, [output_module])

        if args.parallel:
            model = nn.DataParallel(model)
        if logging:
            logging.info(
                "The model you built has: %d parameters" % spk.utils.count_params(model)
            )
        return model
    else:
        raise spk.utils.ScriptError("Invalid mode selected: {}".format(args.mode))


def get_rep_with_dropout(args, train_loader):
    cutoff_network = spk.nn.cutoff.get_cutoff_by_string(args.cutoff_function)
    return DropoutSchNet(
        n_atom_basis=args.features,
        n_filters=args.features,
        n_interactions=args.interactions,
        cutoff=args.cutoff,
        n_gaussians=args.num_gaussians,
        cutoff_network=cutoff_network,
        dropout=args.dropout,
    )


def get_outmod_with_dropout(args, representation, mean, stddev, atomref):
    derivative = spk.utils.get_derivative(args)
    negative_dr = spk.utils.get_negative_dr(args)
    contributions = spk.utils.get_contributions(args)
    stress = spk.utils.get_stress(args)

    return DropoutAtomwise(
        args.features,
        aggregation_mode=spk.utils.get_pooling_mode(args),
        mean=mean[args.property],
        stddev=stddev[args.property],
        atomref=atomref[args.property],
        property=args.property,
        derivative=derivative,
        negative_dr=negative_dr,
        contributions=contributions,
        stress=stress,
        dropout=args.dropout,
        n_layers=args.n_layers,
    )
