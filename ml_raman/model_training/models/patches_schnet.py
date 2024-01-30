import schnetpack as spk
import torch
import torch.nn as nn
from torch.autograd import grad
from schnetpack.atomistic import AtomisticModel
from schnetpack.nn.base import Dense, Aggregate
from schnetpack.nn.blocks import MLP
from schnetpack.nn.activations import shifted_softplus
from schnetpack.atomistic.output_modules import Atomwise
from schnetpack import Properties


class PatchesAtomisticModel(AtomisticModel):
    def forward(self, inputs):
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            raise NotImplementedError()

        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs


class PatchesAtomwise(Atomwise):
    def __init__(
        self,
        n_in,
        n_out=1,
        aggregation_mode="sum",
        n_layers=2,
        n_neurons=None,
        activation=shifted_softplus,
        property="y",
        contributions=None,
        derivative=None,
        negative_dr=False,
        stress=None,
        create_graph=True,
        mean=None,
        stddev=None,
        atomref=None,
        outnet=None,
    ):
        super().__init__(
            n_in,
            n_out=n_out,
            aggregation_mode=aggregation_mode,
            n_layers=n_layers,
            n_neurons=n_neurons,
            activation=activation,
            property=property,
            contributions=contributions,
            derivative=derivative,
            negative_dr=negative_dr,
            stress=stress,
            create_graph=create_graph,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
            outnet=outnet,
        )
        if aggregation_mode == "sum":
            self.atom_pool = OutputAggregate(axis=1, mean=False)
        elif aggregation_mode == "mean":
            self.atom_pool = OutputAggregate(axis=1, mean=True)
        else:
            raise NotImplementedError()

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Properties.Z]

        # run prediction
        yi = self.out_net(inputs)
        yi = self.standardize(yi)

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        y = self.atom_pool(yi)

        # collect results
        result = {self.property: y, "individual_" + self.property: yi.squeeze()}

        if self.contributions is not None:
            raise NotImplementedError()

        create_graph = True if self.training else self.create_graph

        if self.derivative is not None:
            sign = -1.0 if self.negative_dr else 1.0
            dy = grad(
                result[self.property],
                inputs[Properties.R],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=True,
            )[0]
            result[self.derivative] = sign * dy

        if self.stress is not None:
            raise NotImplementedError()
        return result


class OutputAggregate(Aggregate):
    def forward(self, inputs):
        # compute sum of input along axis
        y = torch.sum(inputs, self.axis)
        # compute average of input along axis
        if self.average:
            N = inputs.size(self.axis)
            y = y / N
        return y
