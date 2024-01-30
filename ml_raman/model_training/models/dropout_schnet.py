import schnetpack as spk
import torch.nn as nn
from schnetpack.nn.base import Dense
from schnetpack.nn.blocks import MLP
from schnetpack.nn.cfconv import CFConv
from schnetpack.representation import SchNet, SchNetInteraction
from schnetpack.nn.activations import shifted_softplus
from schnetpack.atomistic.output_modules import Atomwise


class DropoutSchNet(SchNet):
    def __init__(
        self,
        n_atom_basis=128,
        n_filters=128,
        n_interactions=3,
        cutoff=5.0,
        n_gaussians=25,
        cutoff_network=None,
        dropout=0.2,
    ):
        super().__init__(
            n_atom_basis=n_atom_basis,
            n_filters=n_filters,
            n_interactions=n_interactions,
            cutoff=cutoff,
            n_gaussians=n_gaussians,
            cutoff_network=cutoff_network,
        )
        self.interactions = nn.ModuleList(
            [
                DropoutSchNetInteraction(
                    n_atom_basis=n_atom_basis,
                    n_spatial_basis=n_gaussians,
                    n_filters=n_filters,
                    cutoff_network=cutoff_network,
                    cutoff=cutoff,
                    dropout=dropout,
                )
                for _ in range(n_interactions)
            ]
        )


class DropoutSchNetInteraction(SchNetInteraction):
    def __init__(
        self,
        n_atom_basis,
        n_spatial_basis,
        n_filters,
        cutoff_network,
        cutoff,
        dropout,
        normalize_filter=False,
    ):
        super().__init__(
            n_atom_basis=n_atom_basis,
            n_spatial_basis=n_spatial_basis,
            n_filters=n_filters,
            cutoff_network=cutoff_network,
            cutoff=cutoff,
            normalize_filter=normalize_filter,
        )
        self.filter_network = nn.Sequential(
            Dense(n_spatial_basis, n_filters, activation=shifted_softplus,),
            nn.Dropout(dropout),
            Dense(n_filters, n_filters),
        )
        self.dense = nn.Sequential(
            Dense(n_atom_basis, n_atom_basis, activation=shifted_softplus,),
            nn.Dropout(dropout),
            Dense(n_atom_basis, n_atom_basis, activation=None, bias=True),
        )
#        self.cfconv = DropoutCFConv(
#            n_in=n_atom_basis,
#            n_filters=n_filters,
#            n_out=n_atom_basis,
#            filter_network=self.filter_network,
#            cutoff_network=self.cutoff_network,
#            activation=shifted_softplus,
#            normalize_filter=normalize_filter,
#            dropout=dropout,
#        )


class DropoutCFConv(CFConv):
    def __init__(
        self,
        n_in,
        n_filters,
        n_out,
        filter_network,
        cutoff_network=None,
        activation=None,
        normalize_filter=False,
        axis=2,
        dropout=0.2,
    ):
        super().__init__(
            n_in,
            n_filters,
            n_out,
            filter_network,
            cutoff_network=cutoff_network,
            activation=activation,
            normalize_filter=normalize_filter,
            axis=axis,
        )
        self.in2f = nn.Sequential(
            Dense(n_in, n_filters, bias=False, activation=None), nn.Dropout(dropout)
        )
        self.f2out = nn.Sequential(
            Dense(n_filters, n_out, bias=True, activation=activation),
            nn.Dropout(dropout),
        )


class DropoutAtomwise(Atomwise):
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
        dropout=0.2,
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
        activation = shifted_softplus
        self.out_net = nn.Sequential(
            spk.nn.base.GetItem("representation"),
            DropoutMLP(n_in, n_out, n_layers, activation, dropout),
        )


class DropoutMLP(MLP):
    def __init__(self, n_in, n_out, n_layers, activation, dropout):
        super().__init__(n_in, n_out, n_layers=n_layers, activation=activation)

        c_neurons = n_in
        self.n_neurons = []
        for i in range(n_layers):
            self.n_neurons.append(c_neurons)
            c_neurons = c_neurons // 2
        self.n_neurons.append(n_out)

        layers = []
        for i in range(n_layers - 1):
            layers.append(
                Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            )
            layers.append(nn.Dropout(dropout))

        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        self.out_net = nn.Sequential(*layers)
