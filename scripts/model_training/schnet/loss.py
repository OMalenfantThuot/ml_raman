import schnetpack as spk


def get_loss_fn(args):
    derivative = spk.utils.get_derivative(args)
    contributions = spk.utils.get_contributions(args)
    stress = spk.utils.get_stress(args)
    if args.loss in [
        "default",
        "tilted_up",
        "tilted_down",
        "relative_loss",
        "smooth",
        "masked_norm",
    ]:
        loss = args.loss
    else:
        raise ValueError("The loss argument is not recognized.")
    if loss in ["default", "relative_loss", "smooth", "masked_norm"]:
        # simple loss function for training on property only
        if derivative is None and contributions is None and stress is None:
            from utils.functions.schnet_loss import simple_fn

            return simple_fn(args)

        # loss function with tradeoff weights
        if type(args.rho) == float:
            rho = {"property": args.rho, "derivative": 1 - args.rho}
            if loss == "smooth":
                rho["hessian"] = 0.0001
        else:
            rho = dict()
            rho["property"] = (
                1.0 if "property" not in args.rho.keys() else args.rho["property"]
            )
            if derivative is not None:
                rho["derivative"] = (
                    1.0
                    if "derivative" not in args.rho.keys()
                    else args.rho["derivative"]
                )
            if contributions is not None:
                rho["contributions"] = (
                    1.0
                    if "contributions" not in args.rho.keys()
                    else args.rho["contributions"]
                )
            if stress is not None:
                rho["stress"] = (
                    1.0 if "stress" not in args.rho.keys() else args.rho["stress"]
                )
            if loss == "smooth":
                rho["hessian"] = (
                    0.0001 if "hessian" not in args.rho.keys() else args.rho["hessian"]
                )
            # type cast of rho values
            for key in rho.keys():
                rho[key] = float(rho[key])
            # norm rho values
            norm = sum(rho.values())
            for key in rho.keys():
                rho[key] = rho[key] / norm

        if loss == "masked_norm":
            rho["max_norm"] = float(args.max_norm) if args.max_norm is not None else 5.5

        property_names = dict(
            property=args.property,
            derivative=derivative,
            contributions=contributions,
            stress=stress,
        )
        if loss == "default":
            from utils.functions.schnet_loss import tradeoff_loss_fn

            return tradeoff_loss_fn(rho, property_names)
        elif loss == "relative_loss":
            from utils.functions.schnet_loss import relative_loss

            return relative_loss(rho, property_names)
        elif loss == "smooth":
            from utils.functions.schnet_loss import smooth_loss

            return smooth_loss(rho, property_names)
        elif loss == "masked_norm":
            from utils.functions.schnet_loss import norm_mask_loss

            return norm_mask_loss(rho, property_names)

    elif loss == "tilted_down":
        if derivative is None and contributions is None and stress is None:
            from utils.functions.schnet_loss import tilted_down

            return tilted_down(args)
    elif loss == "tilted_up":
        if derivative is None and contributions is None and stress is None:
            from utils.functions.schnet_loss import tilted_up

            return tilted_up(args)
