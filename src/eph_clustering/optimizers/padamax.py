from typing import List, Mapping
import torch
from torch.optim import Adamax


def configure_optimizers(model):
    if "opt_params" in model.optimizer_params:
        if isinstance(model.optimizer_params["opt_params"], List):
            opt_params = []
            named_params = dict(model.named_parameters())
            for l in model.optimizer_params["opt_params"]:
                l = {**l}
                params = l.pop("params")
                model_params = []
                for p in params:
                    model_params.append(named_params[p])
                opt_params.append({**l, **{"params": model_params}})
        else:
            opt_params = {**model.optimizer_params["opt_params"]}
    else:
        opt_params = {}
    optimizer = (
        PAdamax(model.parameters(), **opt_params)
        if isinstance(opt_params, Mapping)
        else PAdamax(opt_params)
    )
    return [optimizer], []


def projection_simplex_sort_torch(v, z=1):
    """
    Vectorized simplex projection for Pytorch weight matrices.
    From http://www.mblondel.org/publications/mblondel-icpr2014.pdf.
    Parameters
    ----------
    v: torch.Tensor
        The input matrix to project.
    z: float
       desired sum of values after projection. Default: 1

    Returns
    -------

    """
    squeeze_after = False
    if len(v.shape) < 2:
        squeeze_after = True
        v = v[None]
    device = v.device
    n_features = v.shape[1]
    simplex_violations = ~(
        (v.max(-1).values <= 1.0)
        & (v.min(-1).values >= 0.0)
        & (torch.isclose(v.sum(-1), torch.tensor(1.0)))
    )
    simplex_violation_ixs = None
    if simplex_violations.float().sum() >= 1:
        simplex_violation_ixs = simplex_violations.nonzero()[:, 0]
        u = torch.sort(v[simplex_violation_ixs], dim=-1, descending=True).values
    else:
        u = torch.sort(v, dim=-1, descending=True).values
    cssv = torch.cumsum(u, dim=-1) - z

    ind = torch.arange(n_features, device=device) + 1
    cond = u - cssv / ind[None] > 0
    ind_r = ind.repeat([v.shape[0], 1])

    sel = cond.mul(torch.arange(ind.shape[0], device=device)[None] + 1).argmax(-1)
    rho = torch.gather(ind_r, dim=-1, index=sel[:, None]).float()
    theta = cssv.gather(dim=-1, index=sel[:, None]) / rho
    if simplex_violation_ixs is not None:
        w = v.clone()
        w[simplex_violation_ixs] = torch.clamp_min(v[simplex_violation_ixs] - theta, 0)
    else:
        w = torch.clamp_min(v - theta, 0)
    if squeeze_after:
        w = w.squeeze(0)
    return w


class PAdamax(Adamax):
    def __init__(self, params, z=1):
        assert z > 0
        self.z = z
        super(PAdamax, self).__init__(params)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = super(PAdamax, self).step(closure=closure)

        for group in self.param_groups:
            for p in group["params"]:
                if len(p.shape) == 2 and p.shape[0] == p.shape[1]:
                    p.masked_scatter_(
                        torch.tril(torch.ones_like(p)).to(torch.bool),
                        torch.ones_like(p) * -1e3,
                    )

                p.set_(projection_simplex_sort_torch(p, self.z))
                p.set_(p / p.sum(-1, keepdim=True))
                assert p.sum(-1).allclose(
                    torch.ones(p.shape[0], dtype=p.dtype, device=p.device), atol=1e-2
                ), p.sum(-1)
                assert p.min() >= 0
                assert p.max() <= 1
        return loss
