"""Implement client behavior, for single-client, ensemble and stuff."""
import torch

from .client_single import _ClientSingle
from .client_single_w_causal import _ClientSingle_w_causal

def Client(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.ensemble == 1:
        return _ClientSingle(args, setup)
    elif args.ensemble == 2 or args.ensemble == 3:
        print("With causal model")
        return _ClientSingle_w_causal(args, setup)
    else:
        raise NotImplementedError()


from .optimization_strategy import training_strategy
__all__ = ['Client', 'training_strategy']
