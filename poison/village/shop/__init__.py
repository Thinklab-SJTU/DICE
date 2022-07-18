"""Interface for poison recipes."""
from .forgemaster_untargeted import ForgemasterUntargeted
from .forgemaster_targeted import ForgemasterTargeted
from .forgemaster_explosion import ForgemasterExplosion
from .forgemaster_tensorclog import ForgemasterTensorclog
from .forgemaster_targeted_mc import ForgemasterTargeted_mc
from .forgemaster_targeted_w_causal import ForgemasterTargeted_w_causal
from .forgemaster_targeted_both_causal import ForgemasterTargeted_both_causal

import torch


def Forgemaster(args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
    """Implement Main interface."""
    if args.recipe == 'grad_explosion':
        return ForgemasterExplosion(args, setup)
    elif args.recipe == 'tensorclog':
        return ForgemasterTensorclog(args, setup)
    elif args.recipe == 'untargeted':
        return ForgemasterUntargeted(args, setup)
    elif args.recipe == 'targeted':
        if args.ensemble == 1:
            return ForgemasterTargeted(args, setup)
        elif args.ensemble == 2:
            print("with causal loss")
            return ForgemasterTargeted_w_causal(args, setup)
        elif args.ensemble == 3:
            print("with two causal loss")
            return ForgemasterTargeted_both_causal(args, setup)
    elif args.recipe == 'targeted_mc':
        return ForgemasterTargeted_mc(args, setup)
    else:
        raise NotImplementedError()


__all__ = ['Forgemaster']
