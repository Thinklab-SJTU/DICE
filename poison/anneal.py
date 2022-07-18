"""General interface script to launch poisoning jobs."""

import torch

import datetime
import time
import os

import village

from dup_stdout_manager import DupStdoutFileManager

torch.backends.cudnn.benchmark = village.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(village.consts.SHARING_STRATEGY)

# Parse input arguments
args = village.options().parse_args()

if args.causal_config is not None:
    for f in args.causal_config:
        import sys
        sys.path.append("../")
        from utils.config import cfg_from_file
        cfg_from_file(f)

# 100% reproducibility?
if args.deterministic:
    village.utils.set_deterministic()


if __name__ == "__main__":
    if not os.path.exists(args.poison_path):
        os.mkdir(args.poison_path)
    with DupStdoutFileManager(os.path.join(args.poison_path, "poison.log")) as _:
        setup = village.utils.system_startup(args)
        model = village.Client(args, setup=setup)
        materials = village.Furnace(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
        #materials = village.Furnace(args, 1, model.defs.augmentations, setup=setup)
        forgemaster = village.Forgemaster(args, setup=setup)

        start_time = time.time()
        if args.pretrained or args.only_causal:
            print('Loading pretrained model...')
            stats_clean = None
        else:
            if os.path.exists(os.path.join(args.poison_path, "model.pth")) :
                model.model.load_state_dict(torch.load(os.path.join(args.poison_path, "model.pth")))
            else:
                stats_clean = model.train(materials, max_epoch=args.max_epoch)
                torch.save(model.model.state_dict(), os.path.join(args.poison_path, "model.pth"))
        train_time = time.time()

        if args.st4causal:
            model.causal_model.load_state_dict(torch.load(os.path.join(args.poison_path, "model.pth")))

        poison_delta = forgemaster.forge(model, materials)
        forge_time = time.time()

        # Export
        if args.save is not None:
            materials.export_poison(poison_delta, path=args.poison_path, mode=args.save)

        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print('---------------------------------------------------')
        print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
        print(f'--------------------------- forge time: {str(datetime.timedelta(seconds=forge_time - train_time))}')
        print('-------------Job finished.-------------------------')

