import torch

# 自分のファイルの import
import utility
import data
# import model
from option import args

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
        loader = data.Data(args)        # Data loader
        # model = model.Model(args, checkpoint)

        checkpoint.done()