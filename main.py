import data
import data2
import loss
import torch
import model
from trainer import Trainer
from trainer2 import Trainer2

from option import args
# from args import args
import utils.utility as utility

ckpt = utility.checkpoint(args)

loader = data.Data(args)
loss1 = loss.Loss(args, ckpt) if not args.test_only and not args.extract_features_only else None

models = [model.Model(args, ckpt)]
losses = [loss1]
if args.mutual_learning:
    models.append(model.Model(args, ckpt))
    loss2 = loss.Loss(args, ckpt) if not args.test_only and not args.extract_features_only else None
    losses.append(loss2)


if args.two_datasets:
    loader2 = data2.Data2(args)
    trainer = Trainer2(args, models[0], losses[0], loader, loader2, ckpt)
else:
    trainer = Trainer(args, models, losses, loader, ckpt)
    
# CSCE 625: Process feature extraction option
if args.extract_features_only:
	trainer.save_features()
else:
	n = 0
	while not trainer.terminate():
		n += 1
		trainer.train()
		if args.test_every!=0 and n%args.test_every==0:
			trainer.test()
