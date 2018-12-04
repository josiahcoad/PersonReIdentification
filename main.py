import data
import data2
import loss
import torch
import model
from trainer import Trainer
from trainer2 import Trainer2
import loss2
from option import args
# from args import args
import utils.utility as utility

ckpt = utility.checkpoint(args)

loader = data.Data(args)
loss = loss.Loss(args, ckpt) if not args.test_only and not args.extract_features_only else None

models = [model.Model(args, ckpt)]
if args.mutual_learning:
    models.append(model.Model(args, ckpt))

if args.two_datasets:
    loader2 = data2.Data2(args)
    loss2 = loss2.Loss(args, ckpt) if not args.test_only and not args.extract_features_only else None
    trainer = Trainer2(args, models[0], loss2, loader, loader2, ckpt)
else:
    trainer = Trainer(args, models, loss, loader, ckpt)
    
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
