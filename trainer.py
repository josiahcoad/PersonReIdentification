import os
import torch
import numpy as np
import utils.utility as utility
from scipy.spatial.distance import cdist
from utils.functions import cmc, mean_ap
from utils.re_ranking import re_ranking
import pdb
import copy

class Trainer():
    def __init__(self, args, models, loss, loader, ckpt):
        self.args = args
        self.train_loader = loader.train_loader
        self.test_loader = loader.test_loader
        self.query_loader = loader.query_loader
        self.testset = loader.testset
        self.queryset = loader.queryset
        self.losses = []
        self.ckpt = ckpt
        self.loss = loss
        self.model = models[0]
        self.lr = 0.
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.device = torch.device('cpu' if args.cpu else 'cuda')

        # Configure second model for mutual learning
        if args.mutual_learning:
            self.model2 = models[1]
            self.optimizer2 = utility.make_optimizer(args, self.model2)
        
        if args.load != '':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckpt.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckpt.log)*args.test_every): self.scheduler.step()

    def train(self):
        self.loss.step()
        # ----------------------------------------------------
        # CSCE 625: Code to switch out the loss function after half way through epochs
        # (only effective if using mixed loss)
        epoch = self.scheduler.last_epoch + 1
        if epoch % self.args.switch_loss_every == 0:
            self.loss.swap_mixed_loss()
        try:
            lr = self.scheduler.get_lr()[0]
        except:
            lr = self.lr
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.loss.start_log()
        self.model.train()
        total_loss = 0
        for batch, (inputs, labels) in enumerate(self.train_loader): 
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # outputs in the market dataset is a tuple of 13 tensor of the following sizes:
            # 1: 32 X 2048 (concatonated features) (used for testing)
            # 4: 32 X 256 (used for backprop)
            # 8: 32 X 14718 (used for backprop)

            # labels is a vector<int> of indices (shape 32 (batchsize) ?) of the images
            # what the output encoding for an image was and then tell the loss what the
            # output encoding should have been, given the labels. If the loss knows the 
            # ground truth encoding of the identity, then it can compare.

            # Feed images through the model and compute losses
            outputs = []
            outputs.append(self.model(inputs))
            if hasattr(self, 'model2'):
                outputs.append(self.model2(inputs))
                loss, loss2 = self.loss(outputs, labels)
                total_loss += loss2
            else:
                loss = self.loss(outputs, labels)
            
            total_loss += loss
            # Back prop
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            if hasattr(self, 'model2'):
                self.optimizer2.zero_grad()
                loss2.backward()
                self.optimizer2.step()

            self.ckpt.write_log('\r[INFO] [{}/{}]\t{}/{}\t{}'.format(
                epoch, self.args.epochs,
                batch + 1, len(self.train_loader),
                self.loss.display_loss(batch)), 
            end='' if batch+1 != len(self.train_loader) else '\n')
            
        if self.args.save_on_min_loss:
            if min(self.losses) == total_loss:
                torch.save(self.model.model.state_dict(), 'models/' + self.args.save + '_' + str(epoch) + '.pt')
        

        if self.args.decay_type == 'reduce_on_plateau':
            self.scheduler.step(total_loss)
        else:
            self.scheduler.step()
        self.loss.end_log(len(self.train_loader))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckpt.write_log('\n[INFO] Test:')
        self.model.eval()

        self.ckpt.add_log(torch.zeros(1, 5))
        # get the encodings/features for the query dataset
        # the result size for example in the market query dataset
        # is 3368 X 2048 since there are 3368 query photos
        qf = self.extract_feature(self.query_loader).numpy()
        # get the encodings/features for the query dataset
        # the result size for example in the market gallery dataset
        # is 15913 X 2048 since there are 15913 gallery photos
        gf = self.extract_feature(self.test_loader).numpy()

        if self.args.re_rank:
            q_g_dist = np.dot(qf, np.transpose(gf))
            q_q_dist = np.dot(qf, np.transpose(qf))
            g_g_dist = np.dot(gf, np.transpose(gf))
            # dist will end up being a 2D matrix of size #query_photos X #gallery_photos
            # in case of market, this is 3368 X 15913
            # thus, dist[0][0] is the distance between query_photo[0] to gallery_photo[0]
            dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        else:
            # each image is represented by a vector of size 2048 which are its "features/encoding"
            # distance is calculated here using euclidian distances between the cross product
            # of each query image with each gallery image
            dist = cdist(qf, gf)
        r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=True)
        m_ap = mean_ap(dist, epoch, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

        self.ckpt.log[-1, 0] = m_ap
        self.ckpt.log[-1, 1] = r[0]
        self.ckpt.log[-1, 2] = r[2]
        self.ckpt.log[-1, 3] = r[4]
        self.ckpt.log[-1, 4] = r[9]
        best = self.ckpt.log.max(0)
        self.ckpt.write_log(
            '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f} (Best: {:.4f} @epoch {})'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
            best[0][0],
            (best[1][0] + 1)*self.args.test_every
            )
        )
        if not self.args.test_only:
            self.ckpt.save(self, epoch, is_best=((best[1][0] + 1)*self.args.test_every == epoch))

    """
    CSCE 625 by Jeffrey Cordero

    Used to evaluate features from a dataset and save them as 
    a mat file along with the corresponding labels
    """
    def save_features(self):
        self.ckpt.write_log('\n[INFO] Saving Features')
        self.model.eval()

        # Generate feature matrices
        qf = self.extract_feature(self.query_loader).numpy()
        gf = self.extract_feature(self.test_loader).numpy()

        query_dict = {'names' : self.queryset.names, 'features' : qf}
        gallery_dict = {'names' : self.testset.names, 'features' : gf}

        # Save to output files
        utility.save_features(gallery_dict, query_dict,
            self.args.gallery_feature_file, self.args.query_feature_file)

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3)-1,-1,-1).long()  # N x C x H x W
        return inputs.index_select(3,inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for (inputs, labels) in loader:
            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i==1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
        return features

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
