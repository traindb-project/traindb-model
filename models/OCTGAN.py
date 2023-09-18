import logging
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from sdgym.synthesizers.utils import BGMTransformer
from octgan.networks import Generator, Discriminator
from octgan.utils import *

from TrainDBBaseModel import TrainDBModel, TrainDBSynopsisModel
import pandas as pd

LOGGER = logging.getLogger(__name__)

class OCTGAN(TrainDBSynopsisModel):

    def __init__(self,
                 embedding_dim=128,
                 gen_dim=(128, 128),
                 dis_dim=(128, 128),
                 lr=2e-3,
                 l2scale=1e-06,
                 batch_size=500,
                 epochs=300,
                 num_split=3):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.lr = lr 

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_split = num_split

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = BGMTransformer()
        self.columns = []

    def train(self, real_data, table_metadata):
        columns, categoricals = self.get_columns(real_data, table_metadata)
        real_data = real_data[columns]
        self.columns = columns

        self.transformer.fit(real_data.to_numpy(), categorical_columns=categoricals)
        model_data = self.transformer.transform(real_data.to_numpy())

        LOGGER.info("Training %s", self.__class__.__name__)
        self.fit(model_data, categoricals, ())

    def save(self, output_path):
        torch.save({
            'transformer': self.transformer,
            'generator': self.generator,
            'cond_generator': self.cond_generator,
            'columns': self.columns
        }, os.path.join(output_path, 'model.pth'))

    def load(self, input_path):
        saved_model = torch.load(os.path.join(input_path, 'model.pth'))
        self.transformer = saved_model['transformer']
        self.generator = saved_model['generator']
        self.cond_generator = saved_model['cond_generator']
        self.columns = saved_model['columns']

    def list_hyperparameters():
        hparams = []
        hparams.append(TrainDBModel.createHyperparameter('embedding_dim', 'int', '128', 'the size of the random sample passed to the generator'))
        hparams.append(TrainDBModel.createHyperparameter('gen_dim', 'tuple or list of ints', '(128, 128)', 'the size of the output samples for each one of the residuals'))
        hparams.append(TrainDBModel.createHyperparameter('dis_dim', 'tuple or list of ints', '(128, 128)', 'the size of the output samples for each one of the discriminator layers'))
        hparams.append(TrainDBModel.createHyperparameter('lr', 'float', '2e-3', 'the learning rate for the generator and discriminator'))
        hparams.append(TrainDBModel.createHyperparameter('l2scale', 'float', '1e-6', 'regularization term'))
        hparams.append(TrainDBModel.createHyperparameter('batch_size', 'int', '500', 'the number of samples to process in each step'))
        hparams.append(TrainDBModel.createHyperparameter('epochs', 'int', '300', 'the number of training epochs'))
        hparams.append(TrainDBModel.createHyperparameter('num_split', 'int', '3', 'the number of splits for the discriminator'))
        return hparams


    def synopsis(self, row_count):
        LOGGER.info("Synopsis Generating %s", self.__class__.__name__)
        sampled_data = self.sample(row_count)
        return pd.DataFrame(sampled_data, columns=self.columns)

    def odetime(self, num_split):
        return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True, device=self.device) for
                    i in range(1, num_split)]

    def fit(self, train_data, categorical_columns, ordinal_columns):

        if len(train_data) <= self.batch_size:
            self.batch_size = (len(train_data) // 10)*10

        data_sampler = Sampler(train_data, self.transformer.output_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        self.generator = Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            data_dim)
        self.generator = nn.DataParallel(self.generator).to(self.device)

        discriminator = Discriminator(
            data_dim + self.cond_generator.n_opt,
            self.dis_dim, self.num_split)
        discriminator = nn.DataParallel(discriminator).to(self.device)

        optimizerG = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=self.lr, betas=(0.5, 0.9))

        all_time = self.odetime(self.num_split)
        optimizerT = optim.Adam(
            all_time, lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = len(train_data) // self.batch_size

        iter = 0

        for i in range(self.epochs):
            print("epoch", i)
            for id_ in range(steps_per_epoch):

                iter += 1
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                real = torch.from_numpy(real.astype('float32')).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                ######## update discriminator #########
                y_fake = discriminator([fake_cat,all_time])
                y_real = discriminator([real_cat,all_time])

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, all_time, self.device, lambda_=10)                
                
                loss_d = loss_d + pen 
                optimizerD.zero_grad()
                optimizerT.zero_grad()

                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerT.step()

                # clipping time points t.
                with torch.no_grad():
                    for j in range(len(all_time)):
                        if j == 0:
                            start = 0 + 0.00001
                        else:
                            start = all_time[j - 1].item() + 0.00001

                        if j == len(all_time) - 1:
                            end = 1 - 0.00001
                        else:
                            end = all_time[j + 1].item() - 0.00001
                        all_time[j] = all_time[j].clamp_(min=start, max=end)

                ######### update generator ##########
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.transformer.output_info)

                if c1 is not None:
                    y_fake = discriminator([torch.cat([fakeact, c1], dim=1), all_time])
                else:
                    y_fake = discriminator([fakeact,all_time])

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.transformer.output_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

    def sample(self, n):

        self.generator.eval()
        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = apply_activate(fake, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data, None)
