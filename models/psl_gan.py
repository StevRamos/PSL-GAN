# Standard library imports
import argparse
import os
import time
import math

#Third party libraries
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from tqdm import tqdm

# Local imports
from models.generator import Generator
from models.discriminator import Discriminator
from utils.psl_dataset import PSLDataset
from utils.utils_model import init_optimizer, save_weights
from losses.wgann import compute_gradient_penalty

class PSLGAN():
    def __init__(self, config, use_wandb):
        self.use_wandb = use_wandb
        self.config = config
        self.device = torch.device("cuda:" + (os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0") if torch.cuda.is_available() else "cpu")
        print(f'Using device {self.device}')

        self.dataset = PSLDataset(self.config.matrix_data)

        self.N, self.C, self.T, self.V = self.dataset.N, self.dataset.C, self.dataset.T, self.dataset.V
        self.n_classes = self.dataset.n_classes 

        self.generator, self.discriminator = self.init_model()


    def init_model(self):
        generator     = Generator(self.config.latent_dim, 2, self.n_classes, 
                                self.T, self.config.mlp_dim,  
                                device=self.device)
        discriminator = Discriminator(2, self.n_classes, 
                                    self.T, self.config.latent_dim, 
                                    device=self.device)

        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)

        if self.use_wandb:
            wandb.watch(generator, log="all")
            wandb.watch(discriminator, log="all")

        return generator, discriminator


    def train_step(self, train_dataloader, opt_g, opt_d, Tensor, LongTensor):
        self.generator.train()
        self.discriminator.train()

        avg_loss_d = []
        avg_loss_g = []

        for i, (imgs, labels) in enumerate(train_dataloader):
            real_imgs = Variable(imgs.type(Tensor))
            labels    = Variable(labels.type(LongTensor))

            self.discriminator.zero_grad()

            z = Variable(Tensor(np.random.normal(0, 1, (self.config.batch_size, self.config.latent_dim))))

            fake_imgs = self.generator(z, labels)

            real_validity = self.discriminator(real_imgs, labels)
            fake_validity = self.discriminator(fake_imgs, labels)

            gradient_penalty = compute_gradient_penalty(self.discriminator, 
                                                    real_imgs.data.to(self.device), 
                                                    fake_imgs.data.to(self.device), 
                                                    labels.data.to(self.device), 
                                                    Tensor, 
                                                    LongTensor, 
                                                    self.device)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.config.lambda_gp * gradient_penalty

            d_loss.backward()
            opt_d.step()

            self.generator.zero_grad()

            avg_loss_d.append(d_loss.item())

            if i % self.config.n_critic==0:

                fake_imgs = self.generator(z, labels)
                fake_validity = self.discriminator(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                opt_g.step()
                avg_loss_g.append(g_loss.item())

        avg_loss_d = np.mean(np.array(avg_loss_d))
        avg_loss_g = np.mean(np.array(avg_loss_g))

        return avg_loss_g, avg_loss_d 


    def train(self):
        if torch.cuda.is_available():
            print(f'Training in {torch.cuda.get_device_name(0)}')
        else:
            print('Training in CPU')

        if self.config.save_weights:
            if self.use_wandb:
                path_saved_weights = os.path.join(self.config.path_saved_weights, wandb.run.id)
            else:
                path_saved_weights = os.path.join(self.config.path_saved_weights, f'{self.config.weights_default}')
            try:
                os.mkdir(path_saved_weights)
            except OSError:
                pass

        params = {
            "batch_size": self.config.batch_size,
            "num_workers": 8
        }

        train_dataloader = torch.utils.data.DataLoader(self.dataset, 
                                                        **params, 
                                                        shuffle=True)

        opt_g, opt_d = init_optimizer(self.generator, 
                                    self.discriminator,
                                    self.config.lr)

        
        Tensor     = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        for epoch in tqdm(range(self.config.n_epochs)):
            loss_gen, loss_disc = self.train_step(train_dataloader, opt_g, opt_d, Tensor, LongTensor)
            
            metrics_log = {
                "epoch": epoch + 1,
                "loss_gen": loss_gen,
                "loss_disc": loss_disc
            }

            if self.config.save_weights and ((epoch + 1) % int(self.config.n_epochs/self.config.num_backups))==0:
                path_save_epoch = os.path.join(path_saved_weights, f'epoch_{epoch+1}')
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    pass
                save_weights(self.generator, self.discriminator, path_save_epoch, self.use_wandb)

            if self.use_wandb:
                wandb.log(metrics_log)

            print("Losses/Metrics")
            print('Epoch [{}/{}], Disc loss: {:.4f}'.format(epoch +1, 
                                    self.config.n_epochs, loss_disc))
            print('Epoch [{}/{}], Gen loss: {:.4f}'.format(epoch +1, 
                                    self.config.n_epochs, loss_gen))

        if self.use_wandb:
            wandb.finish()