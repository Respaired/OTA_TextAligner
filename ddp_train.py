import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
import os.path as osp
import re
import sys
import yaml
import shutil
from utils import *
from optimizers import build_optimizer
from model import *
from meldataset import build_dataloader
from utils import *
from torch.utils.tensorboard import SummaryWriter
import click

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)


import logging
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="DEBUG")

# torch.autograd.detect_anomaly(True)
torch.backends.cudnn.benchmark = True


def log_print(message, logger):
    logger.info(message)
    print(message)

@click.command()
@click.option('-p', '--config_path', default='./Configs/config.yml', type=str)
def main(config_path):

  config = yaml.safe_load(open(config_path))
  log_dir = config['log_dir']
  if not osp.exists(log_dir): os.mkdir(log_dir)
  shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))

  writer = SummaryWriter(log_dir + "/tensorboard")
  
  ddp_kwargs = DistributedDataParallelKwargs()
  accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])    
  if accelerator.is_main_process:
      writer = SummaryWriter(log_dir + "/tensorboard")


  # write logs
  file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
  logger.logger.addHandler(file_handler)

  epoch = config.get('epoch', 100)
  save_iter = 1
  batch_size = config.get('batch_size', 4)
  log_interval = 10
  device = accelerator.device
  train_path = config.get('train_data', None)
  val_path = config.get('val_data', None)
  epochs = config.get('epochs', 1000)

  train_list, val_list = get_data_path_list(train_path, val_path)

  train_dataloader = build_dataloader(train_list,
                                      batch_size=batch_size,
                                      num_workers=8,
                                      dataset_config=config.get('dataset_params', {}),
                                      device=device)

  val_dataloader = build_dataloader(val_list,
                                    batch_size=batch_size,
                                    validation=True,
                                    num_workers=2,
                                    device=device,
                                    dataset_config=config.get('dataset_params', {}))
  


  aligner = AlignerModel()
  forward_sum_loss = ForwardSumLoss()
  best_val_loss = float('inf')


  scheduler_params = {
          "max_lr": float(config['optimizer_params'].get('lr', 5e-4)),
          "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
          "epochs": epochs,
          "steps_per_epoch": len(train_dataloader),
      }


  optimizer, scheduler = build_optimizer(
      {"params": aligner.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})

  
  aligner, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
      aligner, optimizer, train_dataloader, val_dataloader, scheduler
  )

  with accelerator.main_process_first():
      if config.get('pretrained_model', '') != '':
          model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                      load_only_params=config.get('load_only_params', True))
      else:
          start_epoch = 0
          iters = 0
  
  
  # Training loop
  for epoch in range(1, epochs + 1):
      aligner.train()
      train_losses = []
      train_fwd_losses = []
      start_time = time.time()
      
      
      # Training phase
      pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{epochs} [Train]")
      for i, batch in enumerate(pbar):
          batch = [b.to(device) for b in batch]

          text_input, text_input_length, mel_input, mel_input_length, attn_prior = batch
          
          # Forward pass
          attn_soft, attn_logprob = aligner(spec=mel_input, 
                                            spec_len=mel_input_length, 
                                            text=text_input, 
                                            text_len=text_input_length,
                                            attn_prior=attn_prior)
          
          # Calculate loss
          loss = forward_sum_loss(attn_logprob=attn_logprob, 
                                  in_lens=text_input_length, 
                                  out_lens=mel_input_length)
  
          # Backward pass and optimization
          optimizer.zero_grad()
          accelerator.backward(loss)
          
          # Optional gradient clipping
          grad_norm = accelerator.clip_grad_norm_(aligner.parameters(), 5.0)
          
          optimizer.step()
          iters = iters + 1 

          if scheduler is not None:
              scheduler.step()
          

          if (i+1)%log_interval == 0 and accelerator.is_main_process:
              log_print('Epoch [%d/%d], Step [%d/%d], Forward Sum Loss: %.5f'
                      %(epoch+1, epochs, i+1, len(train_list)//batch_size, loss), logger)
              
              writer.add_scalar('train/Forward Sum Loss', loss, iters)
              # writer.add_scalar('train/d_loss', d_loss, iters)

              train_losses.append(loss.item())
              train_fwd_losses.append(loss.item())

              running_loss = 0
              
              accelerator.print('Time elasped:', time.time()-start_time)

      # Calculate average training loss for this epoch
      avg_train_loss = sum(train_losses) / len(train_losses)

      # Validation phase
      aligner.eval()
      val_losses = []

      with torch.no_grad():
          for batch in tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} [Val]"):
              batch = [b.to(device) for b in batch]
              
              text_input, text_input_length, mel_input, mel_input_length = batch
              
              # Forward pass
              attn_soft, attn_logprob = aligner(spec=mel_input, 
                                              spec_len=mel_input_length, 
                                              text=text_input, 
                                              text_len=text_input_length,
                                              attn_prior=None)
              
              # Calculate loss
              val_loss = forward_sum_loss(attn_logprob=attn_logprob, 
                                        in_lens=text_input_length, 
                                        out_lens=mel_input_length)
              
              val_losses.append(val_loss.item())
      
          # Calculate average validation loss
          avg_val_loss = sum(val_losses) / len(val_losses)
          
          # Log to TensorBoard
          writer.add_scalar('epoch/train_loss', avg_train_loss, epoch)
          writer.add_scalar('epoch/val_loss', avg_val_loss, epoch)
        
        # Save checkpoint every N epochs
          
      if (i+1)%save_iter == 0 and accelerator.is_main_process:

          print(f'Saving on step {epoch*len(train_dataloader)+i}...')
          state = {
              'net':  {key: aligner[key].state_dict() for key in aligner}, 
              'optimizer': optimizer.state_dict(),
              'iters': iters,
              'epoch': epoch,
          }
          save_path = os.path.join(log_dir, 'checkpoints', f'TextAligner_checkpoint_epoch_{epoch}.pt')
          torch.save(state, save_path)    
      # Print summary for this epoch
      epoch_time = time.time() - start_time
      accelerator.print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
      
      # # Plot and save attention matrices for visualization
      # if epoch % config.get('plot_every', 10) == 0:
      #     plot_attention_matrices(aligner, val_dataloader, device, 
      #                           os.path.join(log_dir, 'attention_plots', f'epoch_{epoch}'),
      #                           num_samples=4)
  
  writer.close()

if __name__=="__main__":
    main()