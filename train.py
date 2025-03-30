import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from optimizers import build_optimizer

def train_aligner(config, accelerator, train_dataloader, val_dataloader, device, log_dir, epochs=100):
    # Create model
    aligner = AlignerModel().to(device)
    
    # Define loss function
    forward_sum_loss = ForwardSumLoss()
    
    # Setup optimizer
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 5e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    optimizer, scheduler = build_optimizer(
        {"params": aligner.parameters(), "optimizer_params":{}, "scheduler_params": scheduler_params})
    
    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create directories for model checkpoints
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Loss weights
    fwd_sum_loss_weight = config.get('fwd_sum_loss_weight', 1.0)
    
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
            loss.backward()
            
            # Optional gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(aligner.parameters(), config.get('grad_clip', 5.0))
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # Log to TensorBoard
            global_step = (epoch - 1) * len(train_dataloader) + i
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            writer.add_scalar('train/grad_norm', grad_norm, global_step)
            
            # Update progress bar
            train_losses.append(loss.item())
            train_fwd_losses.append(loss.item())
            
            # Update the progress bar description
            pbar.set_description(f"Epoch {epoch}/{epochs} [Train] Loss: {loss.item():.4f}")
        
        # Calculate average training loss for this epoch
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        aligner.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                batch = [b.to(device) for b in batch]
                
                text_input, text_input_length, mel_input, mel_input_length, attn_prior = batch
                
                # Forward pass
                attn_soft, attn_logprob = aligner(spec=mel_input, 
                                               spec_len=mel_input_length, 
                                               text=text_input, 
                                               text_len=text_input_length,
                                               attn_prior=attn_prior)
                
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
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': aligner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(log_dir, 'checkpoints', 'best_model.pt'))
        
        # Save checkpoint every N epochs
        if epoch % config.get('save_every', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': aligner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(log_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pt'))
        
        # Print summary for this epoch
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Plot and save attention matrices for visualization
        if epoch % config.get('plot_every', 10) == 0:
            plot_attention_matrices(aligner, val_dataloader, device, 
                                  os.path.join(log_dir, 'attention_plots', f'epoch_{epoch}'),
                                  num_samples=4)
    
    writer.close()
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return aligner

# Main execution
if __name__ == "__main__":

    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask
    
    # Assuming these variables are defined in your main script
    train_aligner(
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        log_dir=log_dir,
        epochs=epoch
    )