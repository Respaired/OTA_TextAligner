

import torch
import torch.nn.functional as F
from torch.nn import Module

class ForwardSumLoss(Module):
  
    def __init__(self, blank_logprob=-1, loss_scale=1.0):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True, blank=16)
        self.blank_logprob = blank_logprob
        self.loss_scale = loss_scale

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        max_key_len = attn_logprob.size(-1)

        # Reorder input to [query_len, batch_size, key_len]
        attn_logprob = attn_logprob.squeeze(1)
        attn_logprob = attn_logprob.permute(1, 0, 2)

        # Add blank label
        attn_logprob = F.pad(input=attn_logprob, pad=(1, 0, 0, 0, 0, 0), value=self.blank_logprob)

        # Convert to log probabilities
        # Note: Mask out probs beyond key_len
        key_inds = torch.arange(max_key_len + 1, device=attn_logprob.device, dtype=torch.long)
        attn_logprob.masked_fill_(key_inds.view(1, 1, -1) > key_lens.view(1, -1, 1), -1e15)  # key_inds >= key_lens+1
        attn_logprob = self.log_softmax(attn_logprob)

        # Target sequences
        target_seqs = key_inds[1:].unsqueeze(0)
        target_seqs = target_seqs.repeat(key_lens.numel(), 1)

        # Evaluate CTC loss
        cost = self.ctc_loss(attn_logprob, target_seqs, input_lengths=query_lens, target_lengths=key_lens)
        cost *= self.loss_scale

        return cost