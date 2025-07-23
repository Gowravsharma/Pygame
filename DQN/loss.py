import torch
from torch import nn
import torch.nn.functional as F

class weighted_MSE(nn.Module):
  def __init__(self, normalize = False):
    super().__init__()
    self.normalize = normalize
  
  #def forward(self, q_true, q_pred, episode_len, score):
  #    if self.normalize:
  #        # Add small epsilon to avoid division by zero or log(0)
  #        eps = 1e-6
  #        episode_len = torch.tensor(episode_len, dtype=torch.float32)
  #        score = torch.tensor(score, dtype=torch.float32)

          # Stabilized weight (penalize low score + long episode more)
  #        log_len = torch.log(episode_len + 1.0)
  #        score = torch.clamp(score, min=1.0)
  #        weight = log_len / score  # Higher when score is low and episode is long

  #       mse = torch.mean((q_pred - q_true) ** 2)
  #        return mse * weight
  #    else:
  #       return torch.mean((q_pred - q_true) ** 2)
  def forward(self, q_true, q_pred, episode_len=None, score=None):
      # Ensure q_pred and q_true are tensors
      if not torch.is_tensor(q_pred):
          q_pred = torch.tensor(q_pred, dtype=torch.float32)
      if not torch.is_tensor(q_true):
          q_true = torch.tensor(q_true, dtype=torch.float32)

      return F.mse_loss(q_pred, q_true)




