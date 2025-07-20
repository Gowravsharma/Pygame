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
  def forward(self, q_true, q_pred, episode_len, score):
    if self.normalize:
        eps = 1e-6
        episode_len = torch.tensor(episode_len, dtype=torch.float32)
        score = torch.tensor(score, dtype=torch.float32)

        score = torch.clamp(score, min=1.0)
        log_len = torch.log(episode_len + 1.0)

        # Invert score to get high penalty for low scores
        # But also adjust for episode_len / score ratio
        weight = (log_len / score) + (1 / score)

        mse = torch.mean((q_pred - q_true) ** 2)
        return mse * weight
    else:
        return torch.mean((q_pred - q_true) ** 2)


