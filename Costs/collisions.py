def f_collision(x_batch):
    """
    Computes a collision penalty as the mean inverse square of pairwise distances.

    Args:
        x_batch: Tensor [B, 3]

    Returns:
        Mean collision penalty.
    """
    diff = x_batch.unsqueeze(1) - x_batch.unsqueeze(0)  # [B, B, 3]
    dist_sq = torch.sum(diff ** 2, dim=-1)               # [B, B]
    mask = ~torch.eye(dist_sq.size(0), dtype=torch.bool, device=dist_sq.device)
    epsilon = 1e-8  # To avoid division by zero.
    dist_sq_no_diag = dist_sq.masked_select(mask).view(dist_sq.size(0), -1)
    loss_matrix = 1.0 / (dist_sq_no_diag + epsilon)
    if loss_matrix.mean() > 0.03 :
      return 0
    else :
      return loss_matrix.mean()
