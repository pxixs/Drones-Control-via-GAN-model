#the list obstacles is the list you can modify to put obstacles however you like here below lies just an example of obstacles that can be configurated
obstacles=[[(i-1)/2,0.4,(j-5)/10] for i in range(3) for j in range(10)]

def f_obstacle(x, obstacles):
    """
    Computes a obstacles penalty as the mean inverse  of pairwise distances.

    Args:
        x_batch: Tensor [B, 3]
        obstacles: Tensor [B,3]

    Returns:
        Mean obstacles penalty.
    """
    cost = 0
    batch_size = x.size(0)
    for obstacle in obstacles :
        if not isinstance(obstacle, torch.Tensor):
            obstacle_tensor = torch.tensor(obstacle, device=x.device, dtype=x.dtype)
        else:
            obstacle_tensor = obstacle

        for i in range(batch_size):
            # Compute the squared distance to the obstacle.
            Q = torch.norm(x[i] - obstacle_tensor)
            if Q < 0.2:
                cost += 1.0 / (max(Q-0.1, 0) + 1e-8)   # add a small epsilon to avoid division by zero.
    return cost / batch_size
