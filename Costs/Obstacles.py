'''
If you want to modify the positions of the obstacles you have to put them in the obstacles list as coordinates (x,y,z) 
'''
obstacles=[[(i-1)/2,0.4,(j-5)/10] for i in range(3) for j in range(10)]
def f_obstacle(x, obstacles):
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
