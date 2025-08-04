import math
import pygame
import numpy as np

class LiDAR:
  def __init__(self, Range, uncertainity, screen):
    self.ray_angles = [-60, -30, -15, 0, 15, 30, 60]
    self.Range = Range
    self.W, self.H = screen.get_size()
    self.sigma = np.array([uncertainity[0], uncertainity[1]])
    self.position = (0,0)
    self.senseobstacle = []
    self.screen = screen
  
  def update_position(self, car_position):
    self.position = car_position
    
  def uncertainity_add(self, distance, angle, sigma):
    mean = np.array([distance, angle])
    covariance = np.diag(sigma ** 2)
    distance, angle = np.random.multivariate_normal(mean, covariance)
    distance = max(distance, 0)
    angle = max(angle, 0)
    return [distance, angle]
  
  def distance(self, obstacleposition):
    px = (obstacleposition[0] - self.position[0])**2
    py = (obstacleposition[1] - self.position[1])**2
    return math.sqrt(px + py)
  
  def sense_obstacle(self, car_angle):
    results = []
    x1, y1 = self.position[0], self.position[1]
    new_ray_angles = [math.radians(x + car_angle) for x in self.ray_angles]
    for angle in new_ray_angles:
      x2, y2 = (x1 + self.Range * math.cos(angle), y1 - self.Range * math.sin(angle))
      for i in range(0,100):
        u = i/100
        x = int(x2*u + x1 * (1-u))
        y = int(y2* u + y1 * (1-u))
        if 0<x<self.W and 0<y<self.H:
          color = self.screen.get_at((int(x), int(y)))
          if (color[0],color[1], color[2]) == (200,0,200):
            distance = self.distance((x,y))
            distance, noisy_angles = self.uncertainity_add(distance, angle, self.sigma)
            results.append((distance, noisy_angles))
            hit = True
            break
      if not hit:
        results.append((self.Range, angle))
    return results

