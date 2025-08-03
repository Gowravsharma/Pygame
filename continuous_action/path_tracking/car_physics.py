import pygame
import math

class Car:
    def __init__(self, image, start_pos):
        self.image = image
        self.position = pygame.Vector2(start_pos)
        self.velocity = 0
        self.angle = -90  # Car facing up
        self.steering = 0
        self.max_velocity = 5
        self.acceleration = 0.1
        self.friction = 0.05
        self.steering_speed = 3  # degrees/frame

    def update(self):
        # Update angle based on steering and velocity direction
        if self.velocity != 0:
            self.angle += self.steering * self.steering_speed * (self.velocity / self.max_velocity)

        # Move in the direction of the car's angle
        rad = math.radians(self.angle)
        direction = pygame.Vector2(math.cos(rad), math.sin(rad))
        self.position += direction * self.velocity

        # Apply friction
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.friction, 0)
        elif self.velocity < 0:
            self.velocity = min(self.velocity + self.friction, 0)

    def draw(self, surface):
        rotated_image = pygame.transform.rotate(self.image, -self.angle)  # negative because y axis is down
        rect = rotated_image.get_rect(center=self.position)
        surface.blit(rotated_image, rect.topleft)
