import pygame
import random
import numpy as np

class SpaceInvaderGame:
    def __init__(self, headless = True):
        self.headless = headless
        pygame.init()
        
        if not self.headless:
            # Screen
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption('Space Invader')
            self.icon = pygame.image.load('spaceship64.png')
            pygame.display.set_icon(self.icon)
            # Background
            self.background = pygame.image.load('spaceBackground.webp')
            # Player
            self.player_img = pygame.image.load('spaceship64.png')
            # Bullet
            self.bullet_img = pygame.image.load('bullet24.png')
        
        else:
            self.screen = None
            self.background = None

        self.playerX = 370
        self.playerY = 480
        self.player_speed = 200
        self.playerX_change = 0

        self.bulletX = 0
        self.bulletY = self.playerY
        self.bullet_speed = 400
        self.bullet_state = 'ready'

        # Invaders
        self.num_invaders = 1
        if not self.headless:
          self.invader_img = [pygame.image.load('invader64.png') for _ in range(self.num_invaders)]
        self.invaderX = [random.randint(0, 736) for _ in range(self.num_invaders)]
        self.invaderY = [random.randint(20, 300) for _ in range(self.num_invaders)]
        self.invaderX_change = [120 for _ in range(self.num_invaders)]
        self.invaderY_change = [40 for _ in range(self.num_invaders)]

        # Score
        self.score_val = 0
        if not self.headless:
          self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.textX = 10
        self.textY = 10

        # Game Over
        if not self.headless:
          self.over_font = pygame.font.Font('freesansbold.ttf', 64)
        self.game_over = False
        self.GAME_OVER_Y = 450

        # Clock
        self.clock = pygame.time.Clock()

    def show_score(self):
        score = self.font.render('Score : ' + str(self.score_val), True, (255, 255, 255))
        self.screen.blit(score, (self.textX, self.textY))

    def game_over_text(self):
        over_text = self.over_font.render('GAME OVER', True, (255, 0, 0))
        self.screen.blit(over_text, (200, 250))

    def draw_player(self):
        self.screen.blit(self.player_img, (self.playerX, self.playerY))

    def draw_invader(self, x, y, i):
        self.screen.blit(self.invader_img[i], (x, y))

    def fire_bullet(self, x, y):
        self.bullet_state = 'fire'
        if not self.headless:
          self.screen.blit(self.bullet_img, (x + (64 - 24)//2, y))

    def is_collision(self,enemyX, enemyY, bulletX, bulletY):
      distance = ((enemyX - bulletX) ** 2 + (enemyY - bulletY) ** 2) ** 0.5
      return distance < 27

    def get_state(self):
        state = [
            self.playerX / 800,
            self.playerY / 600,
            self.bulletX / 800,
            self.bulletY / 600,
            1 if self.bullet_state == 'fire' else 0
        ]
        for i in range(self.num_invaders):
            state.append(self.invaderX[i] / 800)
            state.append(self.invaderY[i] / 600)
        return state

    def reset(self):
        self.__init__()
        return self.get_state()

    def step(self, action, render = False):
        prev_score = self.score_val
        if render:
            delta_time = self.clock.tick(60) / 1000
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.background, (0, 0))
            # Game Over Line
            pygame.draw.line(self.screen, (255, 0, 0), (0, self.GAME_OVER_Y), (800, self.GAME_OVER_Y), 2)
            # FPS Display
            fps_font = pygame.font.Font(None, 24)
            fps_text = fps_font.render(f"FPS: {int(self.clock.get_fps())}", True, (255, 255, 0))
            self.screen.blit(fps_text, (700, 10))

        elif not render:
            delta_time = 1/60

        # Handle action
        self.playerX_change = 0
        if action == 1:
            self.playerX_change = -self.player_speed
        elif action == 2:
            self.playerX_change = self.player_speed
        elif action == 3 and self.bullet_state == 'ready':
            self.bulletX = self.playerX
            self.bulletY = self.playerY
            self.fire_bullet(self.bulletX, self.bulletY)
        # action == 0 is no-op (stay still)

        # Player movement
        self.playerX += self.playerX_change * delta_time
        self.playerX = max(0, min(self.playerX, 736))

        # Invader movement and collision
        for i in range(self.num_invaders):

            self.invaderX[i] += self.invaderX_change[i] * delta_time
            if self.invaderX[i] <= 0:
                self.invaderX_change[i] = abs(self.invaderX_change[i])
                self.invaderY[i] += self.invaderY_change[i]
            elif self.invaderX[i] >= 736:
                self.invaderX_change[i] = -abs(self.invaderX_change[i])
                self.invaderY[i] += self.invaderY_change[i]
            if self.invaderY[i] > self.GAME_OVER_Y:
                self.game_over = True

            if self.is_collision(self.invaderX[i], self.invaderY[i], self.bulletX, self.bulletY):
                self.bulletY = self.playerY
                self.bullet_state = 'ready'
                self.score_val += 1
                self.invaderX[i] = random.randint(0, 736)
                self.invaderY[i] = random.randint(50, 300)
            
            if render:
              self.draw_invader(self.invaderX[i], self.invaderY[i], i)

        # Bullet movement
        if self.bulletY <= 0:
            self.bulletY = self.playerY
            self.bullet_state = 'ready'

        if self.bullet_state == 'fire':
            self.fire_bullet(self.bulletX, self.bulletY)
            self.bulletY -= self.bullet_speed * delta_time
        
        if render:
            self.draw_player()
            self.show_score()

        if self.game_over and render:
            self.game_over_text()
        
        if render:
            pygame.display.update()

        reward = 0.0

        # --- Small time penalty to encourage efficiency
        reward -= 0.02  # was 0.7

        # --- Game over penalty
        if self.game_over:
            reward -= 10  # was 100

        # --- Target: Closest invader in Y
        target_index = np.argmin([abs(inv_y - self.playerY) for inv_y in self.invaderY])
        target_x = self.invaderX[target_index]

        # --- Movement reward: align or approach
        if abs(self.playerX - target_x) < 10:
            reward += 1.0  # aligned
        elif (self.playerX < target_x and action == 2) or (self.playerX > target_x and action == 1):
            reward += 0.1  # moving toward
        elif action in [1, 2]:
            reward -= 0.05  # moving away

        # --- Optional bonus for acting (keep small)
        # reward += 0.02 if action in [1, 2] else 0

        # --- Fire reward
        if action == 3 and self.bullet_state == 'ready':
            if abs(self.playerX - target_x) < 10:
                reward += 0.5  # good fire
            else:
                reward -= 0.2  # bad fire

        # --- Collision reward
        for i in range(self.num_invaders):
            if self.is_collision(self.invaderX[i], self.invaderY[i], self.bulletX, self.bulletY):
                reward += 5.0  # successful hit

        # --- Invader too close penalty
        for i in range(self.num_invaders):
            distance_to_line = self.GAME_OVER_Y - self.invaderY[i]
            if 0 < distance_to_line < 60:
                reward -= 0.2  # proximity penalty


        return self.get_state(), reward, self.game_over, self.score_val
