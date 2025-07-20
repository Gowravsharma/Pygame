import pygame
import random

class SpaceInvaderGame:
    def __init__(self):
        pygame.init()

        # Screen
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Space Invader')
        self.icon = pygame.image.load('spaceship64.png')
        pygame.display.set_icon(self.icon)

        # Background
        self.background = pygame.image.load('spaceBackground.webp')

        # Player
        self.player_img = pygame.image.load('spaceship64.png')
        self.playerX = 370
        self.playerY = 480
        self.player_speed = 200
        self.playerX_change = 0

        # Bullet
        self.bullet_img = pygame.image.load('bullet24.png')
        self.bulletX = 0
        self.bulletY = self.playerY
        self.bullet_speed = 400
        self.bullet_state = 'ready'

        # Invaders
        self.num_invaders = 6
        self.invader_img = [pygame.image.load('invader64.png') for _ in range(self.num_invaders)]
        self.invaderX = [random.randint(0, 736) for _ in range(self.num_invaders)]
        self.invaderY = [random.randint(20, 300) for _ in range(self.num_invaders)]
        self.invaderX_change = [200 for _ in range(self.num_invaders)]
        self.invaderY_change = [40 for _ in range(self.num_invaders)]

        # Score
        self.score_val = 0
        self.font = pygame.font.Font('freesansbold.ttf', 32)
        self.textX = 10
        self.textY = 10

        # Game Over
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

    def step(self, action):
        prev_score = self.score_val
        delta_time = self.clock.tick(60) / 1000
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.background, (0, 0))

        # Game Over Line
        pygame.draw.line(self.screen, (255, 0, 0), (0, self.GAME_OVER_Y), (800, self.GAME_OVER_Y), 2)

        # FPS Display
        fps_font = pygame.font.Font(None, 24)
        fps_text = fps_font.render(f"FPS: {int(self.clock.get_fps())}", True, (255, 255, 0))
        self.screen.blit(fps_text, (700, 10))

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

            self.draw_invader(self.invaderX[i], self.invaderY[i], i)

        # Bullet movement
        if self.bulletY <= 0:
            self.bulletY = self.playerY
            self.bullet_state = 'ready'

        if self.bullet_state == 'fire':
            self.fire_bullet(self.bulletX, self.bulletY)
            self.bulletY -= self.bullet_speed * delta_time

        self.draw_player()
        self.show_score()

        if self.game_over:
            self.game_over_text()

        pygame.display.update()

        reward = 0

        if self.game_over:
            reward = -100
        elif self.score_val > prev_score:
            reward = +10
        elif self.bullet_state == 'ready' and action == 3:
            reward = -1  # penalty for useless firing
        elif action in [1, 2]:  # movement
            reward = +0.2  # small encouragement to move
        # Penalize invader coming close
        for i in range(self.num_invaders):
            distance_to_line = self.GAME_OVER_Y - self.invaderY[i]
            if 0 < distance_to_line < 60:
                reward -= 0.5

        return self.get_state(), reward, self.game_over, self.score_val
