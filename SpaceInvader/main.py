import pygame
import random

# Initialize the pygame
pygame.init()

# Create the screen
screen = pygame.display.set_mode((800, 600))

# Clock for frame rate management
clock = pygame.time.Clock()

# Background
background = pygame.image.load('spaceBackground.webp')
pygame.display.set_caption('spaceInvader')
icon = pygame.image.load('spaceship.png')
pygame.display.set_icon(icon)

# Player
playerImg = pygame.image.load('spaceship64.png')
playerX = 370
playerY = 480
player_speed = 300  # pixels per second
playerX_change = 0
playerY_change = 0


# Invaders
invaderImg = []
invaderX = []
invaderY = []
invaderX_change = []
invaderY_change = []
num_invader = 6

for i in range(num_invader):
    invaderImg.append(pygame.image.load('invader64.png'))
    x_spawn = random.randint(0,736)
    invaderX.append(x_spawn)
    invaderY.append(random.randint(20, 300))

    if x_spawn <= 50:
        invaderX_change.append(abs(150))
    if x_spawn <= 686:
        invaderX_change.append(-abs(150))
    else:
      invaderX_change.append(random.choice([-150, 150])) # pixels per second
    invaderY_change.append(30)

# Bullet
bulletImg = pygame.image.load('bullet24.png')
bulletX = 0
bulletY = 480
bullet_speed = 700  # pixels per second
bullet_state = 'ready'

# Score
score_val = 0
font = pygame.font.Font('freesansbold.ttf', 32)
textX = 10
textY = 10
over_font = pygame.font.Font('freesansbold.ttf', 64)

def show_score(x, y):
    score = font.render('Score : ' + str(score_val), True, (255, 255, 255))
    screen.blit(score, (x, y))

def game_over_text():
    over_text = over_font.render('Game Over', True, (255, 255, 255))
    screen.blit(over_text, (200, 250))

def player(x, y):
    screen.blit(playerImg, (x, y))

def invader(x, y, i):
    screen.blit(invaderImg[i], (x, y))

def fire_bullet(x, y):
    global bullet_state
    bullet_state = 'fire'
    screen.blit(bulletImg, (x + 20, y))

def isCollision(enemyX, enemyY, bulletX, bulletY):
    distance = ((enemyX - bulletX) ** 2 + (enemyY - bulletY) ** 2) ** 0.5
    return distance < 27

# Game Loop
running = True
while running:
    dt = clock.tick(60) / 1000  # Delta time in seconds
    fps = int(clock.get_fps())

    screen.fill((0, 0, 0))
    screen.blit(background, (0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            print('Final score is:', score_val)
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                playerX_change = -player_speed
            if event.key == pygame.K_RIGHT:
                playerX_change = player_speed
            if event.key == pygame.K_SPACE and bullet_state == 'ready':
                bulletX = playerX
                bulletY = playerY
                fire_bullet(bulletX, bulletY)
        if event.type == pygame.KEYUP:
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                playerX_change = 0

    # Update player position
    playerX += playerX_change * dt
    playerX = max(0, min(playerX, 736))

    # Update invaders
    for i in range(num_invader):
        if invaderY[i] > 400:
            for j in range(num_invader):
                invaderY[j] = 2000
            game_over_text()
            break

        invaderX[i] += invaderX_change[i] * dt
        if invaderX[i] <= 0:
            invaderX_change[i] = abs(invaderX_change[i])
            invaderY[i] += invaderY_change[i]
        elif invaderX[i] >= 736:
            invaderX_change[i] = -abs(invaderX_change[i])
            invaderY[i] += invaderY_change[i]

        # Collision
        if isCollision(invaderX[i], invaderY[i], bulletX, bulletY):
            bulletY = 480
            bullet_state = 'ready'
            score_val += 1
            invaderX[i] = random.randint(0, 736)
            invaderY[i] = random.randint(50, 150)

        invader(invaderX[i], invaderY[i], i)

    # Bullet movement
    if bullet_state == 'fire':
        fire_bullet(bulletX, bulletY)
        bulletY -= bullet_speed * dt
        if bulletY <= 0:
            bulletY = 480
            bullet_state = 'ready'

    # Game visuals
    pygame.draw.line(screen, (255, 0, 0), (0, 422), (800, 422), 5)
    pygame.draw.line(screen, (255, 255, 0), (0, 382), (800, 382), 2)

    fps_text = font.render(f"FPS: {fps}", True, (255,255,0))
    screen.blit(fps_text, (680,10))
    player(playerX, playerY)
    show_score(textX, textY)
    pygame.display.update()
