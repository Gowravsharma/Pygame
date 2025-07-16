import pygame
import random
# Initialize the pygame
pygame.init()

#create the screen
screen = pygame.display.set_mode((800,600))

# Background
background = pygame.image.load('spaceBackground.webp')
# Title & Icon
pygame.display.set_caption('spaceInvader')
icon = pygame.image.load('spaceship.png')
pygame.display.set_icon(icon)

#player
playerImg = pygame.image.load('spaceship64.png')
resized_player_img = pygame.transform.scale(playerImg, (64,64))
playerX = 370
playerY = 480
playerX_change = 0
playerY_change = 0

# Invader
invaderImg = pygame.image.load('invader64.png')
invaderX = random.randint(0,800)
invaderY = random.randint(50, 200)
invaderX_change = 0.2
invaderY_change = 40

# Bullet
bulletImg = pygame.image.load('bullet.png')
bullet_resized = pygame.transform.scale(bulletImg, (24,24))
bulletX = 0
bulletY = 480
bulletX_change = 0
bulletY_change = 1
bullet_state = 'ready' # Ready - you can't seee the bullet on the screen

def player(x,y):
  screen.blit(playerImg, (x, y))

def invader(x,y):
  screen.blit(invaderImg, (x, y))

def fire_bullet(x,y):
  global bullet_state
  bullet_state = 'fire'
  screen.blit(bullet_resized, (x+20, y+10))

# Game Loop
runnning = True
while runnning:
  screen.fill((0,0,0))  # RGB values
  #Background Image
  screen.blit(background, (0,0))

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      print('Quitting')
      runnning = False
    # if Keystroke is pressed check is it -> or <-
    if event.type == pygame.KEYDOWN:
      print('Key is pressed')
      if event.key == pygame.K_LEFT:
        print('Left arrow Key is pressed')
        playerX_change = -0.25
      if event.key == pygame.K_RIGHT:
        print('Right arrow Key is pressed')
        playerX_change = 0.25
      if event.key == pygame.K_UP:
        print('UP arrow Key is pressed')
        playerY_change = -0.25
      if event.key == pygame.K_DOWN:
        print('down arrow Key is pressed')
        playerY_change = 0.25
      if event.key == pygame.K_SPACE:
        if bullet_state == 'ready':
          print('Bullet appears')
          bulletX = playerX
          fire_bullet(bulletX, bulletY)

    if event.type == pygame.KEYUP:
      if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
        print('keystroke has been Released')
        playerX_change = 0
        #playerY_change = 0
  
  #checking boundary so that spaceship dosen't cross boundary
  playerX += playerX_change
  #playerY += playerY_change

  if playerX <= 0:
    playerX = 0
  elif playerX >= 736:
    playerX = 736
  
  #invaderX_changed = random.choice([0.3,-0.3])
  invaderX += invaderX_change

  if invaderX <= 0:
    invaderX_change = 0.1
    invaderY += invaderY_change
  elif invaderX >= 768:
    invaderX_change = -0.1
    invaderY += invaderY_change
  
  # Bullet Movement
  if bulletY <= 0:
    bulletY = 480
    bullet_state = 'ready'
  if bullet_state == 'fire':
    fire_bullet(bulletX, bulletY)
    bulletY -= bulletY_change

  player(playerX, playerY)
  invader(invaderX, invaderY)
  pygame.display.update()