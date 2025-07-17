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
invaderImg = []
invaderX = []
invaderY = []
invaderX_change = []
invaderY_change = []
num_invader = 6

for i in range(num_invader):
  invaderImg.append(pygame.image.load('invader64.png'))
  invaderX.append(random.randint(0,800)) 
  invaderY.append(random.randint(50, 300))
  invaderX_change.append(0.2)
  invaderY_change.append(40)

# Bullet
bulletImg = pygame.image.load('bullet24.png')
#bullet_resized = pygame.transform.scale(bulletImg, (24,24))
bulletX = 0
bulletY = 480
bulletX_change = 0
bulletY_change = 1
bullet_state = 'ready' # Ready - you can't seee the bullet on the screen


# score
score_val = 0
font = pygame.font.Font('freesansbold.ttf',32)

textX = 10
textY = 10

# Game Over Test
over_font = pygame.font.Font('freesansbold.ttf', 64)

def show_score(x,y):
  score = font.render('Score : ' + str(score_val), True, (255,255,255))
  screen.blit(score, (x,y))

def game_over_text():
  over_text = over_font.render('Game Over', True, (255, 255, 255))
  screen.blit(over_text, (200, 250))

def player(x,y):
  screen.blit(playerImg, (x, y))

def invader(x,y, i):
  screen.blit(invaderImg[i], (x, y))

def fire_bullet(x,y):
  global bullet_state
  bullet_state = 'fire'
  screen.blit(bulletImg, (x+20, y))

def isCollision(enemyX, enemyY,bulletX, bulletY):
  # Using Eucledian distance
  distance = ((enemyX - bulletX)**2 + (enemyY-bulletY)**2)**(0.5)
  if distance < 27:
    return True
  else:
    return False


# Game Loop
runnning = True
while runnning:
  screen.fill((0,0,0))  # RGB values
  #Background Image
  screen.blit(background, (0,0))

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      print('Final score is: ', score_val)
      print('Quitting')
      runnning = False
    # if Keystroke is pressed check is it -> or <-
    if event.type == pygame.KEYDOWN:
      #print('Key is pressed')
      if event.key == pygame.K_LEFT:
        #print('Left arrow Key is pressed')
        playerX_change = -0.25
      if event.key == pygame.K_RIGHT:
        #print('Right arrow Key is pressed')
        playerX_change = 0.25
      if event.key == pygame.K_UP:
        #print('UP arrow Key is pressed')
        playerY_change = -0.25
      if event.key == pygame.K_DOWN:
        #print('down arrow Key is pressed')
        playerY_change = 0.25
      if event.key == pygame.K_SPACE:
        if bullet_state == 'ready':
          #print('Bullet appears')
          bulletX = playerX
          bulletY = playerY #-----------------
          fire_bullet(bulletX, bulletY)

    if event.type == pygame.KEYUP:
      if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
        #print('keystroke has been Released')
        playerX_change = 0
        #playerY_change = 0
  
  #checking boundary so that spaceship dosen't cross boundary
  playerX += playerX_change
  #playerY += playerY_change

  if playerX <= 0:
    playerX = 0
  elif playerX >= 736:
    playerX = 736
  
  # invader Movement
  for i in range(num_invader):

    # Game Over
    if invaderY[i] > 400:
      for j in range(num_invader):
        invaderY[j] = 2000
      game_over_text()
      break


    invaderX[i] += invaderX_change[i]
    if invaderX[i] <= 0:
      invaderX_change[i] = 0.2
      invaderY[i] += invaderY_change[i]
    elif invaderX[i] >= 768:
      invaderX_change[i] = -0.2
      invaderY[i] += invaderY_change[i]
    
    # Collision
    collision = isCollision(invaderX[i], invaderY[i], bulletX, bulletY)
    if collision:
      #print('collision Occured')
      bulletY = 480
      bullet_state = 'ready'
      score_val += 1
      invaderX[i] = random.randint(0,736)
      invaderY[i] = random.randint(50, 150)
    invader(invaderX[i], invaderY[i], i)
  # Bullet Movement
  if bulletY <= 0:
    bulletY = 480
    bullet_state = 'ready'
  if bullet_state == 'fire':
    fire_bullet(bulletX, bulletY)
    bulletY -= bulletY_change
  
  
  # Game end Line
  pygame.draw.line(screen, (255,0,0),(0,422,),(800, 422), 5)
  pygame.draw.line(screen, (255,255,0),(0,382,),(800, 382), 2)
  player(playerX, playerY)
  show_score(textX, textY)
  pygame.display.update()