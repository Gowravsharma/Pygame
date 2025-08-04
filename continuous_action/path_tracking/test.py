import pygame
import math
import random
from car_physics import Car

# Initialize
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rally Racer")
clock = pygame.time.Clock()

# Colors
white = (255, 255, 255)
gray = (150, 150, 150)
green = (0, 154, 23)
earthy_green = (75, 105, 56)
earth_brown = (188, 129, 95)

def draw_start_line(path, num_blocks=10, block_size=5):
    if len(path) < 2:
        return
    (x1, y1), (x2, y2) = path[0], path[1]

    # Direction vector
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    dx /= length
    dy /= length

    # Perpendicular direction
    px = -dy
    py = dx

    # Draw blocks across the track width
    for i in range(num_blocks):
        offset = (i - num_blocks / 2) * block_size
        cx = x1 + px * offset
        cy = y1 + py * offset

        color = white if i % 2 == 0 else (0, 0, 0)
        rect = pygame.Rect(0, 0, block_size, int(block_size * 2))
        rect.center = (cx, cy)
        pygame.draw.rect(screen, color, rect)

# Generate looped waypoints
def generate_looped_waypoints(n, radius=250, center=(400, 300)):
    waypoints = []
    angle_step = 2 * math.pi / n
    for i in range(n):
        angle = i * angle_step
        r = radius + random.randint(-50, 50)
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        waypoints.append((x, y))
    waypoints += waypoints[:3]  # loop it
    return waypoints

# Catmull-Rom interpolation
def catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * ((2*p1[0]) + (-p0[0] + p2[0]) * t +
               (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
               (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
    y = 0.5 * ((2*p1[1]) + (-p0[1] + p2[1]) * t +
               (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
               (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
    return (x, y)

# Smooth path
def smooth_path(points, samples_per_segment=20):
    smooth = []
    for i in range(1, len(points) - 2):
        p0, p1, p2, p3 = points[i-1], points[i], points[i+1], points[i+2]
        for t in [j/samples_per_segment for j in range(samples_per_segment)]:
            smooth.append(catmull_rom(p0, p1, p2, p3, t))
    return smooth

# Draw road
def draw_road(path, width):
    left, right = [], []
    for i in range(len(path) - 1):
        (x1, y1), (x2, y2) = path[i], path[i + 1]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        dx /= length
        dy /= length
        px = -dy * width / 2
        py = dx * width / 2
        left.append((x1 + px, y1 + py))
        right.append((x1 - px, y1 - py))
    
    # road surface
    right_reversed = list(reversed(right))
    #right.reverse()
    road_poly = left + right_reversed
    pygame.draw.polygon(screen, gray, road_poly)

    # draw re line
    red = (150, 0,200)
    border_width = 3 

    if len(left) > 1:
        pygame.draw.lines(screen, red, False, left, border_width)
        pygame.draw.lines(screen, red, False, right, border_width)


# Load car
car_img = pygame.image.load(r'assets/car_top32.png')
car_img = pygame.transform.rotate(car_img, -90)  # Assume image faces up initially
#car_img = pygame.transform.scale(car_img, (30, 30))

# Generate track
waypoints = generate_looped_waypoints(25)
path = smooth_path(waypoints)

# Create car
car = Car(car_img, path[0])
car.position = pygame.Vector2(path[0])  # Ensure it's a Vector2
car.angle = -90  # Match rotation

# Main loop
running = True
while running:
    screen.fill(green)
    draw_road(path, width=25)
    draw_start_line(path)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        car.velocity = min(car.velocity + car.acceleration, car.max_velocity)
    if keys[pygame.K_DOWN]:
        car.velocity = max(car.velocity - car.acceleration, -car.max_velocity / 2)
    if keys[pygame.K_LEFT]:
        car.steering = -1
    elif keys[pygame.K_RIGHT]:
        car.steering = 1
    else:
        car.steering = 0

    car.update()

    # Keep car inside screen boundaries
    margin = 15
    car.position.x = max(margin, min(width - margin, car.position.x))
    car.position.y = max(margin, min(height - margin, car.position.y))

    car.draw(screen)
    pygame.display.update()
    clock.tick(60)

pygame.quit()
