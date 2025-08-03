
def generate_vertical_road_path():
    points = []
    y = 0
    x = width // 2  # start from center horizontally
    dx = 0

    for _ in range(height // 10 + 5):  # extra lines to support scrolling
        dx += random.uniform(-0.2, 0.2)
        dx = max(min(dx, 2.0), -2.0)
        x += dx * 5
        x = max(min(x, width - 100), 100)
        points.append((x, y))
        y += 10
    return points

def draw_vertical_road(screen, path, road_width):
    left_edge = []
    right_edge = []

    for i in range(len(path) - 1):
        (x1, y1) = path[i]
        (x2, y2) = path[i + 1]

        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            continue
        dx /= length
        dy /= length

        # Perpendicular vector
        px = -dy * road_width / 2
        py = dx * road_width / 2

        left_edge.append((x1 + px, y1 + py))
        right_edge.append((x1 - px, y1 - py))

    right_edge.reverse()
    road_shape = left_edge + right_edge
    pygame.draw.polygon(screen, gray, road_shape)