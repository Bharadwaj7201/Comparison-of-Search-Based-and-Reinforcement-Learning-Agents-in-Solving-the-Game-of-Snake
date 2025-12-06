import pygame, random, math, heapq

WIDTH, HEIGHT = 720, 720
ROWS = COLS = 30
CELL = WIDTH // COLS
FPS = 11  # slower nostalgic feel

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Retro Snake AI — Search Based")
clock = pygame.time.Clock()

# Load sound + sprites
bg_music = pygame.mixer.Sound("assets/bg_music.wav")
bg_music.play(-1)
eat_sound = pygame.mixer.Sound("assets/eat.wav")
dead_sound = pygame.mixer.Sound("assets/game_over.wav")

snake_head = pygame.image.load("assets/snake_head.png")
snake_body = pygame.image.load("assets/snake_body.png")
apple = pygame.image.load("assets/apple.png")

# ======== Fonts ========
font = pygame.font.SysFont("Arial", 28, bold=True)

# ======== Retro Nokia style display ========
def draw_grid():
    screen.fill((0, 40, 0))   # old green monochrome style
    for x in range(0, WIDTH, CELL):
        pygame.draw.line(screen, (0, 70, 0), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL):
        pygame.draw.line(screen, (0, 70, 0), (0, y), (WIDTH, y))

def draw_snake(s):
    for i, (x, y) in enumerate(s):
        img = snake_head if i == 0 else snake_body
        screen.blit(img, (x * CELL, y * CELL))

def draw_food(f, t):
    sz = int(CELL * ((math.sin(t * 0.15) + 1) / 16 + 0.85))
    img = pygame.transform.scale(apple, (sz, sz))
    screen.blit(img, (f[0] * CELL + (CELL - sz) // 2, f[1] * CELL + (CELL - sz) // 2))

def draw_score(score):
    text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(text, (10, 10))  # top-left corner

def nb(n):
    x, y = n
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS:
            yield (nx, ny)

def h(a, b): 
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(st, gl, blk):
    pq = [(0, st, [st])]
    vis = set()
    while pq:
        cost, n, p = heapq.heappop(pq)
        if n == gl: return p
        if n in vis: continue
        vis.add(n)
        for nxt in nb(n):
            if nxt not in blk:
                heapq.heappush(pq, (cost + h(nxt, gl), nxt, p + [nxt]))
    return None

def safe_tail(s): 
    return astar(s[0], s[-1], set(s[:-1]))

def run():
    snake = [(15, 15), (14, 15), (13, 15)]
    food = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
    score = 0
    t = 0

    while True:
        clock.tick(FPS)
        t += 1
        path = astar(snake[0], food, set(snake[1:]))
        if not path: path = safe_tail(snake)
        if not path:
            for nx, ny in nb(snake[0]):
                if (nx, ny) not in snake:
                    snake.insert(0, (nx, ny))
                    snake.pop()
                    break
        else:
            snake.insert(0, path[1])

        if snake[0] == food:
            score += 1
            eat_sound.play()
            while food in snake:
                food = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
        else:
            snake.pop()

        if snake[0] in snake[1:] or not (0 <= snake[0][0] < COLS and 0 <= snake[0][1] < ROWS):
            dead_sound.play()

            # ===== Terminal Output =====
            print("===================================")
            print("            GAME OVER")
            print(f"          Final Score: {score}")
            print("===================================")

            # ===== On-Screen Output =====
            draw_grid()
            draw_snake(snake)
            draw_food(food, t)
            final_text = font.render(f"Final Score: {score}", True, (255, 255, 255))
            screen.blit(final_text, (WIDTH // 2 - final_text.get_width() // 2,
                                     HEIGHT // 2 - final_text.get_height() // 2))
            pygame.display.update()

            pygame.time.wait(3000)  # wait 3 seconds so player can see score
            pygame.quit()
            return

        draw_grid()
        draw_snake(snake)
        draw_food(food, t)
        draw_score(score)   # live score display
        pygame.display.update()

run()