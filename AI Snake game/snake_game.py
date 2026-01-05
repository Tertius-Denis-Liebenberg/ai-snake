import os
import pygame
import random
from enum import Enum
from collections import namedtuple
import math
import time

pygame.init()

BLOCK_SIZE = 20
SPEED_BASE = 8
SCREEN_W, SCREEN_H = 650, 480

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
NEON_BLUE = (0, 180, 255)
NEON_GREEN = (0, 255, 120)
DARK_GRID = (15, 15, 25)
HUD_BG = (10, 10, 30)

INNER_OFFSET = 4
INNER_SIZE = BLOCK_SIZE - 2 * INNER_OFFSET

# Font
font = pygame.font.Font('Default.ttf', 20)

# ----- Sounds -----
try:
    EAT_SOUND = pygame.mixer.Sound("Sound Effects/eating.mp3")
    SPECIAL_FOOD_SOUND = pygame.mixer.Sound("Sound Effects/special_food.mp3")
    GAME_OVER_SOUND = pygame.mixer.Sound("Sound Effects/game_over.mp3")
except:
    EAT_SOUND = SPECIAL_FOOD_SOUND = GAME_OVER_SOUND = None

# ----- Helper Classes -----
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# ----- Snake Game -----
class AISnakeGame():

    def __init__(self, w=SCREEN_W, h=SCREEN_H):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.start_time = time.time()

        self.direction = Direction.RIGHT
        self.next_direction = self.direction

        # Initial head position
        start_x = (self.w // 2) // BLOCK_SIZE * BLOCK_SIZE
        start_y = (self.h // 2) // BLOCK_SIZE * BLOCK_SIZE
        self.head = Point(start_x, start_y)

        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]

        self.score = 0
        self.food = None
        self.special_food = None
        self.special_food_timer = 0
        
        self.paused = False
        self.high_score = self._load_high_score()

        self._place_food()

    # ----- Food placement -----
    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake and candidate != self.special_food:
                self.food = candidate
                break

    def _place_special_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate = Point(x, y)
            if candidate not in self.snake and candidate != self.food:
                self.special_food = candidate
                self.special_food_timer = 0
                
                if SPECIAL_FOOD_SOUND:
                    SPECIAL_FOOD_SOUND.play()

                break

    # ----- High Score -----
    def _load_high_score(self):
        if os.path.exists("highscore.txt"):
            with open("highscore.txt", "r") as f:
                return int(f.read())
        return 0

    def _save_high_score(self):
        with open("highscore.txt", "w") as f:
            f.write(str(self.high_score))

    # ----- Game Step -----
    def play_step(self):
        self._handle_events()
        game_over = False

        if not self.paused:
            self.direction = self.next_direction
            self._move(self.direction)
            self.snake.insert(0, self.head)

            # Check food
            if self.head == self.food:
                self.score += 1
                if EAT_SOUND:
                    EAT_SOUND.play()
                self._place_food()
            elif self.head == self.special_food:
                self.score += 3
                if EAT_SOUND:
                    EAT_SOUND.play()
                self.special_food = None
            else:
                self.snake.pop()

            if self._is_collision():
                if GAME_OVER_SOUND:
                    GAME_OVER_SOUND.play()
                game_over = True
                if self.score > self.high_score:
                    self.high_score = self.score
                    self._save_high_score()
                return game_over, self.score

            # Special food every 10 points
            if self.score > 0 and self.score % 10 == 0 and self.special_food is None:
                self._place_special_food()

            # Smooth speed scaling
            speed = SPEED_BASE + self.score * 0.5
            self.clock.tick(speed)

            # Timer for special food pulsing
            if self.special_food:
                self.special_food_timer += 1

        # Update UI
        self._update_ui()

        return game_over, self.score

    # ----- Move Snake -----
    def _move(self, direction):
        x, y = self.head.x, self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    # ----- Collision Detection -----
    def _is_collision(self):
        if self.head.x >= self.w or self.head.x < 0 or self.head.y >= self.h or self.head.y < 0:
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    # ----- Input Handling -----
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # Pause toggle
                    self.paused = not self.paused
                if not self.paused:
                    if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and self.direction != Direction.RIGHT:
                        self.next_direction = Direction.LEFT
                    elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and self.direction != Direction.LEFT:
                        self.next_direction = Direction.RIGHT
                    elif (event.key == pygame.K_UP or event.key == pygame.K_w) and self.direction != Direction.DOWN:
                        self.next_direction = Direction.UP
                    elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and self.direction != Direction.UP:
                        self.next_direction = Direction.DOWN

    def _get_elapsed_time(self):
        elapsed = int(time.time() - self.start_time)
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02}:{seconds:02}"
    
    def _draw_glow_rect(self, color, rect, glow_size=6, radius=6):
        for i in range(glow_size, 0, -1):
            alpha = int(20 * (i / glow_size))
            glow_surf = pygame.Surface(
                (rect.width + i*2, rect.height + i*2), pygame.SRCALPHA
            )
            pygame.draw.rect(
                glow_surf,
                (*color, alpha),
                glow_surf.get_rect(),
                border_radius=radius + i
            )
            self.display.blit(
                glow_surf,
                (rect.x - i, rect.y - i)
            )

        pygame.draw.rect(self.display, color, rect, border_radius=radius)

    # ----- Render -----
    def _update_ui(self):
        self.display.fill(BLACK)

        # Neon grid
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, DARK_GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, DARK_GRID, (0, y), (self.w, y))

        # Draw snake with tail fade
        length = len(self.snake)
        for i, pt in enumerate(self.snake):
            fade = int(255 * (1 - i / length))
            body_color = (0, fade, 180)

            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)

            # Glow for head
            if i == 0:
                self._draw_glow_rect(NEON_BLUE, rect, glow_size=8, radius=8)
            else:
                pygame.draw.rect(self.display, body_color, rect, border_radius=6)

        # Draw normal food
        food_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        self._draw_glow_rect((255, 80, 80), food_rect, glow_size=6, radius=10)

        # Draw special food (pulsing)
        if self.special_food:
            pulse = abs(math.sin(self.special_food_timer * 0.15))
            size = int(BLOCK_SIZE * (1 + pulse * 0.5))
            offset = (size - BLOCK_SIZE) // 2

            core_rect = pygame.Rect(
                self.special_food.x - offset,
                self.special_food.y - offset,
                size,
                size
            )

            self._draw_glow_rect(YELLOW, core_rect, glow_size=10, radius=12)

        # Score & High Score
        text = font.render(f"Score: {self.score}", True, CYAN)
        self.display.blit(text, [5, 5])
        text_hs = font.render(f"High Score: {self.high_score}", True, CYAN)
        self.display.blit(text_hs, [5, 30])

        # Timer
        elapsed_time = self._get_elapsed_time()
        text_timer = font.render(f"Time: {elapsed_time}", True, CYAN)
        self.display.blit(text_timer, [5, 55])

        # Pause Message
        if self.paused:
            pause_text = font.render("PAUSED - Press P to Resume", True, YELLOW)
            self.display.blit(pause_text, [self.w//2 - 120, self.h//2])
        
        pygame.display.flip()

# ----- Main Loop -----
if __name__ == '__main__':
    game = AISnakeGame()

    while True:
        game_over, score = game.play_step()
        if game_over:
            break

    print('Final Score', score)
    pygame.quit()
