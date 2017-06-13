# Modified from http://www.pygame.org/project-Very+simple+Pong+game-816-.html
import pygame
from pygame.locals import *

score = 0


def run(screen_width=400., screen_height=400.):
    global score
    pygame.init()

    bar_width, bar_height = screen_width / 32., screen_height / 6.
    bar_dist_from_edge = screen_width / 64.
    circle_diameter = screen_height / 16.
    circle_radius = circle_diameter / 2.
    bar_1_start_x = bar_dist_from_edge
    bar_start_y = (screen_height - bar_height) / 2.
    bar_max_y = screen_height - bar_height - bar_dist_from_edge
    circle_start_x, circle_start_y = (screen_width - circle_diameter), (screen_width - circle_diameter) / 2.

    screen = pygame.display.set_mode((int(screen_width), int(screen_height)), 0, 32)

    # Creating 2 bars, a ball and background.
    back = pygame.Surface((int(screen_width), int(screen_height)))
    background = back.convert()
    background.fill((0, 0, 0))
    bar = pygame.Surface((int(bar_width), int(bar_height)))
    bar1 = bar.convert()
    bar1.fill((255, 255, 255))
    circle_surface = pygame.Surface((int(circle_diameter), int(circle_diameter)))
    pygame.draw.circle(circle_surface, (255, 255, 255), (int(circle_radius), int(circle_radius)), int(circle_radius))
    circle = circle_surface.convert()
    circle.set_colorkey((0, 0, 0))

    # some definitions
    bar1_x = bar_1_start_x
    bar1_y = bar_start_y
    circle_x, circle_y = circle_start_x, circle_start_y
    bar1_move, bar2_move = 0., 0.
    speed_x, speed_y, speed_bar = -screen_width / 1.28, screen_height / 1.92, screen_height * 1.2

    # clock and font objects
    clock = pygame.time.Clock()

    done = False
    while not done:
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    bar1_move = -ai_speed
                elif event.key == K_DOWN:
                    bar1_move = ai_speed
            elif event.type == KEYUP:
                if event.key == K_UP:
                    bar1_move = 0.
                elif event.key == K_DOWN:
                    bar1_move = 0.

        screen.blit(background, (0, 0))
        screen.blit(bar1, (bar1_x, bar1_y))
        screen.blit(circle, (circle_x, circle_y))

        bar1_y += bar1_move

        # movement of circle
        time_passed = clock.tick(30)
        time_sec = time_passed / 1000.0

        circle_x += speed_x * time_sec
        circle_y += speed_y * time_sec
        ai_speed = speed_bar * time_sec

        if bar1_y >= bar_max_y:
            bar1_y = bar_max_y
        elif bar1_y <= bar_dist_from_edge:
            bar1_y = bar_dist_from_edge

        if circle_x < bar_dist_from_edge + bar_width:
            if circle_y >= bar1_y - circle_radius and circle_y <= bar1_y + bar_height + circle_radius:
                circle_x = bar_dist_from_edge + bar_width
                speed_x = -speed_x
        if circle_x < -circle_radius:
            score -= 1
            circle_x, circle_y = circle_start_x, circle_start_y
            bar1_y, bar_2_y = bar_start_y, bar_start_y
        elif circle_x > screen_width - circle_diameter:
            score += 1
            speed_x = -speed_x
        if circle_y <= bar_dist_from_edge:
            speed_y = -speed_y
            circle_y = bar_dist_from_edge
        elif circle_y >= screen_height - circle_diameter - circle_radius:
            speed_y = -speed_y
            circle_y = screen_height - circle_diameter - circle_radius

        pygame.display.update()

    pygame.quit()


if __name__ == '__main__':
    run()
