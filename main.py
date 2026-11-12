import math
import os
import random
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pygame

# ------------------------------
# Configuration & Color Palette
# ------------------------------
WIDTH, HEIGHT = 1040, 700
FPS = 60

STAGE_RECT = pygame.Rect(40, 175, 960, 455)
PIP_RECT = pygame.Rect(STAGE_RECT.right - 185, STAGE_RECT.bottom - 140, 165, 124)

# Colors matched to the web UI
BG_DARK = (9, 13, 24)
STAGE_BG = (24, 24, 58)
STAGE_BORDER = (55, 65, 95)
CARD_BG = (11, 17, 34)
CARD_BORDER = (30, 41, 68)
MODAL_BG = (16, 22, 48)
MODAL_BORDER = (45, 58, 95)

LIME_GREEN = (163, 230, 53)
MUTED_GREEN = (101, 148, 56)
WHITE = (255, 255, 255)
SLATE_GRAY = (148, 163, 184)
DISABLED_GRAY = (108, 116, 134)
DARK_TEXT = (11, 19, 36)
DANGER_RED = (239, 68, 68)

FRUIT_PALETTE = [
    {"name": "watermelon", "color": (255, 77, 109), "rind": (74, 222, 128), "points": 1},
    {"name": "orange", "color": (255, 159, 28), "rind": (251, 146, 60), "points": 1},
    {"name": "apple", "color": (239, 35, 60), "rind": (254, 202, 202), "points": 1},
    {"name": "banana", "color": (250, 204, 21), "rind": (254, 240, 138), "points": 1},
    {"name": "pineapple", "color": (234, 179, 8), "rind": (163, 230, 53), "points": 2},
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BEST_SCORE_FILE = os.path.join(BASE_DIR, "best_score.txt")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")


def load_best_score():
    if os.path.exists(BEST_SCORE_FILE):
        try:
            with open(BEST_SCORE_FILE, "r") as f:
                return int(f.read().strip())
        except ValueError:
            return 0
    return 0


def save_best_score(score):
    try:
        with open(BEST_SCORE_FILE, "w") as f:
            f.write(str(score))
    except OSError:
        pass


def ensure_hand_model():
    """Downloads the MediaPipe Tasks hand_landmarker.task model on first run."""
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100_000:
        print("[Info] Downloading MediaPipe hand_landmarker.task model (~7.5 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[Info] Model download complete.")
    with open(MODEL_PATH, "rb") as f:
        return f.read()


# ------------------------------
# Threaded Webcam & Hand Tracker
# ------------------------------
class HandTrackerThread:
    def __init__(self):
        self.running = False
        self.starting = False
        self.camera_ready = False
        self.camera_error = False
        self.hand_visible = False
        self.finger_norm = None  # (norm_x, norm_y)
        self.latest_frame_surface = None
        self.lock = threading.Lock()
        self.cap = None
        self.thread = None

    def start(self):
        if self.running or self.starting:
            return
        self.starting = True
        self.camera_error = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open_camera(self):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
        for index in (0, 1):
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, _ = cap.read()
                    if ret:
                        return cap
                    cap.release()
        return None

    def _run(self):
        use_tasks_api = False
        landmarker = None
        legacy_hands = None

        # 1. Initialize MediaPipe (Supports Python 3.13+ Tasks API & older Legacy API)
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                legacy_hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    model_complexity=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
            else:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision

                model_bytes = ensure_hand_model()
                base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.6,
                    min_hand_presence_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
                landmarker = mp_vision.HandLandmarker.create_from_options(options)
                use_tasks_api = True
        except Exception as e:
            print(f"[Warning] MediaPipe initialization issue: {e}")

        # 2. Open Webcam
        self.cap = self._open_camera()
        if self.cap is None:
            print("[Error] Could not open webcam. Make sure no browser tab or other app is using it.")
            with self.lock:
                self.starting = False
                self.running = False
                self.camera_ready = False
                self.camera_error = True
            if legacy_hands:
                legacy_hands.close()
            if landmarker:
                landmarker.close()
            return

        with self.lock:
            self.starting = False
            self.camera_ready = True
            self.camera_error = False

        last_ts_ms = 0

        # 3. Frame Processing Loop
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Mirror frame horizontally
            frame = cv2.flip(frame, 1)
            rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            norm_pos = None
            try:
                if use_tasks_api and landmarker is not None:
                    ts_ms = int(time.monotonic() * 1000)
                    if ts_ms <= last_ts_ms:
                        ts_ms = last_ts_ms + 1
                    last_ts_ms = ts_ms

                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect_for_video(mp_image, ts_ms)
                    if result.hand_landmarks:
                        # Landmark 8 = Index fingertip
                        lm = result.hand_landmarks[0][8]
                        norm_pos = (lm.x, lm.y)
                elif legacy_hands is not None:
                    results = legacy_hands.process(rgb_frame)
                    if results.multi_hand_landmarks:
                        lm = results.multi_hand_landmarks[0].landmark[8]
                        norm_pos = (lm.x, lm.y)
            except Exception:
                pass

            # Resize RGB frame for Picture-in-Picture preview (165 x 124)
            pip_rgb = cv2.resize(rgb_frame, (PIP_RECT.width, PIP_RECT.height))
            pip_surface = pygame.surfarray.make_surface(np.transpose(pip_rgb, (1, 0, 2)))

            with self.lock:
                self.hand_visible = norm_pos is not None
                if norm_pos is not None:
                    self.finger_norm = norm_pos
                self.latest_frame_surface = pip_surface

        if self.cap:
            self.cap.release()
        if legacy_hands:
            legacy_hands.close()
        if landmarker:
            landmarker.close()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)


# ------------------------------
# Game Entities
# ------------------------------
class FruitEntity:
    def __init__(self):
        self.is_bomb = random.random() < 0.18
        if self.is_bomb:
            self.color = (45, 55, 72)
            self.rind = DANGER_RED
            self.points = 0
        else:
            spec = random.choice(FRUIT_PALETTE)
            self.color = spec["color"]
            self.rind = spec["rind"]
            self.points = spec["points"]

        margin = 80
        self.x = random.uniform(STAGE_RECT.left + margin, STAGE_RECT.right - margin)
        self.y = STAGE_RECT.bottom + 35

        target_x = random.uniform(STAGE_RECT.left + 180, STAGE_RECT.right - 180)
        self.vx = (target_x - self.x) * 0.015
        self.vy = -random.uniform(11.5, 14.8)
        self.radius = 30
        self.sliced = False

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.24  # Gravity

    def draw(self, surface):
        pos = (int(self.x), int(self.y))
        if self.is_bomb:
            pygame.draw.circle(surface, (30, 35, 45), pos, self.radius)
            pygame.draw.circle(surface, (90, 100, 120), (pos[0] - 8, pos[1] - 8), 7)
            fuse_end = (pos[0] + 18, pos[1] - 28)
            pygame.draw.line(surface, (200, 180, 140), (pos[0] + 8, pos[1] - 18), fuse_end, 4)
            pygame.draw.circle(surface, (250, 204, 21), fuse_end, 6)
            pygame.draw.circle(surface, DANGER_RED, fuse_end, 3)
        else:
            pygame.draw.circle(surface, self.rind, pos, self.radius)
            pygame.draw.circle(surface, self.color, pos, self.radius - 5)
            pygame.draw.circle(surface, WHITE, (pos[0] - 9, pos[1] - 9), 5)


class SplashParticle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, math.tau)
        speed = random.uniform(2.0, 7.5)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.radius = random.uniform(3, 7)
        self.life = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15
        self.life -= 0.03

    def draw(self, surface):
        if self.life > 0:
            r = max(1, int(self.radius * self.life))
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), r)


class FloatingText:
    def __init__(self, x, y, text, color):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.life = 1.0

    def update(self):
        self.y -= 1.5
        self.life -= 0.025

    def draw(self, surface, font):
        if self.life > 0:
            img = font.render(self.text, True, self.color)
            img.set_alpha(int(self.life * 255))
            surface.blit(img, (int(self.x - img.get_width() // 2), int(self.y)))


# ------------------------------
# Main Game Application
# ------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hand Gesture Fruit Ninja")
    clock = pygame.time.Clock()

    # Fonts
    font_title = pygame.font.SysFont("segoeui", 32, bold=True)
    font_modal_h2 = pygame.font.SysFont("segoeui", 28, bold=True)
    font_hud_val = pygame.font.SysFont("segoeui", 24, bold=True)
    font_btn = pygame.font.SysFont("segoeui", 15, bold=True)
    font_body = pygame.font.SysFont("segoeui", 15)
    font_small = pygame.font.SysFont("segoeui", 12, bold=True)

    tracker = HandTrackerThread()

    # State
    score = 0
    time_left = 60
    lives = 3
    best_score = load_best_score()

    game_running = False
    game_paused = False
    modal_title = "Ready to slice?"
    modal_subtext_1 = "Allow camera access, then move your index finger through"
    modal_subtext_2 = "the fruit. Keep your hand visible in the camera preview."
    start_btn_label = "Start game"

    fruits = []
    particles = []
    floating_texts = []
    blade_trail = []  # list of (x, y, timestamp)
    finger_pos = (-100, -100)

    last_spawn_time = 0.0
    last_timer_tick = 0.0

    # Ambient stars in stage
    stars = [
        (
            random.randint(STAGE_RECT.left + 10, STAGE_RECT.right - 10),
            random.randint(STAGE_RECT.top + 10, STAGE_RECT.bottom - 10),
            random.randint(1, 2),
        )
        for _ in range(30)
    ]

    # Interactive Button Rects
    btn_enable_cam = pygame.Rect(655, 22, 130, 38)
    btn_pause = pygame.Rect(795, 22, 85, 38)
    btn_restart = pygame.Rect(890, 22, 95, 38)
    btn_start_game = pygame.Rect(WIDTH // 2 - 65, STAGE_RECT.centery + 48, 130, 40)

    def reset_and_start_game():
        nonlocal score, time_left, lives, game_running, game_paused, last_timer_tick
        if not tracker.running and not tracker.starting:
            tracker.start()
        score = 0
        time_left = 60
        lives = 3
        fruits.clear()
        particles.clear()
        floating_texts.clear()
        game_running = True
        game_paused = False
        last_timer_tick = time.time()

    def trigger_game_over(title):
        nonlocal game_running, modal_title, modal_subtext_1, modal_subtext_2, start_btn_label
        game_running = False
        modal_title = title
        modal_subtext_1 = f"You scored {score} points! Keep your index finger visible"
        modal_subtext_2 = f"in the camera preview and try to beat your best of {best_score}."
        start_btn_label = "Play again"

    running = True
    while running:
        now = time.time()

        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_running:
                    game_paused = not game_paused
                    last_timer_tick = now

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if btn_enable_cam.collidepoint(mx, my):
                    tracker.start()
                elif btn_pause.collidepoint(mx, my) and game_running:
                    game_paused = not game_paused
                    last_timer_tick = now
                elif btn_restart.collidepoint(mx, my):
                    reset_and_start_game()
                elif not game_running and btn_start_game.collidepoint(mx, my):
                    reset_and_start_game()

            elif event.type == pygame.MOUSEMOTION:
                # Mouse fallback when hand isn't detected
                if not tracker.hand_visible and STAGE_RECT.collidepoint(event.pos):
                    finger_pos = event.pos
                    blade_trail.append((finger_pos[0], finger_pos[1], now))

        # 2. Update Hand Tracking Coordinates
        with tracker.lock:
            cam_starting = tracker.starting
            cam_ready = tracker.camera_ready
            cam_error = tracker.camera_error
            hand_vis = tracker.hand_visible
            norm_pos = tracker.finger_norm
            pip_surf = tracker.latest_frame_surface

        if hand_vis and norm_pos:
            fx = int(STAGE_RECT.left + norm_pos[0] * STAGE_RECT.width)
            fy = int(STAGE_RECT.top + norm_pos[1] * STAGE_RECT.height)
            finger_pos = (fx, fy)
            blade_trail.append((fx, fy, now))

        # Keep blade trail from last 0.18s
        blade_trail = [pt for pt in blade_trail if now - pt[2] < 0.18]

        # 3. Game Logic Update
        if game_running and not game_paused:
            if now - last_timer_tick >= 1.0:
                time_left -= 1
                last_timer_tick = now
                if time_left <= 0:
                    trigger_game_over("Time's up!")

            if now - last_spawn_time > 0.85:
                for _ in range(2 if random.random() < 0.35 else 1):
                    fruits.append(FruitEntity())
                last_spawn_time = now

            # Update fruits & check collisions
            for f in fruits[:]:
                f.update()

                # Check blade slice collision
                if STAGE_RECT.collidepoint(finger_pos):
                    dist = math.hypot(f.x - finger_pos[0], f.y - finger_pos[1])
                    if dist < f.radius + 14:
                        if f.is_bomb:
                            lives = max(0, lives - 1)
                            for _ in range(18):
                                particles.append(SplashParticle(f.x, f.y, DANGER_RED))
                            floating_texts.append(FloatingText(f.x, f.y, "-1 LIFE", DANGER_RED))
                            fruits.remove(f)
                            if lives <= 0:
                                trigger_game_over("Game Over!")
                            continue
                        else:
                            score += f.points
                            if score > best_score:
                                best_score = score
                                save_best_score(best_score)
                            for _ in range(16):
                                particles.append(SplashParticle(f.x, f.y, f.color))
                            floating_texts.append(FloatingText(f.x, f.y, f"+{f.points}", LIME_GREEN))
                            fruits.remove(f)
                            continue

                # Check missed fruit falling off bottom
                if f.y > STAGE_RECT.bottom + 50 and f.vy > 0:
                    if not f.is_bomb:
                        lives = max(0, lives - 1)
                        if lives <= 0:
                            trigger_game_over("Game Over!")
                    fruits.remove(f)

        for p in particles[:]:
            p.update()
            if p.life <= 0:
                particles.remove(p)

        for ft in floating_texts[:]:
            ft.update()
            if ft.life <= 0:
                floating_texts.remove(ft)

        # ------------------------------
        # 4. Rendering
        # ------------------------------
        screen.fill(BG_DARK)

        # Header Logo (Watermelon icon + HAND FRUIT NINJA)
        pygame.draw.circle(screen, (74, 222, 128), (58, 41), 16)
        pygame.draw.circle(screen, (255, 77, 109), (58, 41), 13)
        t_hand = font_title.render("HAND ", True, WHITE)
        t_fn = font_title.render("FRUIT NINJA", True, LIME_GREEN)
        screen.blit(t_hand, (84, 20))
        screen.blit(t_fn, (84 + t_hand.get_width(), 20))

        # Header Buttons
        cam_btn_color = MUTED_GREEN if cam_ready else LIME_GREEN
        if cam_starting:
            cam_btn_text = "Starting..."
        elif cam_ready:
            cam_btn_text = "Camera on"
        else:
            cam_btn_text = "Enable camera"
        pygame.draw.rect(screen, cam_btn_color, btn_enable_cam, border_radius=19)
        lbl_cam = font_btn.render(cam_btn_text, True, DARK_TEXT)
        screen.blit(lbl_cam, lbl_cam.get_rect(center=btn_enable_cam.center))

        pause_bg = WHITE if game_running else DISABLED_GRAY
        pause_label = "Resume" if game_paused else "Pause"
        pygame.draw.rect(screen, pause_bg, btn_pause, border_radius=19)
        lbl_pause = font_btn.render(pause_label, True, DARK_TEXT)
        screen.blit(lbl_pause, lbl_pause.get_rect(center=btn_pause.center))

        pygame.draw.rect(screen, WHITE, btn_restart, border_radius=19)
        lbl_restart = font_btn.render("Restart", True, DARK_TEXT)
        screen.blit(lbl_restart, lbl_restart.get_rect(center=btn_restart.center))

        # HUD Cards (SCORE, TIME, LIVES, BEST)
        hud_items = [("SCORE", score), ("TIME", time_left), ("LIVES", lives), ("BEST", best_score)]
        card_w = 228
        for i, (label, val) in enumerate(hud_items):
            crect = pygame.Rect(40 + i * (card_w + 16), 82, card_w, 72)
            pygame.draw.rect(screen, CARD_BG, crect, border_radius=12)
            pygame.draw.rect(screen, CARD_BORDER, crect, width=1, border_radius=12)

            lbl_s = font_small.render(label, True, SLATE_GRAY)
            val_s = font_hud_val.render(str(val), True, WHITE)
            screen.blit(lbl_s, lbl_s.get_rect(center=(crect.centerx, crect.top + 22)))
            screen.blit(val_s, val_s.get_rect(center=(crect.centerx, crect.top + 48)))

        # Stage Container (Clipped so fruits stay inside the rounded play area)
        pygame.draw.rect(screen, STAGE_BG, STAGE_RECT, border_radius=18)
        screen.set_clip(STAGE_RECT)

        for sx, sy, sr in stars:
            pygame.draw.circle(screen, (70, 80, 120), (sx, sy), sr)

        for f in fruits:
            f.draw(screen)

        for p in particles:
            p.draw(screen)

        for ft in floating_texts:
            ft.draw(screen, font_btn)

        # Glowing Blade Trail
        if len(blade_trail) > 1:
            for i in range(1, len(blade_trail)):
                p0 = (blade_trail[i - 1][0], blade_trail[i - 1][1])
                p1 = (blade_trail[i][0], blade_trail[i][1])
                width = max(2, int((i / len(blade_trail)) * 9))
                pygame.draw.line(screen, LIME_GREEN, p0, p1, width)

        if STAGE_RECT.collidepoint(finger_pos):
            pygame.draw.circle(screen, WHITE, finger_pos, 8)
            pygame.draw.circle(screen, LIME_GREEN, finger_pos, 10, width=2)

        # Camera Preview PIP
        if cam_ready and pip_surf is not None:
            badge_rect = pygame.Rect(PIP_RECT.right - 125, PIP_RECT.top - 28, 120, 22)
            pygame.draw.rect(screen, BG_DARK, badge_rect, border_radius=11)
            pygame.draw.rect(screen, STAGE_BORDER, badge_rect, width=1, border_radius=11)
            b_txt = font_small.render("CAMERA PREVIEW", True, SLATE_GRAY)
            screen.blit(b_txt, b_txt.get_rect(center=badge_rect.center))

            screen.blit(pip_surf, PIP_RECT.topleft)
            pygame.draw.rect(screen, WHITE, PIP_RECT, width=2, border_radius=8)

        # Center Modal Overlay (When game is not running)
        if not game_running:
            modal_rect = pygame.Rect(WIDTH // 2 - 220, STAGE_RECT.centery - 115, 440, 230)
            pygame.draw.rect(screen, MODAL_BG, modal_rect, border_radius=18)
            pygame.draw.rect(screen, MODAL_BORDER, modal_rect, width=1, border_radius=18)

            h2_surf = font_modal_h2.render(modal_title, True, WHITE)
            p1_surf = font_body.render(modal_subtext_1, True, SLATE_GRAY)
            p2_surf = font_body.render(modal_subtext_2, True, SLATE_GRAY)

            if cam_starting:
                cam_status_str = "Starting camera..."
                cs_color = LIME_GREEN
            elif cam_ready:
                cam_status_str = "Camera active"
                cs_color = LIME_GREEN
            elif cam_error:
                cam_status_str = "Camera unavailable (Mouse mode active)"
                cs_color = DANGER_RED
            else:
                cam_status_str = "Camera not started"
                cs_color = SLATE_GRAY

            cs_surf = font_small.render(cam_status_str, True, cs_color)

            screen.blit(h2_surf, h2_surf.get_rect(center=(WIDTH // 2, modal_rect.top + 42)))
            screen.blit(p1_surf, p1_surf.get_rect(center=(WIDTH // 2, modal_rect.top + 85)))
            screen.blit(p2_surf, p2_surf.get_rect(center=(WIDTH // 2, modal_rect.top + 108)))
            screen.blit(cs_surf, cs_surf.get_rect(center=(WIDTH // 2, modal_rect.top + 142)))

            pygame.draw.rect(screen, LIME_GREEN, btn_start_game, border_radius=20)
            st_surf = font_btn.render(start_btn_label, True, DARK_TEXT)
            screen.blit(st_surf, st_surf.get_rect(center=btn_start_game.center))

        screen.set_clip(None)
        pygame.draw.rect(screen, STAGE_BORDER, STAGE_RECT, width=1, border_radius=18)

        # Footer Status & Hint Bar
        if cam_error:
            status_msg = "Camera permission was denied. You can still play with your mouse."
            status_color = DANGER_RED
        elif cam_starting:
            status_msg = "Initializing camera & hand model..."
            status_color = LIME_GREEN
        elif not cam_ready:
            status_msg = "Enable the camera to activate hand tracking."
            status_color = LIME_GREEN
        elif not hand_vis:
            status_msg = "Camera ready. Show your hand to the camera."
            status_color = LIME_GREEN
        else:
            status_msg = "Hand detected! Move your index finger to slice."
            status_color = LIME_GREEN

        f_left = font_body.render(status_msg, True, status_color)
        f_right = font_body.render(
            "Index fingertip = blade · Avoid bombs · Space to pause", True, SLATE_GRAY
        )
        screen.blit(f_left, (42, STAGE_RECT.bottom + 15))
        screen.blit(f_right, (STAGE_RECT.right - f_right.get_width() - 4, STAGE_RECT.bottom + 15))

        pygame.display.flip()
        clock.tick(FPS)

    tracker.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
