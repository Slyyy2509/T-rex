import os
import random
import time
import numpy as np
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "q_table.npy")

GAME_URL = "https://wayou.github.io/t-rex-runner/"
TRAIN_MODE = False
LOAD_MODEL = True
SAVE_MODEL = True
EPISODES = 400
PLAY_EPISODES = 5

ALPHA = 0.12
GAMMA = 0.90
EPSILON_START = 0.30
EPSILON_END = 0.05

STEP_DELAY = 0.015
MAX_STEPS = 50000
TRAIN_START_SPEED = 8.6

# Actions
WAIT, JUMP, DUCK = 0, 1, 2
# Obstacle kinds
NONE, SMALL_CACTUS, WIDE_CACTUS, LOW_BIRD, HIGH_BIRD = range(5
)

CLEAR_REWARD = 15
DEATH_REWARD = -100
BAD_ACTION_REWARD = -2

# Q table: kind x phase x action
# phase: 0=far, 1=approaching, 2=immediate
def make_q_table():
    return {(kind, phase): [0.0, 0.0, 0.0]
            for kind in range(5) for phase in range(3)}

q_table = make_q_table()

STATE_JS = r"""() => {
  const runner = window.Runner && window.Runner.instance_;
  if (!runner) return {status: "NO_RUNNER"};
  if (runner.crashed) return {status: "DEAD"};
  if (!runner.playing) return {status: "NOT_PLAYING"};

  const dino = runner.tRex;
  const obstacles = (runner.horizon && runner.horizon.obstacles) || [];
  const speed = runner.currentSpeed || 6;

  if (!obstacles.length) {
    return {status: "OK", kind: 0, phase: 0, distance: 9999,
            width: 0, speed, type: "NONE", yPos: 0,
            jumping: !!dino.jumping, ducking: !!dino.ducking};
  }

  const o = obstacles[0];
  const distance = o.xPos - dino.xPos;
  const type = o.typeConfig ? o.typeConfig.type : "NONE";
  let kind = 1;
  if (type === "PTERODACTYL") kind = (o.yPos >= 90) ? 3 : 4;
  else if (o.width >= 40) kind = 2;

  const velocity = speed * 60;
  const timeToObstacle = distance / velocity;
  const jumpLead = 0.20 + 31 / velocity;
  const phase = timeToObstacle > jumpLead ? 0 :
                timeToObstacle > 0.06 ? 1 : 2;

  return {status: "OK", kind, phase, distance, width: o.width,
          speed, type, yPos: o.yPos,
          jumping: !!dino.jumping, ducking: !!dino.ducking};
}"""

def load_q_table():
    global q_table
    if not LOAD_MODEL or not os.path.isfile(MODEL_FILE):
        print(f"Q-таблица с нуля. Файл: {MODEL_FILE}")
        return
    try:
        loaded = np.load(MODEL_FILE, allow_pickle=True).item()
        if not isinstance(loaded, dict):
            raise ValueError("Файл модели не содержит словарь")
        expected = set(q_table)
        if set(loaded) != expected or any(len(v) != 3 for v in loaded.values()):
            print("Формат старой модели несовместим — создаю новую таблицу.")
            return
        q_table = {key: [float(x) for x in loaded[key]] for key in expected}
        print(f"Модель загружена: {MODEL_FILE}")
    except Exception as exc:
        print(f"Не удалось загрузить модель ({exc}). Начинаю с новой.")

def save_q_table():
    if SAVE_MODEL:
        np.save(MODEL_FILE, q_table)
        print(f"Модель сохранена: {MODEL_FILE}")

def get_state(page):
    info = page.evaluate(STATE_JS)
    if info.get("status") != "OK":
        return None, info
    kind = int(info.get("kind", NONE))
    phase = int(info.get("phase", 0))
    return (kind, phase), info

def epsilon_for(episode):
    if EPISODES <= 1:
        return EPSILON_END
    ratio = (episode - 1) / (EPISODES - 1)
    return EPSILON_START + (EPSILON_END - EPSILON_START) * ratio

def choose_action(state, epsilon):
    if TRAIN_MODE and random.random() < epsilon:
        return random.choice([WAIT, JUMP, DUCK])
    return int(np.argmax(q_table[state]))

def update_q(state, action, reward, next_state):
    old = q_table[state][action]
    future = 0 if next_state is None else max(q_table[next_state])
    q_table[state][action] = old + ALPHA * (reward + GAMMA * future - old)

def safe_controller(info, learned_action):
    """
    Безопасный контроллер поверх Q-learning.
    Q-learning может выбирать поведение, но опасные решения заменяются
    базовой реакцией на препятствие.
    """
    kind = info.get("kind", NONE)
    distance = float(info.get("distance", 9999))
    speed = float(info.get("speed", 6))
    velocity = speed * 60

    # Высокая птица: приседать, пока она не прошла.
    if kind == HIGH_BIRD:
        width = float(info.get("width", 0))
        if distance <= 110 + width:
            return DUCK

    # Кактусы и низкие птицы: прыжок до столкновения.
    if kind in (SMALL_CACTUS, WIDE_CACTUS, LOW_BIRD):
        # Дистанция trigger = скорость * время + запас на взлёт.
        trigger_distance = velocity * 0.20 + 31
        # Для группы кактусов добавляем запас на ширину.
        if kind == WIDE_CACTUS:
            trigger_distance += min(float(info.get("width", 0)) * 0.25, 20)
        if distance <= trigger_distance and distance > -20:
            if not info.get("jumping", False):
                return JUMP

    # Вне опасной зоны оставляем возможность агенту учиться,
    # но действие WAIT безопаснее всего.
    return WAIT

def do_action(page, action, info):
    if action == JUMP:
        # Не нажимаем прыжок повторно, если динозавр уже в воздухе.
        if not info.get("jumping", False):
            page.keyboard.press("Space")
            return JUMP
        return WAIT
    if action == DUCK:
        page.keyboard.down("ArrowDown")
        return DUCK
    page.keyboard.up("ArrowDown")
    return WAIT

def release_keys(page):
    for key in ("ArrowDown", "Space"):
        try:
            page.keyboard.up(key)
        except Exception:
            pass

def start_game(page):
    try:
        page.locator("canvas").click(force=True)
    except Exception:
        pass
    time.sleep(0.25)
    page.keyboard.press("Space")
    time.sleep(0.8)
    return get_state(page)

def open_game(playwright):
    browser = playwright.chromium.launch(
        channel="chrome",
        headless=False,
        args=["--disable-gpu", "--disable-extensions"]
    )
    page = browser.new_page(viewport={"width": 900, "height": 600})
    page.goto(GAME_URL, wait_until="domcontentloaded")
    time.sleep(2)
    return browser, page

def main():
    load_q_table()
    with sync_playwright() as p:
        browser, page = open_game(p)
        episode_count = EPISODES if TRAIN_MODE else PLAY_EPISODES
        recent_scores = []

        for episode in range(1, episode_count + 1):
            try:
                state, info = start_game(page)
                if state is None:
                    print("Игра не запустилась, перезагружаю.")
                    page.reload(wait_until="domcontentloaded")
                    time.sleep(2)
                    continue

                if TRAIN_MODE and TRAIN_START_SPEED and episode % 2 == 1:
                    try:
                        page.evaluate(
                            "(s) => { window.Runner.instance_.currentSpeed = s; }",
                            TRAIN_START_SPEED
                        )
                    except Exception:
                        pass

                total_reward = 0
                steps = 0
                jumps = 0
                ducks = 0
                cleared = 0
                previous_signature = None
                epsilon = epsilon_for(episode) if TRAIN_MODE else 0

                for step in range(1, MAX_STEPS + 1):
                    learned_action = choose_action(state, epsilon)
                    action = safe_controller(info, learned_action)

                    executed = do_action(page, action, info)
                    jumps += int(executed == JUMP)
                    ducks += int(executed == DUCK)

                    time.sleep(STEP_DELAY)
                    next_state, next_info = get_state(page)

                    reward = 1
                    if executed == JUMP and info.get("kind") == NONE:
                        reward += BAD_ACTION_REWARD
                    if executed == DUCK and info.get("kind") != HIGH_BIRD:
                        reward += BAD_ACTION_REWARD

                    signature = (
                        next_info.get("type"),
                        round(next_info.get("distance", 0) / 20)
                    )
                    # Смена препятствия после нахождения рядом с предыдущим
                    # считается подтверждением прохождения.
                    if previous_signature is not None:
                        prev_type = previous_signature[0]
                        new_type = next_info.get("type")
                        if prev_type not in (None, "NONE") and new_type != prev_type:
                            cleared += 1
                            reward += CLEAR_REWARD
                    previous_signature = signature

                    if TRAIN_MODE:
                        update_q(state, learned_action, reward, next_state)

                    total_reward += reward
                    steps = step
                    state, info = next_state, next_info

                    if state is None:
                        total_reward += DEATH_REWARD
                        if TRAIN_MODE:
                            # Обновляем последнюю пару с отрицательной наградой.
                            update_q((int(next_info.get("kind", NONE)), 0),
                                     learned_action, DEATH_REWARD, None)
                        break

                try:
                    score = page.evaluate(
                        "window.Runner && window.Runner.instance_ "
                        "? window.Runner.instance_.distanceRan : 0"
                    )
                    score = float(score or 0)
                except Exception:
                    score = 0

                recent_scores.append(score)
                recent_scores = recent_scores[-20:]
                average = sum(recent_scores) / len(recent_scores)

                print(
                    f"Эпизод {episode:>3}/{episode_count} | "
                    f"очки={score:>6.0f} | среднее20={average:>6.0f} | "
                    f"шаги={steps:>4} | прыжки={jumps:>3} | "
                    f"приседы={ducks:>3} | препятствия≈{cleared:>3} | "
                    f"награда={total_reward:>5} | ε={epsilon:.2f}"
                )

                release_keys(page)
                if TRAIN_MODE and episode % 10 == 0:
                    save_q_table()
                time.sleep(0.7)

            except Exception as exc:
                print(f"Эпизод {episode} прерван: {type(exc).__name__}: {exc}")
                try:
                    browser.close()
                except Exception:
                    pass
                browser, page = open_game(p)

        if TRAIN_MODE:
            save_q_table()
        release_keys(page)
        browser.close()

if __name__ == "__main__":
    main()