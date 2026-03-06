from __future__ import annotations

import csv
import heapq
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

Grid = List[List[str]]
Pos = Tuple[int, int]

PASSABLE = {".", "B", "Y", "G", "S"}
MINERAL_TYPES = {"B", "Y", "G"}

TICK_HOURS = 0.5
DAY_TICKS = 32  # 16 hours
NIGHT_TICKS = 16  # 8 hours
CYCLE_TICKS = DAY_TICKS + NIGHT_TICKS

BATTERY_CAPACITY = 100.0
MOVE_K = 2.0
DAY_CHARGE = 10.0
STANDBY_COST = 1.0
MINING_COST = 2.0
SAFETY_RESERVE = 5.0

# Egyetlen inditasi pont: index.py
# DEBUG=True: nincs input kerdes, fix ertekekkel indul (VS Code Run-barat).
DEBUG = True
DEBUG_MAP_FILE = "mars_map_50x50 1.csv"
DEBUG_HOURS = 1680  # 70 nap
RUN_DASHBOARD = True


@dataclass
class TickLog:
    tick: int
    hour: float
    day_phase: str
    x: int
    y: int
    action: str
    speed: int
    moved_cells: int
    battery: float
    collected_b: int
    collected_y: int
    collected_g: int
    total_collected: int
    distance_travelled: int
    target_x: int
    target_y: int


def is_day(tick: int) -> bool:
    return (tick % CYCLE_TICKS) < DAY_TICKS


def load_map(path: Path) -> Tuple[Grid, Pos]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) != 50:
        raise ValueError(f"Invalid map height: {len(rows)} (expected 50)")
    if any(len(r) != 50 for r in rows):
        widths = sorted(set(len(r) for r in rows))
        raise ValueError(f"Invalid map width(s): {widths} (expected all 50)")

    start_positions: List[Pos] = []
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell not in PASSABLE and cell != "#":
                raise ValueError(f"Invalid cell '{cell}' at ({x}, {y})")
            if cell == "S":
                start_positions.append((x, y))

    if len(start_positions) != 1:
        raise ValueError(f"Expected exactly one 'S', found {len(start_positions)}")

    return rows, start_positions[0]


def neighbors_8(pos: Pos, grid: Grid) -> List[Pos]:
    x, y = pos
    h = len(grid)
    w = len(grid[0])
    result: List[Pos] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[ny][nx] == "#":
                continue

            # Atlos mozgasnal ne lehessen akadaly "sarkan" atcsuszni.
            if dx != 0 and dy != 0:
                if grid[y][x + dx] == "#" or grid[y + dy][x] == "#":
                    continue

            result.append((nx, ny))
    return result


def octile_heuristic(a: Pos, b: Pos) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)


def a_star(grid: Grid, start: Pos, goal: Pos) -> Optional[List[Pos]]:
    if start == goal:
        return [start]

    open_heap: List[Tuple[int, int, Pos]] = []
    g_score: Dict[Pos, int] = {start: 0}
    came_from: Dict[Pos, Pos] = {}
    counter = 0

    heapq.heappush(open_heap, (octile_heuristic(start, goal), counter, start))
    in_open: Set[Pos] = {start}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        in_open.discard(current)
        if current == goal:
            path: List[Pos] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score[current]
        for nxt in neighbors_8(current, grid):
            tentative_g = current_g + 1
            if tentative_g < g_score.get(nxt, math.inf):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                if nxt not in in_open:
                    counter += 1
                    f_score = tentative_g + octile_heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f_score, counter, nxt))
                    in_open.add(nxt)

    return None


class RoverSimulator:
    def __init__(self, grid: Grid, start: Pos, mission_hours: int) -> None:
        if mission_hours < 24:
            raise ValueError("Mission duration must be at least 24 hours.")

        self.grid = grid
        self.start = start
        self.pos = start
        self.mission_ticks = int(mission_hours / TICK_HOURS)
        self.tick = 0
        self.battery = BATTERY_CAPACITY
        self.distance_travelled = 0

        self.collected = {"B": 0, "Y": 0, "G": 0}
        self.collected_positions: Set[Pos] = set()
        self.logs: List[TickLog] = []

        self.current_target: Optional[Pos] = None
        self.current_path: List[Pos] = [start]
        self.return_mode = False

        self.path_cache: Dict[Tuple[Pos, Pos], Optional[List[Pos]]] = {}
        self.minerals: List[Pos] = self._collect_minerals()

    def _collect_minerals(self) -> List[Pos]:
        out: List[Pos] = []
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell in MINERAL_TYPES:
                    out.append((x, y))
        return out

    def _get_path(self, a: Pos, b: Pos) -> Optional[List[Pos]]:
        key = (a, b)
        if key not in self.path_cache:
            self.path_cache[key] = a_star(self.grid, a, b)
        return self.path_cache[key]

    def _distance_steps(self, a: Pos, b: Pos) -> Optional[int]:
        path = self._get_path(a, b)
        if path is None:
            return None
        return max(0, len(path) - 1)

    def _can_move_between(self, a: Pos, b: Pos) -> bool:
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay

        # Csak szomszedos mezore lephet.
        if max(abs(dx), abs(dy)) != 1:
            return False

        # Celmezonek jarhatonak kell lennie.
        if self.grid[by][bx] == "#":
            return False

        # Atlosnal a ket ortogonalis koztes irany sem lehet akadaly.
        if dx != 0 and dy != 0:
            if self.grid[ay][ax + dx] == "#" or self.grid[ay + dy][ax] == "#":
                return False

        return True

    def _conservative_energy_needed(self, go_steps: int, back_steps: int, mine: bool) -> float:
        needed = (2.0 * go_steps) + (2.0 * back_steps) + SAFETY_RESERVE
        if mine:
            needed += MINING_COST
        return needed

    def _pick_target(self) -> Optional[Pos]:
        best: Optional[Tuple[float, Pos, List[Pos]]] = None

        for m in self.minerals:
            if m in self.collected_positions:
                continue

            path_go = self._get_path(self.pos, m)
            if not path_go:
                continue

            path_back = self._get_path(m, self.start)
            if not path_back:
                continue

            go_steps = len(path_go) - 1
            back_steps = len(path_back) - 1

            score = 1.0 / (go_steps + back_steps + 1.0)
            if best is None or score > best[0]:
                best = (score, m, path_go)

        if best is None:
            return None

        self.current_target = best[1]
        self.current_path = best[2]
        return self.current_target

    def _speed_policy(self, day: bool) -> int:
        if day:
            if self.battery >= 70:
                return 3
            if self.battery >= 35:
                return 2
            return 1
        if self.battery >= 80:
            return 2
        return 1

    def _max_affordable_speed(self, requested: int) -> int:
        day_credit = DAY_CHARGE if is_day(self.tick) else 0.0
        for v in range(requested, 0, -1):
            if self.battery + day_credit >= MOVE_K * (v ** 2):
                return v
        return 0

    def _ensure_return_priority(self, remaining_ticks: int) -> None:
        dist_to_start = self._distance_steps(self.pos, self.start)
        if dist_to_start is None:
            self.current_target = None
            self.current_path = [self.pos]
            return

        # Hazateres csak vegjatekban: ha az ido mar csak a biztos hazaeresre eleg.
        # (lassu sebesseggel, 1 blokk/tick, +1 tick puffer)
        min_ticks_back_safe = dist_to_start
        if remaining_ticks <= (min_ticks_back_safe + 1):
            self.return_mode = True

        if self.return_mode:
            path = self._get_path(self.pos, self.start)
            if path:
                self.current_target = self.start
                self.current_path = path

    def _log_tick(self, action: str, speed: int, moved_cells: int) -> None:
        phase = "day" if is_day(self.tick) else "night"
        tx, ty = self.current_target if self.current_target else (-1, -1)
        self.logs.append(
            TickLog(
                tick=self.tick,
                hour=(self.tick + 1) * TICK_HOURS,
                day_phase=phase,
                x=self.pos[0],
                y=self.pos[1],
                action=action,
                speed=speed,
                moved_cells=moved_cells,
                battery=round(self.battery, 2),
                collected_b=self.collected["B"],
                collected_y=self.collected["Y"],
                collected_g=self.collected["G"],
                total_collected=sum(self.collected.values()),
                distance_travelled=self.distance_travelled,
                target_x=tx,
                target_y=ty,
            )
        )

    def _apply_day_charge(self) -> None:
        if is_day(self.tick):
            self.battery += DAY_CHARGE
        self.battery = max(0.0, min(BATTERY_CAPACITY, self.battery))

    def run(self) -> None:
        for t in range(self.mission_ticks):
            self.tick = t
            remaining_ticks = self.mission_ticks - t
            self._ensure_return_priority(remaining_ticks)

            action = "wait"
            speed = 0
            moved = 0

            cell = self.grid[self.pos[1]][self.pos[0]]

            if (not self.return_mode) and cell in MINERAL_TYPES and self.pos not in self.collected_positions:
                day_credit = DAY_CHARGE if is_day(self.tick) else 0.0
                if self.battery + day_credit >= MINING_COST:
                    action = "mine"
                    self.battery -= MINING_COST
                    self.collected[cell] += 1
                    self.collected_positions.add(self.pos)
                    self.current_target = None
                    self.current_path = [self.pos]
                else:
                    action = "wait_low_battery"
                    self.battery -= min(self.battery, STANDBY_COST)
            else:
                if (not self.return_mode) and (self.current_target is None or len(self.current_path) <= 1):
                    target = self._pick_target()
                    if target is None:
                        if self.pos != self.start:
                            path_home = self._get_path(self.pos, self.start)
                            if path_home:
                                self.current_target = self.start
                                self.current_path = path_home

                if self.current_target and len(self.current_path) > 1:
                    requested_speed = 3 if self.return_mode else self._speed_policy(day=is_day(self.tick))
                    affordable_speed = self._max_affordable_speed(requested_speed)
                    max_by_path = len(self.current_path) - 1
                    speed = min(affordable_speed, max_by_path)

                    if speed > 0:
                        action = "move"
                        move_cost = MOVE_K * (speed ** 2)
                        self.battery -= move_cost
                        for _ in range(speed):
                            if len(self.current_path) <= 1:
                                break

                            next_pos = self.current_path[1]
                            if not self._can_move_between(self.pos, next_pos):
                                action = "wait_blocked_step"
                                break

                            self.current_path.pop(0)
                            self.pos = self.current_path[0]
                            moved += 1
                            self.distance_travelled += 1
                        if self.pos == self.current_target:
                            self.current_target = None
                            self.current_path = [self.pos]
                    else:
                        action = "wait_no_energy_for_move"
                        self.battery -= min(self.battery, STANDBY_COST)
                else:
                    action = "wait"
                    self.battery -= min(self.battery, STANDBY_COST)

            self._apply_day_charge()
            self._log_tick(action, speed, moved)

        if self.pos != self.start:
            path_home = self._get_path(self.pos, self.start)
            if path_home and len(path_home) > 1:
                self.current_target = self.start
                self.current_path = path_home

    def print_summary(self) -> None:
        print("=== Mars Rover Simulation Summary ===")
        print(f"Ticks simulated: {len(self.logs)} / {self.mission_ticks}")
        print(f"Final position: {self.pos}")
        print(f"Returned to start: {'yes' if self.pos == self.start else 'no'}")
        print(f"Battery: {self.battery:.2f}")
        print(f"Distance travelled: {self.distance_travelled}")
        print(
            f"Collected minerals - B: {self.collected['B']}, "
            f"Y: {self.collected['Y']}, G: {self.collected['G']}, "
            f"Total: {sum(self.collected.values())}"
        )

    def write_log_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "tick",
                    "hour",
                    "day_phase",
                    "x",
                    "y",
                    "action",
                    "speed",
                    "moved_cells",
                    "battery",
                    "collected_b",
                    "collected_y",
                    "collected_g",
                    "total_collected",
                    "distance_travelled",
                    "target_x",
                    "target_y",
                ]
            )
            for row in self.logs:
                writer.writerow(
                    [
                        row.tick,
                        row.hour,
                        row.day_phase,
                        row.x,
                        row.y,
                        row.action,
                        row.speed,
                        row.moved_cells,
                        row.battery,
                        row.collected_b,
                        row.collected_y,
                        row.collected_g,
                        row.total_collected,
                        row.distance_travelled,
                        row.target_x,
                        row.target_y,
                    ]
                )


CELL_COLORS = {
    ".": "#eadfca",
    "#": "#5c6168",
    "B": "#3d7bd9",
    "Y": "#e2b52b",
    "G": "#4e9f57",
    "S": "#c64545",
}


def blend_with_gray(hex_color: str, gray: int = 140, mix: float = 0.65) -> str:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    nr = int(r * (1.0 - mix) + gray * mix)
    ng = int(g * (1.0 - mix) + gray * mix)
    nb = int(b * (1.0 - mix) + gray * mix)
    return f"#{nr:02x}{ng:02x}{nb:02x}"


class DashboardApp:
    def __init__(self, logs: List[TickLog], grid: Grid, start: Pos) -> None:
        self.logs = logs
        self.grid = grid
        self.start = start
        self.max_tick = len(logs) - 1
        self.current_tick = 0
        self.playing = False

        self.cell_size = 12
        self.map_w = 50 * self.cell_size
        self.map_h = 50 * self.cell_size

        self.root = tk.Tk()
        self.root.title("Mars Rover Vizualizacio + Dashboard")

        self.delay_var = tk.IntVar(value=300)
        self.tick_var = tk.IntVar(value=0)
        self.mined_colors = {
            "B": blend_with_gray(CELL_COLORS["B"]),
            "Y": blend_with_gray(CELL_COLORS["Y"]),
            "G": blend_with_gray(CELL_COLORS["G"]),
        }

        self._build_ui()
        self._draw_static_map()
        self._render_tick(0)

    def _build_ui(self) -> None:
        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(main)
        left.pack(side="left", fill="both", expand=False)

        right = tk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        self.map_canvas = tk.Canvas(left, width=self.map_w, height=self.map_h, bg="#f5f1e8", highlightthickness=0)
        self.map_canvas.pack()

        controls = tk.Frame(right)
        controls.pack(fill="x")

        tk.Button(controls, text="Lejatszas", command=self.play).pack(side="left")
        tk.Button(controls, text="Szünet", command=self.pause).pack(side="left", padx=(6, 0))
        tk.Button(controls, text="Reset", command=self.reset).pack(side="left", padx=(6, 0))
        tk.Button(controls, text="Lepes +1", command=self.step_once).pack(side="left", padx=(6, 0))

        tk.Label(controls, text="Kesleltetes (ms):").pack(side="left", padx=(12, 4))
        tk.Scale(controls, from_=100, to=2000, orient="horizontal", variable=self.delay_var, length=220).pack(
            side="left"
        )

        tk.Label(right, text="Tick:").pack(anchor="w", pady=(8, 0))
        self.tick_scale = tk.Scale(
            right,
            from_=0,
            to=max(0, self.max_tick),
            orient="horizontal",
            variable=self.tick_var,
            command=self.on_tick_slider,
            length=600,
        )
        self.tick_scale.pack(anchor="w")

        self.info_text = tk.StringVar(value="")
        tk.Label(right, textvariable=self.info_text, justify="left", font=("Consolas", 10)).pack(anchor="w", pady=(8, 8))

        tk.Label(right, text="Akkumulator ido diagram (0..100)").pack(anchor="w")
        self.battery_canvas = tk.Canvas(right, width=620, height=180, bg="#ffffff", highlightbackground="#cccccc")
        self.battery_canvas.pack(anchor="w", pady=(2, 8))

    def _draw_static_map(self) -> None:
        self.rect_ids: List[List[int]] = []
        for y, row in enumerate(self.grid):
            line_ids: List[int] = []
            for x, cell in enumerate(row):
                color = CELL_COLORS.get(cell, "#000000")
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                rid = self.map_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
                line_ids.append(rid)
            self.rect_ids.append(line_ids)

        sx, sy = self.start
        self.map_canvas.create_rectangle(
            sx * self.cell_size,
            sy * self.cell_size,
            sx * self.cell_size + self.cell_size,
            sy * self.cell_size + self.cell_size,
            outline="#ffffff",
            width=2,
        )

        self.path_line = self.map_canvas.create_line(0, 0, 0, 0, fill="#ff6f3c", width=2)
        self.rover_dot = self.map_canvas.create_oval(0, 0, 0, 0, fill="#ff2222", outline="#7a0000")

    def _update_mined_visuals(self, tick: int) -> None:
        # Slider visszalepes miatt minden rendernel ujraallitjuk az erc szineket.
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell in MINERAL_TYPES:
                    self.map_canvas.itemconfig(self.rect_ids[y][x], fill=CELL_COLORS[cell])

        for row in self.logs[: tick + 1]:
            if row.action == "mine":
                x, y = row.x, row.y
                cell = self.grid[y][x]
                if cell in MINERAL_TYPES:
                    self.map_canvas.itemconfig(self.rect_ids[y][x], fill=self.mined_colors[cell])

    def _set_rover(self, x: int, y: int) -> None:
        r = self.cell_size * 0.4
        cx = x * self.cell_size + self.cell_size / 2
        cy = y * self.cell_size + self.cell_size / 2
        self.map_canvas.coords(self.rover_dot, cx - r, cy - r, cx + r, cy + r)

    def _set_path(self, tick: int) -> None:
        if tick <= 0:
            self.map_canvas.coords(self.path_line, 0, 0, 0, 0)
            return
        points: List[float] = []
        for row in self.logs[: tick + 1]:
            px = row.x * self.cell_size + self.cell_size / 2
            py = row.y * self.cell_size + self.cell_size / 2
            points.extend([px, py])
        self.map_canvas.coords(self.path_line, *points)

    def _draw_battery_chart(self, tick: int) -> None:
        w = 620
        h = 180
        pad = 20
        self.battery_canvas.delete("all")
        self.battery_canvas.create_rectangle(pad, pad, w - pad, h - pad, outline="#bbbbbb")

        if tick <= 0:
            return

        max_x = max(1, self.max_tick)
        points: List[float] = []
        for i in range(tick + 1):
            row = self.logs[i]
            x = pad + (i / max_x) * (w - 2 * pad)
            y = (h - pad) - (row.battery / 100.0) * (h - 2 * pad)
            points.extend([x, y])
        if len(points) >= 4:
            self.battery_canvas.create_line(*points, fill="#1d4ed8", width=2, smooth=True)

    def _render_tick(self, tick: int) -> None:
        tick = max(0, min(tick, self.max_tick))
        row = self.logs[tick]
        self.current_tick = tick
        self.tick_var.set(tick)

        self.root.configure(bg="#fff7e8" if row.day_phase == "day" else "#131722")

        self._update_mined_visuals(tick)
        self._set_path(tick)
        self._set_rover(row.x, row.y)
        self._draw_battery_chart(tick)

        info = (
            f"tick: {row.tick} | ido: {row.hour:.1f} h | napszak: {row.day_phase}\n"
            f"pozicio: ({row.x}, {row.y}) | akcio: {row.action} | sebesseg: {row.speed}\n"
            f"akku: {row.battery:.1f}% | tavolsag: {row.distance_travelled}\n"
            f"asvanyok: B={row.collected_b}, Y={row.collected_y}, G={row.collected_g}, osszes={row.total_collected}"
        )
        self.info_text.set(info)

    def on_tick_slider(self, value: str) -> None:
        self.pause()
        self._render_tick(int(float(value)))

    def play(self) -> None:
        if self.playing:
            return
        self.playing = True
        self._loop()

    def pause(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.pause()
        self._render_tick(0)

    def step_once(self) -> None:
        self.pause()
        self._render_tick(min(self.current_tick + 1, self.max_tick))

    def _loop(self) -> None:
        if not self.playing:
            return
        if self.current_tick < self.max_tick:
            self._render_tick(self.current_tick + 1)
            self.root.after(self.delay_var.get(), self._loop)
        else:
            self.playing = False

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if DEBUG:
        map_file = DEBUG_MAP_FILE
        hours = DEBUG_HOURS
    else:
        map_file_in = input(f"Terkep fajl [{DEBUG_MAP_FILE}]: ").strip()
        hours_in = input(f"Idokeret ora [{DEBUG_HOURS}]: ").strip()
        map_file = map_file_in if map_file_in else DEBUG_MAP_FILE
        if hours_in:
            try:
                hours = int(hours_in)
            except ValueError:
                print("Hibas oraszam, alapertelmezett ertek lesz hasznalva.")
                hours = DEBUG_HOURS
        else:
            hours = DEBUG_HOURS

    map_path = Path(map_file)
    grid, start = load_map(map_path)

    simulator = RoverSimulator(grid=grid, start=start, mission_hours=hours)
    simulator.run()

    if RUN_DASHBOARD:
        app = DashboardApp(logs=simulator.logs, grid=grid, start=start)
        app.run()
    else:
        simulator.print_summary()
        simulator.write_log_csv(Path("simulation_log.csv"))
        print("Log written to: simulation_log.csv")


if __name__ == "__main__":
    main()
