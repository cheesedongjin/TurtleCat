import argparse
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional, Iterable

from PIL import Image
import turtle


@dataclass
class Settings:
    max_side: int = 240
    tolerance: int = 24

    min_gap: int = 1
    max_gap: int = 10

    pen_size: int = 1
    pen_color: str = "black"
    bg_color: str = "white"

    speed: int = 0
    update_every: int = 8
    margin: int = 20


CANVAS_SCALE = 1.0

DIRS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def load_gray_resized(path: str, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return img.convert("L")


def quantize_gray(gray: Image.Image, tolerance: int) -> List[List[int]]:
    w, h = gray.size
    tol = max(1, int(tolerance))
    q = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            b = int(gray.getpixel((x, y)))
            q[y][x] = b // tol
    return q


def segment_regions(q: List[List[int]]) -> Tuple[List[List[int]], int]:
    h = len(q)
    w = len(q[0]) if h else 0
    rid = [[-1] * w for _ in range(h)]
    region_count = 0

    for y0 in range(h):
        for x0 in range(w):
            if rid[y0][x0] != -1:
                continue

            region_label = q[y0][x0]
            rid[y0][x0] = region_count
            dq = deque([(x0, y0)])

            while dq:
                x, y = dq.popleft()
                for dx, dy in DIRS:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if rid[ny][nx] != -1:
                        continue
                    if q[ny][nx] != region_label:
                        continue
                    rid[ny][nx] = region_count
                    dq.append((nx, ny))

            region_count += 1

    return rid, region_count


def box_blur_gray(gray: Image.Image, radius: int) -> Image.Image:
    """Simple box blur without extra deps. radius 0이면 그대로."""
    radius = int(radius)
    if radius <= 0:
        return gray

    w, h = gray.size
    src = gray.load()
    out = Image.new("L", (w, h))
    dst = out.load()

    r = radius
    # 누적합(적분 이미지)로 O(1) 블러
    integ = [[0] * (w + 1) for _ in range(h + 1)]
    for y in range(1, h + 1):
        row_sum = 0
        for x in range(1, w + 1):
            row_sum += int(src[x - 1, y - 1])
            integ[y][x] = integ[y - 1][x] + row_sum

    def rect_sum(x0, y0, x1, y1) -> int:
        # [x0,x1) [y0,y1)
        return integ[y1][x1] - integ[y0][x1] - integ[y1][x0] + integ[y0][x0]

    for y in range(h):
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        for x in range(w):
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            s = rect_sum(x0, y0, x1, y1)
            area = (x1 - x0) * (y1 - y0)
            dst[x, y] = int(round(s / area))

    return out


def sobel_edges(gray: Image.Image) -> List[List[int]]:
    """Sobel gradient magnitude (0~255 정도 스케일)."""
    w, h = gray.size
    p = gray.load()
    g = [[0] * w for _ in range(h)]

    def get(x, y):
        x = 0 if x < 0 else (w - 1 if x >= w else x)
        y = 0 if y < 0 else (h - 1 if y >= h else y)
        return int(p[x, y])

    for y in range(h):
        for x in range(w):
            gx = (
                -1 * get(x - 1, y - 1) + 1 * get(x + 1, y - 1) +
                -2 * get(x - 1, y)     + 2 * get(x + 1, y) +
                -1 * get(x - 1, y + 1) + 1 * get(x + 1, y + 1)
            )
            gy = (
                -1 * get(x - 1, y - 1) + -2 * get(x, y - 1) + -1 * get(x + 1, y - 1) +
                 1 * get(x - 1, y + 1) +  2 * get(x, y + 1) +  1 * get(x + 1, y + 1)
            )
            mag = int(min(255, (gx * gx + gy * gy) ** 0.5))
            g[y][x] = mag

    return g


def segment_regions_edge_aware(
    gray_smooth: Image.Image,
    edges: List[List[int]],
    tolerance: int,
    edge_th: int,
) -> Tuple[List[List[int]], int]:
    """
    Edge-aware region growing:
    - 밝기 차이가 tolerance 이하면 같은 면으로 확장
    - 단, 에지가 강한 픽셀(edge >= edge_th)은 경계로 취급(넘어가지 않음)
    """
    w, h = gray_smooth.size
    pix = gray_smooth.load()

    rid = [[-1] * w for _ in range(h)]
    region_count = 0

    tol = max(0, int(tolerance))
    edge_th = max(0, int(edge_th))

    for y0 in range(h):
        for x0 in range(w):
            if rid[y0][x0] != -1:
                continue

            # seed
            region_id = region_count
            region_count += 1

            seed_b = int(pix[x0, y0])
            rid[y0][x0] = region_id

            dq = deque([(x0, y0)])
            # region mean을 누적하며 갱신(좀 더 안정적)
            s = seed_b
            c = 1

            while dq:
                x, y = dq.popleft()
                mean_b = s / c

                for dx, dy in DIRS:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= w or ny < 0 or ny >= h:
                        continue
                    if rid[ny][nx] != -1:
                        continue

                    # 강한 에지는 '벽'으로 취급
                    if edges[ny][nx] >= edge_th:
                        continue

                    nb = int(pix[nx, ny])

                    # mean 기반 확장(노이즈에 덜 쪼개짐)
                    if abs(nb - mean_b) <= tol:
                        rid[ny][nx] = region_id
                        dq.append((nx, ny))
                        s += nb
                        c += 1

            # 끝

    return rid, region_count


def build_adjacency_from_rid(rid: List[List[int]], region_count: int) -> List[Set[int]]:
    h = len(rid)
    w = len(rid[0]) if h else 0
    adj = [set() for _ in range(region_count)]
    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            if x + 1 < w:
                r2 = rid[y][x + 1]
                if r2 != r:
                    adj[r].add(r2); adj[r2].add(r)
            if y + 1 < h:
                r2 = rid[y + 1][x]
                if r2 != r:
                    adj[r].add(r2); adj[r2].add(r)
    return adj


def merge_small_regions(
    rid: List[List[int]],
    gray_smooth: Image.Image,
    min_region: int,
) -> Tuple[List[List[int]], int]:
    """
    작은 영역을 주변 영역 중 '밝기 평균이 가장 가까운' 곳으로 병합.
    반복 1회로도 효과 큼.
    """
    w, h = gray_smooth.size
    pix = gray_smooth.load()

    # stats
    # region_id 재매핑을 위해 먼저 크기/평균 계산
    # (region_count는 rid에서 추정)
    max_id = -1
    for y in range(h):
        for x in range(w):
            if rid[y][x] > max_id:
                max_id = rid[y][x]
    region_count = max_id + 1

    sums = [0] * region_count
    counts = [0] * region_count
    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            b = int(pix[x, y])
            sums[r] += b
            counts[r] += 1

    avg = [ (sums[i] / counts[i]) if counts[i] else 255.0 for i in range(region_count) ]

    # 인접 후보 수집
    # small region -> neighbor ids
    neighbors = [set() for _ in range(region_count)]
    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                r2 = rid[ny][nx]
                if r2 != r:
                    neighbors[r].add(r2)

    min_region = max(0, int(min_region))
    target = list(range(region_count))  # default self

    for r in range(region_count):
        if counts[r] >= min_region:
            continue
        if not neighbors[r]:
            continue

        best = None
        best_d = 1e18
        for nb in neighbors[r]:
            d = abs(avg[r] - avg[nb])
            if d < best_d:
                best_d = d
                best = nb
        if best is not None:
            target[r] = best

    # apply merge mapping (1-step)
    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            rid[y][x] = target[r]

    # compress ids
    mapping: Dict[int, int] = {}
    new_id = 0
    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            if r not in mapping:
                mapping[r] = new_id
                new_id += 1
            rid[y][x] = mapping[r]

    return rid, new_id


def compute_region_stats(gray: Image.Image, rid: List[List[int]], region_count: int):
    w, h = gray.size
    sums = [0] * region_count
    counts = [0] * region_count
    minx = [10**9] * region_count
    miny = [10**9] * region_count
    maxx = [-1] * region_count
    maxy = [-1] * region_count

    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            b = int(gray.getpixel((x, y)))
            sums[r] += b
            counts[r] += 1
            if x < minx[r]:
                minx[r] = x
            if y < miny[r]:
                miny[r] = y
            if x > maxx[r]:
                maxx[r] = x
            if y > maxy[r]:
                maxy[r] = y

    avg = [0.0] * region_count
    for r in range(region_count):
        avg[r] = sums[r] / counts[r] if counts[r] else 255.0

    bboxes = [(minx[r], miny[r], maxx[r], maxy[r]) for r in range(region_count)]
    return avg, counts, bboxes


def build_adjacency(rid: List[List[int]], region_count: int) -> List[Set[int]]:
    h = len(rid)
    w = len(rid[0]) if h else 0
    adj = [set() for _ in range(region_count)]

    for y in range(h):
        for x in range(w):
            r = rid[y][x]
            if x + 1 < w:
                r2 = rid[y][x + 1]
                if r2 != r:
                    adj[r].add(r2)
                    adj[r2].add(r)
            if y + 1 < h:
                r2 = rid[y + 1][x]
                if r2 != r:
                    adj[r].add(r2)
                    adj[r2].add(r)

    return adj


def dsatur_4color(adj: List[Set[int]], k: int = 4) -> List[int]:
    """
    Non-recursive DSATUR-style greedy coloring (fast, no recursion).
    Always returns a coloring with colors in [0..k-1], but may contain conflicts
    if the graph is not k-colorable (rare for planar-ish regions, but possible due to segmentation artifacts).
    We'll optionally repair conflicts later.
    """
    n = len(adj)
    colors = [-1] * n
    neighbor_colors = [set() for _ in range(n)]
    uncolored = set(range(n))

    def saturation(v: int) -> int:
        return len(neighbor_colors[v])

    def degree(v: int) -> int:
        return len(adj[v])

    while uncolored:
        # pick vertex with max saturation, then max degree
        v = max(uncolored, key=lambda x: (saturation(x), degree(x), -x))

        used = neighbor_colors[v]
        # pick smallest available color
        c = None
        for cand in range(k):
            if cand not in used:
                c = cand
                break
        if c is None:
            # no available color: assign 0 anyway (will be repaired if desired)
            c = 0

        colors[v] = c
        uncolored.remove(v)

        for u in adj[v]:
            if colors[u] == -1:
                neighbor_colors[u].add(c)

    return colors


def greedy_4color(adj: List[Set[int]], k: int = 4) -> List[int]:
    n = len(adj)
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    colors = [-1] * n
    for v in order:
        used = {colors[u] for u in adj[v] if colors[u] != -1}
        for c in range(k):
            if c not in used:
                colors[v] = c
                break
        if colors[v] == -1:
            colors[v] = 0
    return colors


def brightness_to_gap(avg_brightness: float, min_gap: int, max_gap: int) -> int:
    t = clamp(avg_brightness / 255.0, 0.0, 1.0)
    gap = min_gap + (max_gap - min_gap) * t
    return int(round(clamp(gap, min_gap, max_gap)))


def to_canvas_coords(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    cx = (x - (w / 2.0)) * CANVAS_SCALE
    cy = ((h / 2.0) - y) * CANVAS_SCALE
    return cx, cy


def setup_turtle(w: int, h: int, s: Settings) -> turtle.Turtle:
    global CANVAS_SCALE

    screen = turtle.Screen()

    # 실제 화면 크기 (px)
    screen_w = screen.window_width()
    screen_h = screen.window_height()

    # 여백 고려
    avail_w = screen_w - s.margin * 2
    avail_h = screen_h - s.margin * 2

    # 스케일 계산 (작으면 확대, 크면 축소)
    scale_w = avail_w / w
    scale_h = avail_h / h
    CANVAS_SCALE = min(scale_w, scale_h)

    # 너무 과한 확대/축소 방지
    CANVAS_SCALE = clamp(CANVAS_SCALE, 0.5, 3.0)

    canvas_w = int(w * CANVAS_SCALE) + s.margin * 2
    canvas_h = int(h * CANVAS_SCALE) + s.margin * 2

    screen.setup(width=canvas_w, height=canvas_h)
    screen.bgcolor(s.bg_color)

    t = turtle.Turtle(visible=False)
    t.speed(max(0, min(10, int(s.speed))))
    t.pencolor(s.pen_color)
    t.pensize(s.pen_size)
    t.penup()

    turtle.tracer(0, 0)
    return t


def draw_segments_along_line(t: turtle.Turtle, points: List[Tuple[int, int]], rid: List[List[int]], region_id: int, w: int, h: int):
    pen_down = False

    def goto_px(px: int, py: int):
        cx, cy = to_canvas_coords(px, py, w, h)
        t.goto(cx, cy)

    first = True
    for (x, y) in points:
        if first:
            goto_px(x, y)
            first = False

        inside = (rid[y][x] == region_id)
        if inside and not pen_down:
            t.pendown()
            pen_down = True
        elif (not inside) and pen_down:
            t.penup()
            pen_down = False

        goto_px(x, y)

    if pen_down:
        t.penup()


def gen_points_horizontal(y: int, x0: int, x1: int, step: int) -> List[Tuple[int, int]]:
    if x0 <= x1:
        return [(x, y) for x in range(x0, x1 + 1, step)]
    return [(x, y) for x in range(x0, x1 - 1, -step)]


def gen_points_vertical(x: int, y0: int, y1: int, step: int) -> List[Tuple[int, int]]:
    if y0 <= y1:
        return [(x, y) for y in range(y0, y1 + 1, step)]
    return [(x, y) for y in range(y0, y1 - 1, -step)]


def gen_points_diag_down_left_to_up_right(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    pts = []
    x, y = x0, y0
    dx = 1 if x1 >= x0 else -1
    dy = -1 if y1 <= y0 else 1
    while True:
        pts.append((x, y))
        if x == x1 or y == y1:
            break
        x += dx
        y += dy
    return pts


def gen_points_diag_up_left_to_down_right(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    pts = []
    x, y = x0, y0
    dx = 1 if x1 >= x0 else -1
    dy = 1 if y1 >= y0 else -1
    while True:
        pts.append((x, y))
        if x == x1 or y == y1:
            break
        x += dx
        y += dy
    return pts


def draw_region_hatching(
    t: turtle.Turtle,
    rid: List[List[int]],
    region_id: int,
    avg_brightness: float,
    bbox: Tuple[int, int, int, int],
    direction: int,
    s: Settings,
    w: int,
    h: int,
    update_counter: List[int],
):
    x0, y0, x1, y1 = bbox
    if x1 < x0 or y1 < y0:
        return

    gap = brightness_to_gap(avg_brightness, s.min_gap, s.max_gap)

    if direction == 0:
        y = y0
        left_to_right = True
        while y <= y1:
            pts = gen_points_horizontal(y, x0, x1, 1) if left_to_right else gen_points_horizontal(y, x1, x0, 1)
            draw_segments_along_line(t, pts, rid, region_id, w, h)
            y += gap
            left_to_right = not left_to_right

    elif direction == 2:
        x = x0
        top_to_bottom = True
        while x <= x1:
            pts = gen_points_vertical(x, y0, y1, 1) if top_to_bottom else gen_points_vertical(x, y1, y0, 1)
            draw_segments_along_line(t, pts, rid, region_id, w, h)
            x += gap
            top_to_bottom = not top_to_bottom

    elif direction == 1:
        start_s = (x0 + y0)
        end_s = (x1 + y1)
        sidx = start_s
        flip = False
        while sidx <= end_s:
            pts = []

            x_start = max(x0, sidx - y1)
            x_end = min(x1, sidx - y0)

            if x_start <= x_end:
                if not flip:
                    x = x_start
                    while x <= x_end:
                        y = sidx - x
                        pts.append((x, y))
                        x += 1
                else:
                    x = x_end
                    while x >= x_start:
                        y = sidx - x
                        pts.append((x, y))
                        x -= 1

                draw_segments_along_line(t, pts, rid, region_id, w, h)

            sidx += gap
            flip = not flip

    elif direction == 3:
        start_d = (x0 - y1)
        end_d = (x1 - y0)
        didx = start_d
        flip = False
        while didx <= end_d:
            pts = []

            y_start = max(y0, x0 - didx)
            y_end = min(y1, x1 - didx)

            if y_start <= y_end:
                if not flip:
                    y = y_start
                    while y <= y_end:
                        x = didx + y
                        pts.append((x, y))
                        y += 1
                else:
                    y = y_end
                    while y >= y_start:
                        x = didx + y
                        pts.append((x, y))
                        y -= 1

                draw_segments_along_line(t, pts, rid, region_id, w, h)

            didx += gap
            flip = not flip

    update_counter[0] += 1
    if update_counter[0] % max(1, s.update_every) == 0:
        turtle.update()


def repair_coloring(adj: List[Set[int]], colors: List[int], k: int = 4, max_iters: int = 50) -> Tuple[List[int], int]:
    """
    Try to reduce/resolve conflicts where adjacent nodes share same color.
    Returns (colors, conflict_count).
    """
    n = len(adj)
    colors = colors[:]

    def conflict_edges() -> List[Tuple[int, int]]:
        edges = []
        for v in range(n):
            cv = colors[v]
            for u in adj[v]:
                if u > v and colors[u] == cv:
                    edges.append((v, u))
        return edges

    for _ in range(max_iters):
        edges = conflict_edges()
        if not edges:
            return colors, 0

        # count conflicts per vertex
        bad = [0] * n
        for v, u in edges:
            bad[v] += 1
            bad[u] += 1

        # fix the worst vertex first
        v = max(range(n), key=lambda i: bad[i])
        if bad[v] == 0:
            break

        used = {colors[u] for u in adj[v]}
        best_c = colors[v]
        best_score = bad[v]

        for c in range(k):
            if c in used:
                continue
            # compute new conflict count for v if assign c
            score = 0
            for u in adj[v]:
                if colors[u] == c:
                    score += 1
            if score < best_score:
                best_score = score
                best_c = c

        colors[v] = best_c

    # final conflicts
    final = len(conflict_edges())
    return colors, final


def _edge_key(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    return (a, b) if a <= b else (b, a)


def collect_boundary_edges(rid: List[List[int]]) -> Dict[Tuple[Tuple[float, float], Tuple[float, float]], int]:
    """
    Collect boundary edges on pixel grid as undirected edges between vertices.
    Pixel (x,y) occupies [x, x+1] x [y, y+1].
    If rid[y][x] != rid[y][x+1], we add vertical edge at x+1 from (x+1,y) to (x+1,y+1).
    If rid[y][x] != rid[y+1][x], we add horizontal edge at y+1 from (x,y+1) to (x+1,y+1).
    Returns multiset edge->count (rarely >1 but safe).
    """
    h = len(rid)
    w = len(rid[0]) if h else 0
    edges: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}

    # Vertical boundaries
    for y in range(h):
        for x in range(w - 1):
            r1 = rid[y][x]
            r2 = rid[y][x + 1]
            if r1 != r2:
                a = (float(x + 1), float(y))
                b = (float(x + 1), float(y + 1))
                k = _edge_key(a, b)
                edges[k] = edges.get(k, 0) + 1

    # Horizontal boundaries
    for y in range(h - 1):
        for x in range(w):
            r1 = rid[y][x]
            r2 = rid[y + 1][x]
            if r1 != r2:
                a = (float(x), float(y + 1))
                b = (float(x + 1), float(y + 1))
                k = _edge_key(a, b)
                edges[k] = edges.get(k, 0) + 1

    return edges


def build_vertex_adjacency(
    edges: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int]
) -> Dict[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Build adjacency list for contour tracing.
    We expand multiplicities as repeated neighbors to keep counts consistent.
    """
    adj: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)
    for (a, b), cnt in edges.items():
        for _ in range(cnt):
            adj[a].append(b)
            adj[b].append(a)
    return adj


def _consume_edge(
    edges: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int],
    a: Tuple[float, float],
    b: Tuple[float, float]
) -> None:
    k = _edge_key(a, b)
    c = edges.get(k, 0)
    if c <= 1:
        edges.pop(k, None)
    else:
        edges[k] = c - 1


def trace_contours_from_edges(
    edges_in: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int]
) -> List[List[Tuple[float, float]]]:
    """
    Trace polylines/loops from boundary edges.
    Returns list of paths; most will be closed loops.
    """
    edges = dict(edges_in)
    adj = build_vertex_adjacency(edges)

    def pop_any_edge() -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        for k in edges.keys():
            return k
        return None

    contours: List[List[Tuple[float, float]]] = []

    while edges:
        e = pop_any_edge()
        if e is None:
            break
        a, b = e
        _consume_edge(edges, a, b)

        path = [a, b]
        # extend forward
        curr = b
        prev = a

        while True:
            nbrs = adj.get(curr, [])
            # choose a neighbor that still has an edge remaining
            nxt = None
            for cand in nbrs:
                k = _edge_key(curr, cand)
                if k in edges:
                    # prefer not going back if possible
                    if cand != prev:
                        nxt = cand
                        break
            if nxt is None:
                # maybe only back edge remains
                for cand in nbrs:
                    k = _edge_key(curr, cand)
                    if k in edges:
                        nxt = cand
                        break

            if nxt is None:
                break

            _consume_edge(edges, curr, nxt)
            path.append(nxt)
            prev, curr = curr, nxt

            # closed loop check
            if curr == path[0]:
                break

        contours.append(path)

    return contours


def rdp_simplify(points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
    """
    Ramer–Douglas–Peucker polyline simplification.
    epsilon in image coordinate units (pixels).
    """
    if epsilon <= 0 or len(points) < 3:
        return points

    def dist_point_to_segment(p, a, b) -> float:
        (px, py), (ax, ay), (bx, by) = p, a, b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        vv = vx * vx + vy * vy
        if vv == 0:
            dx, dy = px - ax, py - ay
            return (dx * dx + dy * dy) ** 0.5
        t = (wx * vx + wy * vy) / vv
        t = clamp(t, 0.0, 1.0)
        cx, cy = ax + t * vx, ay + t * vy
        dx, dy = px - cx, py - cy
        return (dx * dx + dy * dy) ** 0.5

    def _rdp(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(pts) < 3:
            return pts
        a = pts[0]
        b = pts[-1]
        max_d = -1.0
        idx = -1
        for i in range(1, len(pts) - 1):
            d = dist_point_to_segment(pts[i], a, b)
            if d > max_d:
                max_d = d
                idx = i
        if max_d > epsilon:
            left = _rdp(pts[: idx + 1])
            right = _rdp(pts[idx:])
            return left[:-1] + right
        else:
            return [a, b]

    return _rdp(points)


def chaikin_smooth(points: List[Tuple[float, float]], iterations: int, closed: bool) -> List[Tuple[float, float]]:
    """
    Chaikin corner cutting. Produces smoother curves.
    """
    if iterations <= 0 or len(points) < 3:
        return points

    pts = points[:]
    for _ in range(iterations):
        new_pts: List[Tuple[float, float]] = []
        n = len(pts)

        if closed:
            # ensure closed for processing
            if pts[0] != pts[-1]:
                pts = pts + [pts[0]]
            n = len(pts)

            for i in range(n - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_pts.extend([q, r])

            # re-close
            if new_pts and new_pts[0] != new_pts[-1]:
                new_pts.append(new_pts[0])

        else:
            new_pts.append(pts[0])
            for i in range(n - 1):
                p0 = pts[i]
                p1 = pts[i + 1]
                q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
                r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
                new_pts.extend([q, r])
            new_pts.append(pts[-1])

        pts = new_pts

    return pts


def draw_polyline(
    t: turtle.Turtle,
    pts: List[Tuple[float, float]],
    w: int,
    h: int,
    closed: bool,
):
    if not pts:
        return

    # turtle coordinates
    cx0, cy0 = to_canvas_coords(pts[0][0], pts[0][1], w, h)
    t.penup()
    t.goto(cx0, cy0)
    t.pendown()

    for (x, y) in pts[1:]:
        cx, cy = to_canvas_coords(x, y, w, h)
        t.goto(cx, cy)

    if closed and pts[0] != pts[-1]:
        t.goto(cx0, cy0)

    t.penup()


def draw_region_outlines_vector(
    t: turtle.Turtle,
    rid: List[List[int]],
    w: int,
    h: int,
    outline_color: str = "black",
    outline_size: int = 1,
    simplify_epsilon: float = 1.0,
    smooth_iters: int = 1,
    update_every: int = 200,
):
    """
    Vector-like contour tracing:
    1) collect boundary edges on pixel grid
    2) trace contours into polylines
    3) simplify (RDP)
    4) smooth (Chaikin)
    5) draw
    """
    prev_color = t.pencolor()
    prev_size = t.pensize()
    t.pencolor(outline_color)
    t.pensize(outline_size)

    edges = collect_boundary_edges(rid)
    contours = trace_contours_from_edges(edges)

    c = 0
    for path in contours:
        closed = (len(path) >= 3 and path[0] == path[-1])

        pts = path
        # simplify first (keeps shape), then smooth
        if simplify_epsilon > 0:
            # for closed loops, simplify without the duplicated last point to avoid artifacts
            if closed:
                core = pts[:-1]
                core2 = rdp_simplify(core + [core[0]], simplify_epsilon)
                pts2 = core2
                if pts2[0] != pts2[-1]:
                    pts2.append(pts2[0])
                pts = pts2
            else:
                pts = rdp_simplify(pts, simplify_epsilon)

        if smooth_iters > 0 and len(pts) >= 4:
            pts = chaikin_smooth(pts, smooth_iters, closed=closed)
            # safety: keep closed
            if closed and pts[0] != pts[-1]:
                pts.append(pts[0])

        draw_polyline(t, pts, w, h, closed=closed)

        c += 1
        if c % max(1, int(update_every)) == 0:
            turtle.update()

    turtle.update()
    t.pencolor(prev_color)
    t.pensize(prev_size)


def main():
    ap = argparse.ArgumentParser(description="Region-based turtle hatching with 4-direction coloring")
    ap.add_argument("image_path", help="input image path")
    ap.add_argument("--max-side", type=int, default=Settings.max_side, help="resize so max(width,height)=this")
    ap.add_argument("--tolerance", type=int, default=Settings.tolerance, help="brightness quantization bucket size (bigger -> fewer regions)")
    ap.add_argument("--min-gap", type=int, default=Settings.min_gap, help="min hatch gap for dark regions")
    ap.add_argument("--max-gap", type=int, default=Settings.max_gap, help="max hatch gap for bright regions")
    ap.add_argument("--pen-size", type=int, default=Settings.pen_size, help="turtle pensize")
    ap.add_argument("--pen-color", type=str, default=Settings.pen_color, help="turtle pencolor (name or #RRGGBB)")
    ap.add_argument("--no-outline", action="store_true", help="disable region outline (default: enabled)")
    ap.add_argument("--outline-color", type=str, default="black", help="outline color")
    ap.add_argument("--outline-size", type=int, default=1, help="outline pensize")
    ap.add_argument("--outline-epsilon", type=float, default=1.2, help="RDP simplify epsilon (px). bigger -> smoother/simpler")
    ap.add_argument("--outline-smooth", type=int, default=1, help="Chaikin smoothing iterations (0..)")
    ap.add_argument("--bg-color", type=str, default=Settings.bg_color, help="background color")
    ap.add_argument("--blur", type=int, default=1, help="pre-blur radius to reduce noise (0..)")
    ap.add_argument("--edge-th", type=int, default=28, help="edge threshold (lower -> edges block more)")
    ap.add_argument("--min-region", type=int, default=80, help="merge regions smaller than this (px count)")
    ap.add_argument("--speed", type=int, default=Settings.speed, help="turtle speed 0..10 (0 fastest)")
    ap.add_argument("--update-every", type=int, default=Settings.update_every, help="screen update every N regions")
    
    args = ap.parse_args()

    s = Settings(
        max_side=args.max_side,
        tolerance=args.tolerance,
        min_gap=args.min_gap,
        max_gap=args.max_gap,
        pen_size=args.pen_size,
        pen_color=args.pen_color,
        bg_color=args.bg_color,
        speed=args.speed,
        update_every=args.update_every,
    )

    gray = load_gray_resized(args.image_path, s.max_side)
    w, h = gray.size

    # 1) smooth + edges
    gray_s = box_blur_gray(gray, args.blur)
    edges = sobel_edges(gray_s)

    # 2) edge-aware segmentation
    rid, region_count = segment_regions_edge_aware(
        gray_smooth=gray_s,
        edges=edges,
        tolerance=s.tolerance,
        edge_th=args.edge_th,
    )

    # 3) merge tiny regions (reduces "too many tiny pieces")
    rid, region_count = merge_small_regions(
        rid=rid,
        gray_smooth=gray_s,
        min_region=args.min_region,
    )

    # 4) stats/adj based on final rid
    avg, counts, bboxes = compute_region_stats(gray, rid, region_count)  # 원본 gray로 평균 잡아도 OK
    adj = build_adjacency_from_rid(rid, region_count)

    colors = dsatur_4color(adj, 4)
    colors, conflicts = repair_coloring(adj, colors, 4, max_iters=80)
    used_fallback = (conflicts > 0)

    t = setup_turtle(w, h, s)

    direction_names = {
        0: "H",
        1: "D1",
        2: "V",
        3: "D2",
    }

    update_counter = [0]

    order = sorted(range(region_count), key=lambda r: counts[r], reverse=True)

    for r in order:
        direction = colors[r] % 4
        draw_region_hatching(
            t=t,
            rid=rid,
            region_id=r,
            avg_brightness=avg[r],
            bbox=bboxes[r],
            direction=direction,
            s=s,
            w=w,
            h=h,
            update_counter=update_counter,
        )

    # Draw outlines on top (default: enabled)
    if not args.no_outline:
        draw_region_outlines_vector(
            t=t,
            rid=rid,
            w=w,
            h=h,
            outline_color=args.outline_color,
            outline_size=args.outline_size,
            simplify_epsilon=args.outline_epsilon,
            smooth_iters=args.outline_smooth,
            update_every=200,
        )

    turtle.update()

    title = f"Done - regions={region_count}"
    if used_fallback:
        title += " (warning: used greedy fallback)"
    turtle.Screen().title(title)
    turtle.Screen().exitonclick()


if __name__ == "__main__":
    main()
