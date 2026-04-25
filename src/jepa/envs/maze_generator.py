from __future__ import annotations

from collections import deque

import numpy as np


EIGHT_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def ca_open_to_wall(interior: np.ndarray, threshold: int) -> np.ndarray:
    """One CA pass: an open cell with more than `threshold` open 8-neighbors flips to wall.

    interior: bool array, True = open, False = wall.
    """
    H, W = interior.shape
    out = interior.copy()
    for r in range(H):
        for c in range(W):
            if not interior[r, c]:
                continue
            n_open = 0
            for dr, dc in EIGHT_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and interior[nr, nc]:
                    n_open += 1
            if n_open > threshold:
                out[r, c] = False
    return out


def all_open_connected(grid: np.ndarray) -> bool:
    """BFS over 4-neighbors from the first open cell; True iff every open cell is reached."""
    H, W = grid.shape
    first = None
    total = 0
    for r in range(H):
        for c in range(W):
            if grid[r, c]:
                total += 1
                if first is None:
                    first = (r, c)
    if first is None:
        return False

    visited = np.zeros_like(grid, dtype=bool)
    queue = deque([first])
    visited[first] = True
    reached = 1
    while queue:
        r, c = queue.popleft()
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                reached += 1
                queue.append((nr, nc))
    return reached == total


def bfs_distances(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """BFS distance from `source` to every cell. -1 for unreachable / wall cells."""
    H, W = grid.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    dist[source] = 0
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in FOUR_NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def sample_far_pair(
    grid: np.ndarray,
    rng: np.random.Generator,
    min_distance: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Uniformly sample (reset, goal) open-cell pair with BFS distance >= min_distance."""
    open_cells = np.argwhere(grid)
    if len(open_cells) < 2:
        return None

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in open_cells:
        dist = bfs_distances(grid, (int(r), int(c)))
        far = np.argwhere(dist >= min_distance)
        for fr, fc in far:
            pairs.append(((int(r), int(c)), (int(fr), int(fc))))

    if not pairs:
        return None
    idx = int(rng.integers(len(pairs)))
    return pairs[idx]


def sample_maze_map(
    rng: np.random.Generator,
    inner_shape: tuple[int, int] = (4, 4),
    space_frac: tuple[float, float] = (0.5, 0.75),
    min_start_goal_cells: int = 3,
    ca_iterations: int = 2,
    ca_open_neighbor_threshold: int = 6,
    max_tries: int = 1000,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Sample a random maze_map following the PLDM algorithm.

    Returns (maze_map, reset_cell, goal_cell). maze_map has shape (inner_H + 2, inner_W + 2)
    int8 with outer ring of 1s (walls) and interior 0/1 (0 = open). reset_cell and goal_cell
    are (row, col) indices into the full maze_map with BFS distance >= min_start_goal_cells
    on the open-cell graph.
    """
    inner_h, inner_w = inner_shape
    low, high = float(space_frac[0]), float(space_frac[1])

    for _ in range(max_tries):
        interior = rng.random((inner_h, inner_w)) < 0.5
        for _ in range(ca_iterations):
            interior = ca_open_to_wall(interior, ca_open_neighbor_threshold)

        open_frac = float(interior.mean())
        if not (low <= open_frac <= high):
            continue

        full_h, full_w = inner_h + 2, inner_w + 2
        full = np.zeros((full_h, full_w), dtype=bool)
        full[1:-1, 1:-1] = interior

        if not all_open_connected(full):
            continue

        pair = sample_far_pair(full, rng, min_start_goal_cells)
        if pair is None:
            continue

        reset_cell, goal_cell = pair
        maze_map = np.ones((full_h, full_w), dtype=np.int8)
        maze_map[full] = 0
        return maze_map, reset_cell, goal_cell

    raise RuntimeError(
        f"Failed to sample a valid maze after {max_tries} tries "
        f"(inner_shape={inner_shape}, space_frac={space_frac}, min_start_goal_cells={min_start_goal_cells})."
    )


def maze_map_to_list(maze_map: np.ndarray) -> list[list[int]]:
    """Convert the int8 ndarray to a plain list-of-lists for gym.make(maze_map=...)."""
    return [[int(v) for v in row] for row in maze_map]
