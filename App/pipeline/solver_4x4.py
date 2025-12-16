import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


GRID_ROWS = 4
GRID_COLS = 4

# --- Tuned Weights for Cartoons ---
WEIGHT_RATIO = 30.0        # Aggressive against ambiguity (sky vs sky)
WEIGHT_MUTUAL_BB = 15.0    # Trust "Best Buddies" implicitly

# --- Hybrid penalties (soft, like notebook hybrid) ---
VIGNETTE_PENALTY   = 50.0   # soft cost instead of 1e9
DARKNESS_THRESHOLD = 60
FLATNESS_PENALTY   = 100.0
FLATNESS_THRESHOLD = 5.0

# --- Path Setup ---
DATASET_PATH = "/content/drive/MyDrive/dataset"
CORRECT_PATH = os.path.join(DATASET_PATH, "correct")
PUZZLE_PATH = os.path.join(DATASET_PATH, "puzzle_4x4")
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')


class JigsawSolverV7_HybridGUI:
    def __init__(self, piece_images):
        self.pieces_dict = piece_images
        self.piece_ids = list(piece_images.keys())
        self.n = len(self.piece_ids)
        self.pieces_list = [piece_images[pid] for pid in self.piece_ids]

        sample = self.pieces_list[0]
        self.h, self.w = sample.shape[:2]

        # Hybrid-style feature extraction (LAB + grayscale edges + flatness)
        self._extract_features_hybrid()
        # Hybrid soft costs (no infinite walls – just big penalties)
        self.raw_ver, self.raw_hor = self._calculate_hybrid_costs()

        # Same best‑buddy machinery as V7
        self.norm_ver = self._normalize(self.raw_ver)
        self.norm_hor = self._normalize(self.raw_hor)

        self.ratio_ver = self._compute_ratios(self.raw_ver)
        self.ratio_hor = self._compute_ratios(self.raw_hor)

        self.bb_ver, self.bb_hor = self._calculate_mutual_best_buddies(self.raw_ver, self.raw_hor)
        self.sbb_ver, self.sbb_hor = self._calculate_single_best_buddies(self.raw_ver, self.raw_hor)

    # ====================================================
    # FEATURE EXTRACTION (HYBRID)
    # ====================================================
    def _extract_features_hybrid(self):
        n = self.n
        h, w = self.h, self.w

        # LAB edges
        self.edges = {
            'top': np.zeros((n, w, 3), np.float32),
            'bot': np.zeros((n, w, 3), np.float32),
            'lef': np.zeros((n, h, 3), np.float32),
            'rig': np.zeros((n, h, 3), np.float32),
        }
        # Normalized gray edges for pattern correlation
        self.gray_edges = {
            'top': np.zeros((n, w), np.float32),
            'bot': np.zeros((n, w), np.float32),
            'lef': np.zeros((n, h), np.float32),
            'rig': np.zeros((n, h), np.float32),
        }
        # Flatness stats
        self.edge_std = {'top': [], 'bot': [], 'lef': [], 'rig': []}

        for i in range(n):
            img = self.pieces_list[i]
            img = cv2.GaussianBlur(img, (3, 3), 0)

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # LAB edges
            self.edges['top'][i] = lab[0, :, :]
            self.edges['bot'][i] = lab[-1, :, :]
            self.edges['lef'][i] = lab[:, 0, :]
            self.edges['rig'][i] = lab[:, -1, :]

            # Normalized grayscale edges
            self.gray_edges['top'][i] = self._normalize_vec(gray[0, :])
            self.gray_edges['bot'][i] = self._normalize_vec(gray[-1, :])
            self.gray_edges['lef'][i] = self._normalize_vec(gray[:, 0])
            self.gray_edges['rig'][i] = self._normalize_vec(gray[:, -1])

            # Flatness (mean std across LAB channels along the edge)
            self.edge_std['top'].append(np.mean(np.std(self.edges['top'][i], axis=0)))
            self.edge_std['bot'].append(np.mean(np.std(self.edges['bot'][i], axis=0)))
            self.edge_std['lef'].append(np.mean(np.std(self.edges['lef'][i], axis=0)))
            self.edge_std['rig'].append(np.mean(np.std(self.edges['rig'][i], axis=0)))

    def _normalize_vec(self, v):
        v = v.astype(np.float32)
        std = np.std(v)
        if std < 1e-5:
            return v - np.mean(v)
        return (v - np.mean(v)) / (std + 1e-9)


    def _calculate_hybrid_costs(self):
        n = self.n
        cost_ver = np.full((n, n), np.inf, np.float32)
        cost_hor = np.full((n, n), np.inf, np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # --------- VERTICAL: i bottom -> j top ---------
                color_diff = np.mean(np.abs(self.edges['bot'][i] - self.edges['top'][j]))
                corr = np.dot(self.gray_edges['bot'][i], self.gray_edges['top'][j]) / self.w
                pattern_dist = 1.0 - corr

                pen = 0.0
                # soft vignette
                dark_bot = np.mean(self.edges['bot'][i][:, 0]) < DARKNESS_THRESHOLD
                dark_top = np.mean(self.edges['top'][j][:, 0]) < DARKNESS_THRESHOLD
                if dark_bot and dark_top:
                    pen += VIGNETTE_PENALTY

                # soft flatness
                flat_bot = self.edge_std['bot'][i] < FLATNESS_THRESHOLD
                flat_top = self.edge_std['top'][j] < FLATNESS_THRESHOLD
                if flat_bot and flat_top:
                    pen += FLATNESS_PENALTY

                cost_ver[i, j] = color_diff + 50.0 * pattern_dist + pen

                # --------- HORIZONTAL: i right -> j left ---------
                color_diff_h = np.mean(np.abs(self.edges['rig'][i] - self.edges['lef'][j]))
                corr_h = np.dot(self.gray_edges['rig'][i], self.gray_edges['lef'][j]) / self.h
                pattern_dist_h = 1.0 - corr_h

                pen_h = 0.0
                dark_rig = np.mean(self.edges['rig'][i][:, 0]) < DARKNESS_THRESHOLD
                dark_lef = np.mean(self.edges['lef'][j][:, 0]) < DARKNESS_THRESHOLD
                if dark_rig and dark_lef:
                    pen_h += VIGNETTE_PENALTY

                flat_rig = self.edge_std['rig'][i] < FLATNESS_THRESHOLD
                flat_lef = self.edge_std['lef'][j] < FLATNESS_THRESHOLD
                if flat_rig and flat_lef:
                    pen_h += FLATNESS_PENALTY

                cost_hor[i, j] = color_diff_h + 50.0 * pattern_dist_h + pen_h

        return cost_ver, cost_hor

    def _normalize(self, m):
        valid = m[np.isfinite(m)]
        if len(valid) == 0:
            return m
        mn, mx = np.min(valid), np.max(valid)
        if mx == mn:
            return np.zeros_like(m)
        out = m.copy()
        out[np.isfinite(m)] = (m[np.isfinite(m)] - mn) / (mx - mn)
        return out

    def _compute_ratios(self, cost_matrix):
        n = self.n
        ratios = np.ones((n, n), dtype=np.float32)
        for i in range(n):
            row = cost_matrix[i, :].copy()
            row[i] = np.inf
            sorted_indices = np.argsort(row)
            best_val = row[sorted_indices[0]]
            second_val = row[sorted_indices[1]]

            if not np.isfinite(best_val):
                ratios[i, :] = 1.0
                continue

            if second_val < 1e-5:
                second_val = 1e-5
            ratios[i, sorted_indices[0]] = best_val / second_val
        return ratios

    def _calculate_mutual_best_buddies(self, ver, hor):
        n = self.n
        bb_ver = np.zeros((n, n), dtype=int)
        bb_hor = np.zeros((n, n), dtype=int)

        mins_ver_fwd = np.argmin(ver, axis=1)
        mins_ver_bwd = np.argmin(ver, axis=0)
        mins_hor_fwd = np.argmin(hor, axis=1)
        mins_hor_bwd = np.argmin(hor, axis=0)

        for i in range(n):
            if np.isfinite(ver[i, mins_ver_fwd[i]]):
                if mins_ver_bwd[mins_ver_fwd[i]] == i:
                    bb_ver[i, mins_ver_fwd[i]] = 1

            if np.isfinite(hor[i, mins_hor_fwd[i]]):
                if mins_hor_bwd[mins_hor_fwd[i]] == i:
                    bb_hor[i, mins_hor_fwd[i]] = 1
        return bb_ver, bb_hor

    def _calculate_single_best_buddies(self, ver, hor):
        n = self.n
        sbb_ver = np.zeros((n, n), dtype=int)
        sbb_hor = np.zeros((n, n), dtype=int)
        mins_ver = np.argmin(ver, axis=1)
        mins_hor = np.argmin(hor, axis=1)
        for i in range(n):
            if np.isfinite(ver[i, mins_ver[i]]):
                sbb_ver[i, mins_ver[i]] = 1
            if np.isfinite(hor[i, mins_hor[i]]):
                sbb_hor[i, mins_hor[i]] = 1
        return sbb_ver, sbb_hor

    def _get_dynamic_cost(self, i, j, relation):
        if relation == 'ver':
            raw = self.norm_ver[i, j]
            ratio = self.ratio_ver[i, j]
            is_bb = self.bb_ver[i, j]
        else:
            raw = self.norm_hor[i, j]
            ratio = self.ratio_hor[i, j]
            is_bb = self.bb_hor[i, j]

        score = raw + (WEIGHT_RATIO * ratio)
        if is_bb:
            score -= WEIGHT_MUTUAL_BB
        return score

    def _calculate_grid_score(self, grid):
        score = 0.0
        # Horizontal Links
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS - 1):
                p1, p2 = grid[r, c], grid[r, c + 1]
                if p1 != -1 and p2 != -1:
                    if not np.isfinite(self.raw_hor[p1, p2]):
                        return -np.inf

                    val = 0.0
                    if self.bb_hor[p1, p2]:
                        val += 5.0
                    elif self.sbb_hor[p1, p2]:
                        val += 1.0
                    val -= self.norm_hor[p1, p2] * 2.0
                    score += val

        # Vertical Links
        for r in range(GRID_ROWS - 1):
            for c in range(GRID_COLS):
                p1, p2 = grid[r, c], grid[r + 1, c]
                if p1 != -1 and p2 != -1:
                    if not np.isfinite(self.raw_ver[p1, p2]):
                        return -np.inf

                    val = 0.0
                    if self.bb_ver[p1, p2]:
                        val += 5.0
                    elif self.sbb_ver[p1, p2]:
                        val += 1.0
                    val -= self.norm_ver[p1, p2] * 2.0
                    score += val
        return score

    # ====================================================
    # REFINEMENT & SOLVE (same as V7)
    # ====================================================
    def _refine_grid_swapping(self, grid, max_iters=50):
        current_grid = grid.copy()
        best_score = self._calculate_grid_score(current_grid)
        for _ in range(max_iters):
            improved = False
            coords = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS)]
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    r1, c1 = coords[i]
                    r2, c2 = coords[j]
                    current_grid[r1, c1], current_grid[r2, c2] = current_grid[r2, c2], current_grid[r1, c1]
                    new_score = self._calculate_grid_score(current_grid)
                    if new_score > best_score:
                        best_score = new_score
                        improved = True
                    else:
                        current_grid[r1, c1], current_grid[r2, c2] = current_grid[r2, c2], current_grid[r1, c1]
            if not improved:
                break
        return current_grid

    def _refine_grid_topology(self, grid):
        best_grid = grid.copy()
        best_score = self._calculate_grid_score(grid)
        for r_shift in range(GRID_ROWS):
            for c_shift in range(GRID_COLS):
                rolled = np.roll(grid, r_shift, axis=0)
                rolled = np.roll(rolled, c_shift, axis=1)
                score = self._calculate_grid_score(rolled)
                if score > best_score:
                    best_score = score
                    best_grid = rolled
        return best_grid

    def solve(self):
        candidates = []
        rows, cols = np.where(self.bb_ver == 1)
        for i, j in zip(rows, cols):
            candidates.append({'pair': (i, j), 'rel': 'ver', 'score': self.norm_ver[i, j]})
        rows, cols = np.where(self.bb_hor == 1)
        for i, j in zip(rows, cols):
            candidates.append({'pair': (i, j), 'rel': 'hor', 'score': self.norm_hor[i, j]})
        candidates.sort(key=lambda x: x['score'])

        if not candidates:
            return self._solve_fallback()

        best_final_grid = None
        best_final_score = -np.inf
        attempts = min(5, len(candidates))

        for k in range(attempts):
            seed = candidates[k]
            placed_pieces = {}
            used_ids = set()
            p1, p2 = seed['pair']
            placed_pieces[p1] = (0, 0)
            used_ids.add(p1)
            if seed['rel'] == 'ver':
                placed_pieces[p2] = (0, 1)
            else:
                placed_pieces[p2] = (1, 0)
            used_ids.add(p2)

            while len(placed_pieces) < self.n:
                best_match_score = np.inf
                best_move = None
                open_slots = set()
                for pid, (px, py) in placed_pieces.items():
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = px + dx, py + dy
                        if (nx, ny) not in placed_pieces.values():
                            open_slots.add((nx, ny))

                for (sx, sy) in open_slots:
                    curr_xs = [p[0] for p in placed_pieces.values()] + [sx]
                    curr_ys = [p[1] for p in placed_pieces.values()] + [sy]
                    if (max(curr_xs) - min(curr_xs) >= GRID_COLS) or (max(curr_ys) - min(curr_ys) >= GRID_ROWS):
                        continue

                    neighbors = {}
                    for pid, pos in placed_pieces.items():
                        if pos == (sx - 1, sy):
                            neighbors['left'] = pid
                        elif pos == (sx + 1, sy):
                            neighbors['right'] = pid
                        elif pos == (sx, sy - 1):
                            neighbors['top'] = pid
                        elif pos == (sx, sy + 1):
                            neighbors['bottom'] = pid

                    for candidate in range(self.n):
                        if candidate in used_ids:
                            continue
                        score_sum = 0.0
                        count = 0
                        valid_move = True

                        if 'left' in neighbors:
                            if not np.isfinite(self.raw_hor[neighbors['left'], candidate]):
                                valid_move = False
                            score_sum += self._get_dynamic_cost(neighbors['left'], candidate, 'hor')
                            count += 1
                        if 'right' in neighbors:
                            if not np.isfinite(self.raw_hor[candidate, neighbors['right']]):
                                valid_move = False
                            score_sum += self._get_dynamic_cost(candidate, neighbors['right'], 'hor')
                            count += 1
                        if 'top' in neighbors:
                            if not np.isfinite(self.raw_ver[neighbors['top'], candidate]):
                                valid_move = False
                            score_sum += self._get_dynamic_cost(neighbors['top'], candidate, 'ver')
                            count += 1
                        if 'bottom' in neighbors:
                            if not np.isfinite(self.raw_ver[candidate, neighbors['bottom']]):
                                valid_move = False
                            score_sum += self._get_dynamic_cost(candidate, neighbors['bottom'], 'ver')
                            count += 1

                        if valid_move and count > 0:
                            avg_score = score_sum / count
                            if avg_score < best_match_score:
                                best_match_score = avg_score
                                best_move = (candidate, (sx, sy))

                if best_move:
                    pid, coords = best_move
                    placed_pieces[pid] = coords
                    used_ids.add(pid)
                else:
                    break

            xs = [p[0] for p in placed_pieces.values()]
            ys = [p[1] for p in placed_pieces.values()]
            if not xs:
                continue
            min_x, min_y = min(xs), min(ys)
            temp_grid = np.full((GRID_ROWS, GRID_COLS), -1, dtype=int)
            valid_config = True
            for pid, (px, py) in placed_pieces.items():
                r = py - min_y
                c = px - min_x
                if r < GRID_ROWS and c < GRID_COLS:
                    temp_grid[r, c] = pid
                else:
                    valid_config = False

            if valid_config:
                temp_grid = self._refine_grid_topology(temp_grid)
                temp_grid = self._refine_grid_swapping(temp_grid)
                final_score = self._calculate_grid_score(temp_grid)
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_final_grid = temp_grid

        if best_final_grid is None:
            return self._solve_fallback()
        return self.render(best_final_grid.flatten())

    def _solve_fallback(self):
        return np.zeros((self.h * GRID_ROWS, self.w * GRID_COLS, 3), dtype=np.uint8)

    def render(self, arrangement):
        canvas = np.zeros((self.h * GRID_ROWS, self.w * GRID_COLS, 3), dtype=np.uint8)
        for i, piece_idx in enumerate(arrangement):
            if piece_idx == -1:
                continue
            r, c = divmod(i, GRID_COLS)
            piece = self.pieces_list[piece_idx]
            canvas[r * self.h:(r + 1) * self.h, c * self.w:(c + 1) * self.w] = piece
        return canvas
