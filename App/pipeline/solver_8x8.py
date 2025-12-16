import cv2
import numpy as np


GRID_ROWS = 8
GRID_COLS = 8

WEIGHT_RATIO     = 12.0
WEIGHT_MUTUAL_BB = 8.0
WEIGHT_GRADIENT  = 0.8

class JigsawSolver8x8:
    def __init__(self, piece_images):
        """
        piece_images: dict {id: image}
        """
        self.piece_ids = list(piece_images.keys())
        self.n = len(self.piece_ids)
        self.pieces_list = [piece_images[i] for i in self.piece_ids]

        sample = self.pieces_list[0]
        self.h, self.w = sample.shape[:2]

        # Pipeline
        self._extract_features()
        self.raw_ver, self.raw_hor = self._calculate_complex_costs()
        self.norm_ver = self._normalize(self.raw_ver)
        self.norm_hor = self._normalize(self.raw_hor)
        self.bb_ver, self.bb_hor = self._calculate_mutual_best_buddies(self.raw_ver, self.raw_hor)
        self.sbb_ver, self.sbb_hor = self._calculate_single_best_buddies(self.raw_ver, self.raw_hor)
        self.ratio_ver = self._compute_ratios(self.raw_ver)
        self.ratio_hor = self._compute_ratios(self.raw_hor)


    def _extract_features(self):
        self.feat_edges = []

        for img in self.pieces_list:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

            top = lab[0, :, :]
            bottom = lab[-1, :, :]
            left = lab[:, 0, :]
            right = lab[:, -1, :]

            self.feat_edges.append({
                'top': top, 'top_in': lab[1, :, :], 'std_top': np.mean(np.std(top, axis=1)),
                'bottom': bottom, 'bottom_in': lab[-2, :, :], 'std_bottom': np.mean(np.std(bottom, axis=1)),
                'left': left, 'left_in': lab[:, 1, :], 'std_left': np.mean(np.std(left, axis=1)),
                'right': right, 'right_in': lab[:, -2, :], 'std_right': np.mean(np.std(right, axis=1))
            })

    def _calculate_complex_costs(self):
        n = self.n
        cost_ver = np.full((n, n), np.inf)
        cost_hor = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # Vertical
                d_col_v = np.sum(np.abs(self.feat_edges[i]['bottom'] - self.feat_edges[j]['top']))
                grad_i_v = self.feat_edges[i]['bottom'] - self.feat_edges[i]['bottom_in']
                grad_j_v = self.feat_edges[j]['top'] - self.feat_edges[j]['top_in']
                d_grad_v = np.sum(np.abs(grad_i_v - grad_j_v))

                detail_penalty = 0.0
                if self.feat_edges[i]['std_bottom'] < 5.0 and self.feat_edges[j]['std_top'] < 5.0:
                    detail_penalty = 500.0

                cost_ver[i, j] = d_col_v + (WEIGHT_GRADIENT * d_grad_v) + detail_penalty

                # Horizontal
                d_col_h = np.sum(np.abs(self.feat_edges[i]['right'] - self.feat_edges[j]['left']))
                grad_i_h = self.feat_edges[i]['right'] - self.feat_edges[i]['right_in']
                grad_j_h = self.feat_edges[j]['left'] - self.feat_edges[j]['left_in']
                d_grad_h = np.sum(np.abs(grad_i_h - grad_j_h))

                detail_penalty_h = 0.0
                if self.feat_edges[i]['std_right'] < 5.0 and self.feat_edges[j]['std_left'] < 5.0:
                    detail_penalty_h = 500.0

                cost_hor[i, j] = d_col_h + (WEIGHT_GRADIENT * d_grad_h) + detail_penalty_h

        return cost_ver, cost_hor

    # ==========================================================
    # HELPERS
    # ==========================================================
    def _normalize(self, m):
        valid = m[np.isfinite(m)]
        if len(valid) == 0:
            return m
        mn, mx = np.min(valid), np.max(valid)
        if mx == mn:
            return np.zeros_like(m)
        out = m.copy()
        out[np.isfinite(m)] = (m[np.isfinite(m)] - mn) / (mx - mn + 1e-9)
        return out

    def _compute_ratios(self, cost):
        n = self.n
        ratios = np.ones((n, n))
        for i in range(n):
            row = cost[i].copy()
            row[i] = np.inf
            idx = np.argsort(row)
            ratios[i, idx[0]] = row[idx[0]] / (row[idx[1]] + 1e-9)
        return ratios

    def _calculate_mutual_best_buddies(self, ver, hor):
        n = self.n
        bbv = np.zeros((n, n), int)
        bbh = np.zeros((n, n), int)

        v_f = np.argmin(ver, axis=1)
        v_b = np.argmin(ver, axis=0)
        h_f = np.argmin(hor, axis=1)
        h_b = np.argmin(hor, axis=0)

        for i in range(n):
            if v_b[v_f[i]] == i:
                bbv[i, v_f[i]] = 1
            if h_b[h_f[i]] == i:
                bbh[i, h_f[i]] = 1

        return bbv, bbh

    def _calculate_single_best_buddies(self, ver, hor):
        n = self.n
        sbbv = np.zeros((n, n), int)
        sbbh = np.zeros((n, n), int)
        for i in range(n):
            sbbv[i, np.argmin(ver[i])] = 1
            sbbh[i, np.argmin(hor[i])] = 1
        return sbbv, sbbh

    def _get_dynamic_cost(self, i, j, relation):
        if relation == 'ver':
            base  = self.norm_ver[i, j]
            ratio = self.ratio_ver[i, j]
            bb    = self.bb_ver[i, j]
        else:
            base  = self.norm_hor[i, j]
            ratio = self.ratio_hor[i, j]
            bb    = self.bb_hor[i, j]

        score = base + (WEIGHT_RATIO * ratio)
        if bb:
            score -= WEIGHT_MUTUAL_BB
        return score

    def _calculate_grid_score(self, grid):
        score = 0.0
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                p = grid[r, c]
                if p == -1:
                    continue
                # Right link
                if c < GRID_COLS - 1:
                    pr = grid[r, c + 1]
                    if pr != -1:
                        if self.bb_hor[p, pr]:
                            score += 5.0
                        elif self.sbb_hor[p, pr]:
                            score += 1.0
                        score -= self.norm_hor[p, pr] * 3.0
                # Bottom link
                if r < GRID_ROWS - 1:
                    pb = grid[r + 1, c]
                    if pb != -1:
                        if self.bb_ver[p, pb]:
                            score += 5.0
                        elif self.sbb_ver[p, pb]:
                            score += 1.0
                        score -= self.norm_ver[p, pb] * 3.0
        return score

    def _refine_strip_shifting(self, grid):
        current_grid = grid.copy()
        best_score = self._calculate_grid_score(current_grid)
        improved = True

        while improved:
            improved = False
            # Shift columns
            for c in range(GRID_COLS):
                col = current_grid[:, c].copy()
                best_shift = 0
                col_best_score = best_score
                for s in range(1, GRID_ROWS):
                    current_grid[:, c] = np.roll(col, s)
                    score = self._calculate_grid_score(current_grid)
                    if score > col_best_score:
                        col_best_score = score
                        best_shift = s
                if best_shift != 0:
                    current_grid[:, c] = np.roll(col, best_shift)
                    best_score = col_best_score
                    improved = True
                else:
                    current_grid[:, c] = col

            # Shift rows
            for r in range(GRID_ROWS):
                row = current_grid[r, :].copy()
                best_shift = 0
                row_best_score = best_score
                for s in range(1, GRID_COLS):
                    current_grid[r, :] = np.roll(row, s)
                    score = self._calculate_grid_score(current_grid)
                    if score > row_best_score:
                        row_best_score = score
                        best_shift = s
                if best_shift != 0:
                    current_grid[r, :] = np.roll(row, best_shift)
                    best_score = row_best_score
                    improved = True
                else:
                    current_grid[r, :] = row

        return current_grid

    def _refine_swapping_lines(self, grid):
        current_grid = grid.copy()
        best_score = self._calculate_grid_score(current_grid)
        improved = True

        while improved:
            improved = False
            # Swap columns
            for c1 in range(GRID_COLS):
                for c2 in range(c1 + 1, GRID_COLS):
                    current_grid[:, [c1, c2]] = current_grid[:, [c2, c1]]
                    score = self._calculate_grid_score(current_grid)
                    if score > best_score:
                        best_score = score
                        improved = True
                    else:
                        current_grid[:, [c1, c2]] = current_grid[:, [c2, c1]]

            # Swap rows
            for r1 in range(GRID_ROWS):
                for r2 in range(r1 + 1, GRID_ROWS):
                    current_grid[[r1, r2], :] = current_grid[[r2, r1], :]
                    score = self._calculate_grid_score(current_grid)
                    if score > best_score:
                        best_score = score
                        improved = True
                    else:
                        current_grid[[r1, r2], :] = current_grid[[r2, r1], :]

        return current_grid


    def solve(self):
        # Candidate generation from mutual best‑buddies
        candidates = []
        rows, cols = np.where(self.bb_ver == 1)
        for i, j in zip(rows, cols):
            candidates.append({'pair': (i, j), 'rel': 'ver', 'score': self.norm_ver[i, j]})
        rows, cols = np.where(self.bb_hor == 1)
        for i, j in zip(rows, cols):
            candidates.append({'pair': (i, j), 'rel': 'hor', 'score': self.norm_hor[i, j]})

        # Fallback if too few candidates
        if len(candidates) < 5:
            rows, cols = np.where(self.sbb_ver == 1)
            for i, j in zip(rows, cols):
                candidates.append({'pair': (i, j), 'rel': 'ver', 'score': self.norm_ver[i, j]})

        candidates.sort(key=lambda x: x['score'])

        best_final_grid = None
        best_final_score = -np.inf
        attempts = min(12, len(candidates))

        for k in range(attempts):
            seed = candidates[k]
            placed = {}
            used = set()
            p1, p2 = seed['pair']
            placed[p1] = (0, 0)
            used.add(p1)
            if seed['rel'] == 'ver':
                placed[p2] = (0, 1)
            else:
                placed[p2] = (1, 0)
            used.add(p2)

            # Greedy growth
            while len(placed) < self.n:
                best_avg = np.inf
                best_move = None
                slots = set()
                for pid, (px, py) in placed.items():
                    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                        nx, ny = px + dx, py + dy
                        if (nx, ny) not in placed.values():
                            slots.add((nx, ny))

                for (sx, sy) in slots:
                    xs = [p[0] for p in placed.values()] + [sx]
                    ys = [p[1] for p in placed.values()] + [sy]
                    if (max(xs) - min(xs) >= GRID_COLS) or (max(ys) - min(ys) >= GRID_ROWS):
                        continue

                    neighbors = {}
                    for pid, pos in placed.items():
                        if pos == (sx - 1, sy):
                            neighbors['left'] = pid
                        elif pos == (sx + 1, sy):
                            neighbors['right'] = pid
                        elif pos == (sx, sy - 1):
                            neighbors['top'] = pid
                        elif pos == (sx, sy + 1):
                            neighbors['bottom'] = pid

                    for cand in range(self.n):
                        if cand in used:
                            continue
                        cost = 0.0
                        count = 0
                        if 'left' in neighbors:
                            cost += self._get_dynamic_cost(neighbors['left'], cand, 'hor')
                            count += 1
                        if 'right' in neighbors:
                            cost += self._get_dynamic_cost(cand, neighbors['right'], 'hor')
                            count += 1
                        if 'top' in neighbors:
                            cost += self._get_dynamic_cost(neighbors['top'], cand, 'ver')
                            count += 1
                        if 'bottom' in neighbors:
                            cost += self._get_dynamic_cost(cand, neighbors['bottom'], 'ver')
                            count += 1

                        if count > 0:
                            avg = cost / count
                            if avg < best_avg:
                                best_avg = avg
                                best_move = (cand, (sx, sy))

                if best_move:
                    pid, coords = best_move
                    placed[pid] = coords
                    used.add(pid)
                else:
                    break

            # Convert to grid
            xs = [p[0] for p in placed.values()]
            ys = [p[1] for p in placed.values()]
            if not xs:
                continue
            min_x, min_y = min(xs), min(ys)

            temp_grid = np.full((GRID_ROWS, GRID_COLS), -1, dtype=int)
            valid = True
            for pid, (px, py) in placed.items():
                r, c = py - min_y, px - min_x
                if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
                    temp_grid[r, c] = pid
                else:
                    valid = False

            if valid:
                temp_grid = self._refine_strip_shifting(temp_grid)
                temp_grid = self._refine_swapping_lines(temp_grid)
                score = self._calculate_grid_score(temp_grid)
                if score > best_final_score:
                    best_final_score = score
                    best_final_grid = temp_grid

        if best_final_grid is None:
            return self.render(np.arange(self.n))

        return self.render(best_final_grid.flatten())

    def render(self, arrangement):
        canvas = np.zeros((self.h * GRID_ROWS, self.w * GRID_COLS, 3), dtype=np.uint8)
        for i, idx in enumerate(arrangement):
            r, c = divmod(i, GRID_COLS)
            canvas[
                r * self.h:(r + 1) * self.h,
                c * self.w:(c + 1) * self.w
            ] = self.pieces_list[idx]
        return canvas

# Clean public alias for GUI import
class JigsawSolver8x8Clean(JigsawSolver8x8):
    pass
