import cv2
import numpy as np

GRID_ROWS = 4
GRID_COLS = 4

WEIGHT_RATIO = 30.0
WEIGHT_MUTUAL_BB = 15.0
WEIGHT_GRADIENT = 1.0
WEIGHT_L_CHANNEL = 2.0

VIGNETTE_PENALTY = 1e9
FIXED_DARK_THRESH = 30


class JigsawSolver4x4:
    def __init__(self, piece_images):
        self.pieces_dict = piece_images
        self.piece_ids = list(piece_images.keys())
        self.n = len(self.piece_ids)
        self.pieces_list = [piece_images[pid] for pid in self.piece_ids]

        self.h, self.w = self.pieces_list[0].shape[:2]

        self._extract_features()
        self.raw_ver, self.raw_hor = self._calculate_complex_costs()
        self.norm_ver = self._normalize(self.raw_ver)
        self.norm_hor = self._normalize(self.raw_hor)
        self.ratio_ver = self._compute_ratios(self.raw_ver)
        self.ratio_hor = self._compute_ratios(self.raw_hor)
        self.bb_ver, self.bb_hor = self._calculate_mutual_best_buddies(self.raw_ver, self.raw_hor)
        self.sbb_ver, self.sbb_hor = self._calculate_single_best_buddies(self.raw_ver, self.raw_hor)

    # ================= FEATURE EXTRACTION =================
    def _extract_features(self):
        self.feat_edges = []
        self.is_dark = {'top': [], 'bottom': [], 'left': [], 'right': []}

        for img in self.pieces_list:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

            t, b = lab[0], lab[-1]
            l, r = lab[:, 0], lab[:, -1]

            self.feat_edges.append({
                'top': t, 'top_in': lab[1],
                'bottom': b, 'bottom_in': lab[-2],
                'left': l, 'left_in': lab[:, 1],
                'right': r, 'right_in': lab[:, -2],
            })

            piece_L = np.mean(lab[:, :, 0])
            thresh = min(FIXED_DARK_THRESH, piece_L * 0.6)

            self.is_dark['top'].append(np.mean(t[:, 0]) < thresh)
            self.is_dark['bottom'].append(np.mean(b[:, 0]) < thresh)
            self.is_dark['left'].append(np.mean(l[:, 0]) < thresh)
            self.is_dark['right'].append(np.mean(r[:, 0]) < thresh)

    # ================= COST COMPUTATION =================
    def _calculate_complex_costs(self):
        n = self.n
        ver = np.full((n, n), np.inf)
        hor = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                diff = np.abs(self.feat_edges[i]['bottom'] - self.feat_edges[j]['top'])
                color = diff[:, 0] * WEIGHT_L_CHANNEL + diff[:, 1] + diff[:, 2]
                grad = np.abs(
                    (self.feat_edges[i]['bottom'] - self.feat_edges[i]['bottom_in']) -
                    (self.feat_edges[j]['top'] - self.feat_edges[j]['top_in'])
                )

                penalty = VIGNETTE_PENALTY if (
                    self.is_dark['bottom'][i] or self.is_dark['top'][j]
                ) else 0

                ver[i, j] = np.sum(color) + WEIGHT_GRADIENT * np.sum(grad) + penalty

                diff_h = np.abs(self.feat_edges[i]['right'] - self.feat_edges[j]['left'])
                color_h = diff_h[:, 0] * WEIGHT_L_CHANNEL + diff_h[:, 1] + diff_h[:, 2]
                grad_h = np.abs(
                    (self.feat_edges[i]['right'] - self.feat_edges[i]['right_in']) -
                    (self.feat_edges[j]['left'] - self.feat_edges[j]['left_in'])
                )

                penalty_h = VIGNETTE_PENALTY if (
                    self.is_dark['right'][i] or self.is_dark['left'][j]
                ) else 0

                hor[i, j] = np.sum(color_h) + WEIGHT_GRADIENT * np.sum(grad_h) + penalty_h

        return ver, hor

    # ================= HELPERS =================
    def _normalize(self, m):
        valid = m[np.isfinite(m)]
        if len(valid) == 0:
            return m
        mn, mx = np.min(valid), np.max(valid)
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
    
    def _calculate_grid_score(self, grid):
      score = 0.0
      for r in range(GRID_ROWS):
          for c in range(GRID_COLS):
              p = grid[r, c]
              if c < GRID_COLS - 1:
                  pr = grid[r, c+1]
                  score -= self.norm_hor[p, pr]
                  if self.bb_hor[p, pr]: score += 5
              if r < GRID_ROWS - 1:
                  pb = grid[r+1, c]
                  score -= self.norm_ver[p, pb]
                  if self.bb_ver[p, pb]: score += 5
      return score


    # ================= SOLVE =================
    def solve(self):
        best_grid = None
        best_score = -np.inf

        for _ in range(200):  # random restarts
            grid = np.random.permutation(self.n).reshape(GRID_ROWS, GRID_COLS)
            score = self._calculate_grid_score(grid)
            if score > best_score:
                best_score = score
                best_grid = grid.copy()

        return self.render(best_grid.flatten())


    def render(self, arrangement):
        canvas = np.zeros((self.h * GRID_ROWS, self.w * GRID_COLS, 3), dtype=np.uint8)
        for i, idx in enumerate(arrangement):
            r, c = divmod(i, GRID_COLS)
            canvas[
                r*self.h:(r+1)*self.h,
                c*self.w:(c+1)*self.w
            ] = self.pieces_list[idx]
        return canvas
