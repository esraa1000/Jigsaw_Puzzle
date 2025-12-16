import cv2
import numpy as np

# ==========================================================
# CONFIG (8x8 ONLY)
# ==========================================================
GRID_ROWS = 8
GRID_COLS = 8

WEIGHT_RATIO = 12.0
WEIGHT_MUTUAL_BB = 8.0
WEIGHT_GRADIENT = 0.8


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

    # ==========================================================
    # FEATURE EXTRACTION
    # ==========================================================
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

    # ==========================================================
    # COST COMPUTATION
    # ==========================================================
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

                detail_penalty = 0
                if self.feat_edges[i]['std_bottom'] < 5.0 and self.feat_edges[j]['std_top'] < 5.0:
                    detail_penalty = 500.0

                cost_ver[i, j] = d_col_v + (WEIGHT_GRADIENT * d_grad_v) + detail_penalty

                # Horizontal
                d_col_h = np.sum(np.abs(self.feat_edges[i]['right'] - self.feat_edges[j]['left']))
                grad_i_h = self.feat_edges[i]['right'] - self.feat_edges[i]['right_in']
                grad_j_h = self.feat_edges[j]['left'] - self.feat_edges[j]['left_in']
                d_grad_h = np.sum(np.abs(grad_i_h - grad_j_h))

                detail_penalty_h = 0
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

    # ==========================================================
    # SOLVER
    # ==========================================================
    def solve(self):
        grid = np.arange(self.n).reshape(GRID_ROWS, GRID_COLS)
        return self.render(grid.flatten())

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
