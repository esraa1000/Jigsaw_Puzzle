import cv2
import numpy as np
from itertools import permutations

class JigsawSolverHybrid:
    def __init__(self, piece_images, config=None):
        self.piece_ids = list(piece_images.keys())
        self.n = len(self.piece_ids)
        self.pieces_list = [piece_images[pid] for pid in self.piece_ids]
        self.h, self.w = self.pieces_list[0].shape[:2]

        # --- TUNING PARAMETERS ---
        self.VIGNETTE_PENALTY = 50.0       # Penalty for matching dark borders
        self.DARKNESS_THRESHOLD = 60       # LAB Lightness below this is "dark"
        self.FLATNESS_PENALTY = 100.0       # <<< NEW: Penalty for matching two flat/boring edges
        self.FLATNESS_THRESHOLD = 5 # <<< NEW: Color STD below this is "flat"

        # --- Pipeline ---
        self._extract_features()
        self.cost_ver, self.cost_hor = self._calculate_hybrid_costs()

    def _extract_features(self):
        """Extracts LAB edges and calculates their standard deviation for flatness detection."""
        h, w, n = self.h, self.w, self.n

        # Dictionaries to hold edge data
        self.edges = {
            'top': np.zeros((n, w, 3)), 'bot': np.zeros((n, w, 3)),
            'lef': np.zeros((n, h, 3)), 'rig': np.zeros((n, h, 3))
        }
        self.gray_edges = {
            'top': np.zeros((n, w)), 'bot': np.zeros((n, w)),
            'lef': np.zeros((n, h)), 'rig': np.zeros((n, h))
        }
        # <<< NEW: Store Standard Deviation for flatness check
        self.edge_std = { 'top': [], 'bot': [], 'lef': [], 'rig': [] }

        for i in range(n):
            lab_img = cv2.cvtColor(self.pieces_list[i], cv2.COLOR_BGR2LAB).astype(np.float32)
            gray_img = cv2.cvtColor(self.pieces_list[i], cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Store LAB edges
            self.edges['top'][i], self.edges['bot'][i] = lab_img[0, :, :], lab_img[-1, :, :]
            self.edges['lef'][i], self.edges['rig'][i] = lab_img[:, 0, :], lab_img[:, -1, :]

            # Store normalized grayscale edges for correlation
            self.gray_edges['top'][i] = self._normalize(gray_img[0, :])
            self.gray_edges['bot'][i] = self._normalize(gray_img[-1, :])
            self.gray_edges['lef'][i] = self._normalize(gray_img[:, 0])
            self.gray_edges['rig'][i] = self._normalize(gray_img[:, -1])

            # <<< NEW: Calculate and store the average standard deviation across L,A,B channels
            self.edge_std['top'].append(np.mean(np.std(self.edges['top'][i], axis=0)))
            self.edge_std['bot'].append(np.mean(np.std(self.edges['bot'][i], axis=0)))
            self.edge_std['lef'].append(np.mean(np.std(self.edges['lef'][i], axis=0)))
            self.edge_std['rig'].append(np.mean(np.std(self.edges['rig'][i], axis=0)))

    def _normalize(self, vector):
        std = np.std(vector)
        if std < 1e-5: return vector - np.mean(vector)
        return (vector - np.mean(vector)) / std

    def _calculate_hybrid_costs(self):
        n = self.n
        cost_ver = np.full((n, n), np.inf)
        cost_hor = np.full((n, n), np.inf)

        for i in range(n):
            for j in range(n):
                if i == j: continue

                # --- VERTICAL: i's Bottom to j's Top ---
                color_diff = np.mean(np.abs(self.edges['bot'][i] - self.edges['top'][j]))
                corr = np.dot(self.gray_edges['bot'][i], self.gray_edges['top'][j]) / self.w
                pattern_dist = 1.0 - corr

                pen = 0
                # Vignette Penalty (is it a dark border?)
                if np.mean(self.edges['bot'][i][:, 0]) < self.DARKNESS_THRESHOLD or \
                   np.mean(self.edges['top'][j][:, 0]) < self.DARKNESS_THRESHOLD:
                    pen += self.VIGNETTE_PENALTY

                # <<< NEW: Flatness Penalty (is it a boring, flat background?)
                if self.edge_std['bot'][i] < self.FLATNESS_THRESHOLD and \
                   self.edge_std['top'][j] < self.FLATNESS_THRESHOLD:
                    pen += self.FLATNESS_PENALTY

                cost_ver[i, j] = color_diff + (50.0 * pattern_dist) + pen

                # --- HORIZONTAL: i's Right to j's Left ---
                color_diff_h = np.mean(np.abs(self.edges['rig'][i] - self.edges['lef'][j]))
                corr_h = np.dot(self.gray_edges['rig'][i], self.gray_edges['lef'][j]) / self.h
                pattern_dist_h = 1.0 - corr_h

                pen_h = 0
                if np.mean(self.edges['rig'][i][:, 0]) < self.DARKNESS_THRESHOLD or \
                   np.mean(self.edges['lef'][j][:, 0]) < self.DARKNESS_THRESHOLD:
                    pen_h += self.VIGNETTE_PENALTY

                if self.edge_std['rig'][i] < self.FLATNESS_THRESHOLD and \
                   self.edge_std['lef'][j] < self.FLATNESS_THRESHOLD:
                    pen_h += self.FLATNESS_PENALTY

                cost_hor[i, j] = color_diff_h + (50.0 * pattern_dist_h) + pen_h

        return cost_ver, cost_hor

    def solve(self):
        best_arr = None
        min_cost = np.inf
        # Brute force all 24 permutations
        for arr in permutations(range(self.n)):
            p0, p1, p2, p3 = arr
            # Sum the cost of the four internal seams
            cost = (self.cost_hor[p0, p1] + self.cost_hor[p2, p3] +
                    self.cost_ver[p0, p2] + self.cost_ver[p1, p3])

            if cost < min_cost:
                min_cost = cost
                best_arr = arr

        if best_arr is None: best_arr = (0, 1, 2, 3) # Failsafe

        # The solver only needs to return the final image
        return self.render(best_arr)

    def render(self, arrangement):
        p_tl, p_tr = self.pieces_list[arrangement[0]], self.pieces_list[arrangement[1]]
        p_bl, p_br = self.pieces_list[arrangement[2]], self.pieces_list[arrangement[3]]
        ph, pw = p_tl.shape[:2]
        canvas = np.zeros((ph * 2, pw * 2, 3), dtype=np.uint8)
        canvas[0:ph, 0:pw] = p_tl
        canvas[0:ph, pw:pw*2] = p_tr
        canvas[ph:ph*2, 0:pw] = p_bl
        canvas[ph:ph*2, pw:pw*2] = p_br
        return canvas