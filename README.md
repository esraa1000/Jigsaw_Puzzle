# Jigsaw_Puzzle

## 1. Project Overview
This project is a Python-based solver for jigsaw puzzles from the cartoon show *Gravity Falls*. The dataset contains three types of puzzles: **2×2, 4×4, and 8×8**, meaning each puzzle consists of 4, 16, or 64 pieces.

The goal of the project is to **solve these puzzles using image processing techniques only**, without employing machine learning or artificial intelligence.

## 2. Our Approach
We divided the project into **three separate pipelines**, one for each puzzle type. This decision is motivated by the **difference in puzzle complexity**:

- Techniques suitable for 4×4 or 8×8 puzzles would **overcomplicate** the simpler 2×2 puzzles, potentially introducing noise and reducing accuracy.
- By customizing the solver per puzzle type, we could optimize **speed and reliability** for each scenario.

In the following sections, we break down each pipeline, discuss the **techniques used**, and show how we addressed **failure cases** to improve the solver.

## 3. The 2x2 solver
###Our first Attempt: Descriptor-Based Brute Force Attempt
For the 2x2 puzzles, we initially attempted a **brute-force solver using handcrafted descriptors**

1. **Edge Descriptors**
   - After dividing the puzzles based off their dimensions, each puzzle piece's edges were described using:
     - `mean_profile`: the mean color along the edge.
     - `grad_hist`: gradient histogram along the edge.
     - `lab_hist`: LAB color histogram along the edge.
   - These features were combined into a weighted score using the `edge_score` function.
2. **Permutation and Flip Search**
   -We tried all the permutations of the arrangement of the pieces in the 2x2 grid.
   -We tested using 4 different flip configurations:
     -No flip
     -Left-right flip
     -Top-bottom flip
     -full 180° rotation
   Note: These flips were to make it more generic but the pieces provided in the dataset were correctlt oriented in the first place. We then discovered that it adds unnecessary computation, so, this step can be ignored.
3. **Scoring**
   -For each arrangement, the **total score** was computed as the sum of edge differences between adjacent pieces:
     - Top-left -> Top-right
     - Bottom-left -> Bottom-right
     - Top-left -> Bottom-left
     - Top-right -> Bottom-right
   -The arrangement with the **lowest total edge score** (highest similarity) was selected.
4. **Reconstruction**
   - Pieces were places in a 2x2 canvas according to the best permutation.

#### Challenges and Failure cases
  - While this approach is exhaustive, it relied heavily on the descriptors.
  - In some images, edges had similar colors or gradients, causing mismatches.
    -####The main problem:
      
       Pieces were correctly matched locally but not aligned globally, meaning, like the image shown below, the pieces were correctly matched vertically but the right and left were flipped, this was the case for at least 10 pictures, so the overall accuracy was a little below 80%, so this method was doing generally well, but failing in cases where the background is dominant like in this picture.
     <img width="999" height="480" alt="image" src="https://github.com/user-attachments/assets/7cdf49af-cfd9-440a-b1a5-5e66723110ff" />


### Our Second Attempt: SSD-Only Brute-Force Solver

## Overview

This solver attempts to solve the 2x2 jigsaw puzzles using **Sum of Squared Differences (SSD)** on the outer edges of the pieces, combined with a **Best Buddy (BB) bonus** to encourage mutually strong matches. From what we learned from pur first attempt is that are descriptors are adding a lot of metrics into account that may be confusing the solver but didn't resolve the global ambiguity, and sometimes even made things worse because they emphasized minor differences that weren't relevant alignment. That's why switching to SSD with a Best Buddy check helped improve consistency: SSD focus directly on edge differences and the BB bonus encourages mutually strong matches, which partially enforces global consistency.

---
### How the Solver Works
1. **Feature Extraction**
  -Convert each puzzle piece to LAB color space; which is a device-independent model that describes colors based on human perception. The key components are L* (lightness), a* (Red-Green Axis), b* (Yellow-Blue Axis).
  -Like the descriptors, it onlt extracts the outermost edges or bounds of the image.
  -Then, these edges are stored in arrays: `ssd_tops`, `ssd_bottoms`, `ssd_lefts` and `ssd_rights`.
2. **Compute Raw SSD Matrices**
   - For each pair of pieces `(i, j)`:
     - **Vertical SSD**: compare `i`'s bottom edge with `j`'s top edge.
     - **Horizontal** SSD: compare `i`'s right edge with `j`'s left edge.
   - Results stored in `raw_ssd_ver` and `raw_ssd_hor`.
3. **Normalize SSD Scores**
   - Min-max normalization scales all SSD scores to `[0,1]`.
   - Normalized scores: `norm_ssd_ver` and `norm_ssd_hor`.

4. **Best Buddy Detection**
   - Pieces `i` and `j` are **mutual best buddies** if:
     - `j` is the best vertical/horizontal match for `i` **and**
     - `i` is the best vertical/horizontal match for `j`.
   - Matrices `bb_ver` and `bb_hor` store these relationships.

5. **Arrangement Cost Calculation**
   - For each 2×2 arrangement `[TL, TR, BL, BR]`:
     - Local SSD cost = sum of SSD for the 4 touching edges (TL→TR, BL→BR, TL→BL, TR→BR).
     - BB bonus = count of edges where pieces are mutual best buddies.
     - Total cost = `(SSD cost) - (BB bonus * weight)`.
   - Lower cost → better arrangement.

6. **Brute-Force Search**
   - All `4! = 24` permutations of the 4 pieces are evaluated.
   - Arrangement with the **lowest total cost** is chosen as the solution.

7. **Rendering**
   - Pieces are placed on a 2×2 canvas according to the chosen arrangement.
   - The solved puzzle image is returned.

---

### Evaluation
The `evaluate_solver_v0` function compares each solved puzzle to the **ground truth**:

- **Success**: whether the puzzle matches exactly (pixel difference < threshold).
- **Average accuracy**: computed from pixel differences.
- Displays **top failures and successes** for visual inspection.
#### THE OVERALL ACCURACY WAS BETTER, reaching a succes rate of 88.54%.

---

### Advantages
- Simple and deterministic.
- Works well when edges are visually distinct.
- Brute-force ensures all arrangements are checked.

### Limitations and failure cases
- Uses only **1-pixel outer edges**, so similar edges may cause mismatches.
- Ignores global structure; pieces may match locally but be **flipped globally**.
- Best Buddy bonus improves matches but cannot fully resolve ambiguous cases.
Still the same observation of global misalignment was present but less, like the image shown below:
<img width="995" height="483" alt="image" src="https://github.com/user-attachments/assets/8320b67b-271e-468f-9e54-06b91a0464a0" />

# Our final attempt: Hybrid Solver: ZNCC Pattern Matching + Vignette Penalty 
Our final approach combined color matching, pattern correlation, and vignette awareness, which resulted in a success rate of 98.18% on the 2x2 puzzles.

The previous descriptor-based and SSD-only methods suffered from global misalignment, especially in images with large uniform backgrounds. Pieces could match locally but be flipped or swapped horizontally/vertically. We needed a method that preserved local edge alignment, respected global orinetation, and avoided placing dark or vignetted edges in central positions where they would confuse the solver.

## The key techniques used:
1. Like previous attempts, the boundaries of the pieces were extracted and converted to the LAB space.
2. Added Pattern Matching via ZNCC:
  Zero Normalized Cross Correlation (ZNCC) along adjacent edges to match patterns (e.g., tree lines, textures), independent of exact lighting.
  This helped us make a distance metric: 1 - correlation (0=perfect match, 1=completely uncorrelated)
3. We measured the average Lightness along each edge, edges below a darkness threshold are considered vignetted and heavily penalized if placed internally, avoiding misalignment due to misleading low-contrast edges.
4. After all this computed, the cost is computed, using this formula: total_cost = color_diff + (50 * pattern_dist) + vignette_penalty
5. After that, like in the first attempt, the puzzle was assembled using Brute Force.
6. We finetuned the parameters and were able to reduce failure cases from 6 to just 2.

Images that were previously left unassembled were finally constructed:

<img width="473" height="476" alt="image" src="https://github.com/user-attachments/assets/99d9a6b1-b329-441b-bfa1-410c76bd67bb" />
<img width="472" height="467" alt="image" src="https://github.com/user-attachments/assets/ef10c02e-d210-4cfb-96f1-164cf11c1d60" />




Final Comparison:

| Method             | Main Issue                                                                                               | Success Rate |
| ------------------ | -------------------------------------------------------------------------------------------------------- | ------------ |
| Descriptor-based   | Local matches correct, global alignment often wrong, background ambiguity                                | ~80%         |
| SSD + Best Buddy   | Better local-global consistency, but could fail with dark/vignetted edges                                | ~90–95%      |
| **Hybrid (Final)** | Combines color, pattern correlation, and dark-edge penalties to ensure both local and global correctness | **98.3%**    |




# Conclusion for 2x2 puzzles
Through a series of iterative approaches, we demonstarted that solving 2x2 jigsaw puzzles with image processing alone is feasible, even without machine learning. Starting from descriptor-based matching, we encountered challenged with global misalignment and ambiguous edges. Transitioning to an SSD-based solver with Best Buddy logic improved consistency, but certain dark or uniform regions caused errors. 

Our final hybrid solver, combining color matching, pattern correlation (ZNCC), and vignette-aware penalties, achieved a success rate of 98.3%, effectively handling both local edge compatibility and global piece arrangement. Only a few edge cases with extreme vignetting remained unsolved, highlighting the robustness and precision of our approach.

This part illustrates the importance of combining complementary techniques, carefully analyzing failure cases, and designing domain-specific heuristics to maximize solver performance. 



  



     


   










