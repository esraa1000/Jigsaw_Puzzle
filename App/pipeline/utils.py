import numpy as np

def split_grid(image, rows, cols):
    """
    Split an image into rows x cols equally sized pieces.
    Returns: dict {id: piece_image}
    """
    h, w = image.shape[:2]
    ph, pw = h // rows, w // cols

    pieces = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            pieces[idx] = image[
                r * ph:(r + 1) * ph,
                c * pw:(c + 1) * pw
            ]
            idx += 1
    return pieces
