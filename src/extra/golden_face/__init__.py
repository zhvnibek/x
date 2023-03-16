"""
https://github.com/Aksoylu/GoldenFace
"""

import GoldenFace


def apply_golden_face_masks(image_filepath: str):
    color_a = (0, 0, 255)

    masked_images = []

    for mask in ['drawFaceCover', 'drawLandmark', 'drawMask', 'drawTGSM', 'drawVFM', 'drawTZM', 'drawLC', 'drawTSM']:
        gf = GoldenFace.goldenFace(image_filepath)
        getattr(gf, mask)(color_a)
        masked_images.append(gf.img)

    return masked_images
