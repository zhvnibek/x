"""
- https://github.com/andfanilo/streamlit-drawable-canvas
- https://developer.mozilla.org/en-US/docs/Web/SVG/Attribute/d#moveto_path_commands
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import svgpathtools
from streamlit_drawable_canvas import st_canvas
from functools import lru_cache

from fegan.model import Model


@lru_cache()
def get_model():
    return Model()


st.sidebar.title(f"Victara AI")

st.title('Show me your face!')

image_filepath = 'face.jpg'
bg_image = None

picture = st.camera_input("First, take a picture...")
if picture:
    with open(image_filepath, "wb+") as f:
        f.write(picture.read())
    bg_image = Image.open(image_filepath)


def make_mask(mask_paths):
    mask = np.zeros((512, 512, 3))

    for points in mask_paths:
        for i in range(1, len(points)):
            cv2.line(mask, points[i - 1], points[i], (255, 255, 255), 10)

    mask = np.asarray(mask[:, :, 0] / 255, dtype=np.uint8)
    mask = np.expand_dims(mask, axis=2)
    mask = np.expand_dims(mask, axis=0)
    return mask


def make_sketch(sketch_paths):
    sketch = np.zeros((512, 512, 3))

    for points in sketch_paths:
        for i in range(1, len(points)):
            cv2.line(sketch, points[i - 1], points[i], (255, 255, 255), 1)

    sketch = np.asarray(sketch[:, :, 0] / 255, dtype=np.uint8)
    sketch = np.expand_dims(sketch, axis=2)
    sketch = np.expand_dims(sketch, axis=0)
    return sketch


def make_stroke(stroke_paths):
    """
    Not used yet. so zero tensor is returned.
    """
    stroke = np.zeros((512, 512, 3))

    stroke = stroke / 127.5 - 1
    stroke = np.expand_dims(stroke, axis=0)
    return stroke


def make_noise():
    noise = np.zeros([512, 512, 1], dtype=np.uint8)
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = np.expand_dims(noise, axis=0)
    return noise


if bg_image:
    canvas_result = st_canvas(
        stroke_color="yellow",
        stroke_width=3,
        background_image=bg_image,
        height=512,
        width=512,
        # height=bg_image.height,
        # width=bg_image.width,
        drawing_mode="freedraw",
        key="compute_arc_length",
    )
    if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
        sketch_paths = []

        paths = pd.json_normalize(canvas_result.json_data["objects"])["path"].tolist()
        for ind, path in enumerate(paths):
            path = svgpathtools.parse_path(" ".join([str(e) for line in path for e in line]))
            sketch_paths.append([
                (int(path.point(pos).real), int(path.point(pos).imag))
                for pos in np.linspace(0, 1, int(path.length()))
            ])

        # prepare original input image as tensor
        mat_img = cv2.imread(image_filepath)
        mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        mat_img = mat_img / 127.5 - 1
        mat_img = np.expand_dims(mat_img, axis=0)

        # prepare sketch, color, mask, noise as tensors
        sketch = make_sketch(sketch_paths=sketch_paths)
        color = make_stroke(stroke_paths=sketch_paths)
        mask = make_mask(mask_paths=sketch_paths)

        noise = make_noise()

        sketch = sketch * mask
        color = color * mask
        noise = noise * mask

        # concatenate all tensors as a single input tensor for FEGAN model
        batch = np.concatenate(
            [
                mat_img,  # original image, 0:3
                sketch,  # sketch, 3:4
                color,  # color, 4:7
                mask,  # mask, 7:8
                noise  # noise, 8:9
            ],
            axis=3
        )
        print(batch.shape)

        fegan = get_model()
        result = fegan.demo(batch)
        result = (result + 1) * 127.5
        output_img = np.asarray(result[0, :, :, :], dtype=np.uint8)
        print(output_img.shape)
        cv2.imwrite('output.jpg', output_img)
        st.image(output_img, channels='BGR')
