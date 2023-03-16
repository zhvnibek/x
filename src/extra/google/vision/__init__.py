import io
from typing import List

from google.cloud import vision
from PIL import Image, ImageDraw
from google.cloud.vision_v1 import FaceAnnotation

service_account_json = "/home/user/Desktop/projects/beautyx/friendly-joy-373716-3734c30398e6.json"


def detect_faces(path):
    """
    Detects faces in an image.
    - https://cloud.google.com/vision/docs/face-tutorial
    - https://github.com/GoogleCloudPlatform/python-docs-samples/blob/HEAD/vision/snippets/face_detection/faces.py
    """

    client = vision.ImageAnnotatorClient.from_service_account_json(service_account_json)

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces: List[FaceAnnotation] = response.face_annotations

    im = Image.open(path)
    draw = ImageDraw.Draw(im)
    # Specify the font-family and the font-size
    for face in faces:
        box = [(vertex.x, vertex.y)
               for vertex in face.bounding_poly.vertices]
        draw.line(box + [box[0]], width=5, fill='#00ff00')
        # Place the confidence value/score of the detected faces above the
        # detection box in the output image
        draw.text(((face.bounding_poly.vertices)[0].x,
                   (face.bounding_poly.vertices)[0].y - 30),
                  str(format(face.detection_confidence, '.3f')) + '%',
                  fill='#FF0000')
    im.save('out.jpg')


detect_faces(path='/home/user/Documents/Images/face.jpg')
