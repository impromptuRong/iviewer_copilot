import cv2
import torch
import pandas as pd

from PIL import Image
from io import BytesIO


def approx_poly_epsilon(contour, factor=0.01):
    """ Get epsilon for simplify polygon.
        Smaller epsilon: tighter approximation to the original contour. It tends to preserve more details of the contour, resulting in a polygon with more vertices that closely follows the original shape.
        Larger epsilon: Conversely, a larger value of epsilon leads to a looser approximation. It simplifies the contour by reducing the number of vertices. This can result in a smoother, less detailed polygon that approximates the overall shape of the original contour.
    """
    # area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    # Define a factor to control the trade-off between detail and simplification

    epsilon = max(factor * perimeter, 0.5)  # * (1 + area / perimeter)
    # print({'area': area, 'perimeter': perimeter, 'epsilon': epsilon})
    return epsilon


def export_detections_to_table(res, labels_text=None, save_masks=True):
    df = {}
    df['x0'], df['y0'], df['x1'], df['y1'] = res['boxes'].round().to(torch.int32).T  
    if labels_text is not None:
        df['label'] = [labels_text[x] for x in res['labels'].tolist()]
    else:
        df['label'] = [f'cls_{x}' for x in res['labels']]
    if 'scores' in res:
        df['score'] = res['scores'].round(decimals=4)
    if save_masks and 'masks' in res:
        poly_x, poly_y = [], []
        for poly in res['masks']:
            poly_x.append(','.join([f'{_:.2f}' for _ in poly[:, 0]]))
            poly_y.append(','.join([f'{_:.2f}' for _ in poly[:, 1]]))

        df['poly_x'] = poly_x
        df['poly_y'] = poly_y

    return pd.DataFrame(df)


def pad_pil(img, pad_width, color=0):
    pad_l, pad_r, pad_u, pad_d = pad_width
    w, h = img.size

    res = Image.new(img.mode, (w + pad_l + pad_r, h + pad_u + pad_d), color)
    res.paste(img, (pad_l, pad_u))

    return res


def get_dzi(image_size, tile_size=254, overlap=1, format='jpeg'):
    """ Return a string containing the XML metadata for the .dzi file.
        image_size: (w, h)
        tile_size: tile size
        overlap: overlap size
        format: the format of the individual tiles ('png' or 'jpeg')
    """
    import xml.etree.ElementTree as ET
    image = ET.Element(
        'Image',
        TileSize=str(tile_size),
        Overlap=str(overlap),
        Format=format,
        xmlns='http://schemas.microsoft.com/deepzoom/2008',
    )
    w, h = image_size
    ET.SubElement(image, 'Size', Width=str(w), Height=str(h))
    tree = ET.ElementTree(element=image)
    buf = BytesIO()
    tree.write(buf, encoding='UTF-8')

    return buf.getvalue().decode('UTF-8')
