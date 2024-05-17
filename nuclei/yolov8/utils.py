import cv2
import time
import numbers
import torch
import torchvision
import numpy as np
import pandas as pd
from io import BytesIO


def approx_poly_epsilon(contour, factor=0.01):
    """ Get epsilon for simplify polygon.
        Smaller epsilon: tighter approximation to the original contour. It tends to preserve more details of the contour, resulting in a polygon with more vertices that closely follows the original shape.
        Larger epsilon: Conversely, a larger value of epsilon leads to a looser approximation. It simplifies the contour by reducing the number of vertices. This can result in a smoother, less detailed polygon that approximates the overall shape of the original contour.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    # Define a factor to control the trade-off between detail and simplification
    
    epsilon = max(factor * perimeter, 0.5)  # * (1 + area / perimeter)
    # print({'area': area, 'perimeter': perimeter, 'epsilon': epsilon})
    return epsilon


def masks2segments(masks, output_shape=None, strategy='largest', approx=False):
    """ modify from ultralytics.utils.ops.masks2segments. """
    segments = []
    
    if output_shape is not None:
        h_scale = output_shape[0]/masks.shape[-2]
        w_scale = output_shape[1]/masks.shape[-1]
    else:
        h_scale, w_scale = 1.0, 1.0

    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        c = (c * np.array([w_scale, h_scale])).astype('float32')

        if approx and len(c):
            if isinstance(approx, numbers.Number):
                epsilon = approx
            else:
                epsilon = approx(c)
            c = cv2.approxPolyDP(c, epsilon, True)[:, 0, :]

        segments.append(c)

    return segments


def process_mask(protos, masks_in, boxes, shape):
    """
    Apply masks to bounding boxes using the output of the mask head.
    (ultralytics.utils.ops.process_mask.)

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        boxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    boxes = boxes.clone()
    boxes[:, [0, 2]] *= mw / iw
    boxes[:, [1, 3]] *= mh / ih

    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(mw, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(mh, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.
    (ultralytics.utils.ops.non_max_suppression)
    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def map_coords(r, patch_info):
    # trim border objects, map to original coords
    # patch_info: [x0_s, y0_s, w_p(w_s), h_p(h_s), pad_w(x0_p), pad_h(y0_p)]
    x0_s, y0_s, w_p, h_p, x0_p, y0_p = patch_info
    # assert x0_p == 64 and y0_p == 64 and w_s == w_p and h_s == h_p, f"{roi_slide}, {roi_patch}"
    x_c, y_c = r['boxes'][:,[0,2]].mean(1), r['boxes'][:,[1,3]].mean(1)
    keep = (x_c > x0_p) & (x_c < x0_p + w_p) & (y_c > y0_p) & (y_c < y0_p + h_p)

    res = {k: r[k][keep] for k in ['boxes', 'labels', 'scores']}
    res['boxes'][:, [0, 2]] += x0_s - x0_p
    res['boxes'][:, [1, 3]] += y0_s - y0_p
    if 'masks' in r:
        res['masks'] = [m + [x0_s - x0_p, y0_s - y0_p] 
                        for m, tag in zip(r['masks'], keep) if tag]

    return res


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
