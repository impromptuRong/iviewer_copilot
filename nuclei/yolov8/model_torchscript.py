import cv2
import time
import torch
import numbers
import torchvision
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as T

from .utils import rgba2rgb, masks2segments, process_mask, non_max_suppression

YOLOv8_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "yolov8-lung-nuclei",
]

proto_scale_factor = 2.



class Yolov8SegmentationTorchscript:
    def __init__(self, configs, device='cpu'):
        self.configs = configs
        self.set_input_size(configs.default_input_size)
        self.model, self.device, self.model_dtype = self.load_model(configs.model_path, device=device)
        self.nms_params = configs.nms_params

    def load_model(self, model, device='cpu'):
        if isinstance(model, str):
            model = torch.jit.load(model)

        if isinstance(device, str):
            device = torch.device(device)
        model.eval()
        model.to(device)

        if device.type == 'cpu':  # half precision only supported on CUDA
            model.float()
        else:
            model.half()

        model_dtype = next(model.model.parameters()).dtype

        # warmup for 2 epochs, jit is super slow on 2nd run...
        inputs = [torch.rand([4, 3, self.input_size[0], self.input_size[1]])] * 2
        for x in inputs:
            x = x.to(device, model_dtype)
            dummy = model(x)

        return model, device, model_dtype    

    def set_input_size(self, size):
        if isinstance(size, numbers.Number):
            size = [int(size), int(size)]
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
        self.input_size = [size[0], size[1]]

        return self

    def to(self, device):
        self.model, self.device, self.model_dtype = self.load_model(self.model, device=device)

        return self

    def __call__(self, images, patch_infos=None, nms_params={}):
        with torch.inference_mode():
            s0 = time.time()
            inputs, image_sizes = self.preprocess(images)
            s1 = time.time()
            preds = self.predict(inputs)
            s2 = time.time()
            results = self.postprocess(
                preds, patch_infos, image_sizes, nms_params,
            )
            s3 = time.time()
            print(f"preprocess: {s1-s0}, inference: {s2-s1}, postprocess: {s3-s2}")

        return results

    def preprocess(self, images):
        h, w = self.input_size

        inputs, image_sizes = [], []
        for img in images:
            img = rgba2rgb(img)
            h_ori, w_ori = img.shape[0], img.shape[1]
            image_sizes.append([h_ori, w_ori])
            if h != h_ori or w != w_ori:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = T.ToTensor()(img)
            inputs.append(img)

        return torch.stack(inputs), torch.tensor(image_sizes)

    def predict(self, inputs):
        inputs = inputs.to(self.device, self.model_dtype, non_blocking=True)
        with torch.inference_mode():
            preds = self.model(inputs)

        return preds

    def postprocess(self, preds, patch_infos=None, image_sizes=None, nms_params={}):
        if isinstance(preds, tuple):
            preds, protos = preds
        else:
            preds, protos = preds, [None] * len(preds)

        if patch_infos is None:
            patch_infos = [None] * len(preds)
        if image_sizes is None:
            image_sizes = [self.input_size] * len(preds)

        nms_params = {**self.nms_params, **nms_params}
        preds = non_max_suppression(preds, **nms_params)

        h, w = self.input_size

        res = []
        for pred, proto, info, (h_ori, w_ori) in zip(preds, protos, patch_infos, image_sizes):
            o = {'boxes': pred[:, :4], 'labels': pred[:, 5] + 1, 'scores': pred[:, 6],}

            if len(pred):
                if proto is not None:
                    masks = process_mask(proto, pred[:, 6:], pred[:, :4], [h, w])  # HWC
                    masks = F.interpolate(masks[None], scale_factor=proto_scale_factor, 
                                          mode='bilinear', align_corners=False)[0]  # CHW
                    o['masks'] = masks2segments(masks.gt_(0.5), output_shape=[h_ori, w_ori])

                # max number of float16 is 65536, half will lead to inf for large image.
                o['boxes'][:, [0, 2]] *= w_ori/w  # rescale to image size
                o['boxes'][:, [1, 3]] *= h_ori/h
                o['boxes'] = o['boxes'].to(torch.float32).cpu()
                o['labels'] = o['labels'].to(torch.int32).cpu()
                o['scores'] = o['scores'].to(torch.float32).cpu()

                # trim border objects, map to original coords
                if info is not None:
                    # info: [x0_s, y0_s, w_p(w_s), h_p(h_s), pad_w(x0_p), pad_h(y0_p)]
                    x0_s, y0_s, w_p, h_p, x0_p, y0_p = info
                    # assert x0_p == 64 and y0_p == 64 and w_s == w_p and h_s == h_p, f"{roi_slide}, {roi_patch}"
                    x_c, y_c = o['boxes'][:,[0,2]].mean(1), o['boxes'][:,[1,3]].mean(1)
                    keep = (x_c > x0_p) & (x_c < x0_p + w_p) & (y_c > y0_p) & (y_c < y0_p + h_p)
                    o = {k: v[keep] for k, v in o.items()}

                    o['boxes'][:, [0, 2]] += x0_s - x0_p
                    o['boxes'][:, [1, 3]] += y0_s - y0_p
                    for m in o.get('masks', []):
                        m += [x0_s - x0_p, y0_s - y0_p]

            res.append(o)

        return res
