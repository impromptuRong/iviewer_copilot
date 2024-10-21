import cv2
import time
import torch
import numbers
import torchvision
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as T

from .utils import masks2segments, process_mask, approx_poly_epsilon, non_max_suppression
from .utils import export_detections_to_table, map_coords

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
        self.input_size = (size[0], size[1])

        return self

    def to(self, device):
        self.model, self.device, self.model_dtype = self.load_model(self.model, device=device)

        return self

    def __call__(self, images, nms_params={}, preprocess=True):
        with torch.inference_mode():
            s0 = time.time()
            if preprocess:
                inputs, image_sizes = self.preprocess(images)
            else:
                # no process: images = np.float(batch, 3, 640, 640)
                # images = np.stack([np.array(_) for _ in pil_images]).transpose(0, 3, 1, 2) / 255
                assert images.shape[1:] == (3,) + self.input_size, f"Inputs require preprocess."
                inputs, image_sizes = images, None
            s1 = time.time()
            preds = self.predict(inputs)
            s2 = time.time()
            results = self.postprocess(
                preds, image_sizes, nms_params,
            )
            s3 = time.time()
            print(f"preprocess: {s1-s0}, inference: {s2-s1}, postprocess: {s3-s2}")

        return results

    def preprocess(self, images):
        h, w = self.input_size

        inputs, image_sizes = [], []
        for img in images:
            img = np.array(img)
            h_ori, w_ori = img.shape[0], img.shape[1]
            image_sizes.append([h_ori, w_ori])
            if h != h_ori or w != w_ori:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            img = T.ToTensor()(img)
            inputs.append(img)

        return torch.stack(inputs), image_sizes

    def predict(self, inputs):
        inputs = inputs.to(self.device, self.model_dtype, non_blocking=True)
        with torch.inference_mode():
            preds = self.model(inputs)

        return preds

    def postprocess(self, preds, image_sizes=None, nms_params={}):
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
        for pred, proto, (h_ori, w_ori) in zip(preds, protos, image_sizes):
            o = {'boxes': pred[:, :4], 'labels': pred[:, 5] + 1, 'scores': pred[:, 4],}

            if len(pred):
                if proto is not None:
                    masks = process_mask(proto, pred[:, 6:], pred[:, :4], [h, w])  # HWC
                    masks = F.interpolate(masks[None], scale_factor=proto_scale_factor, 
                                          mode='bilinear', align_corners=False)[0]  # CHW
                    masks = masks2segments(masks.gt_(0.5), output_shape=[h_ori, w_ori], 
                                           approx=lambda x: approx_poly_epsilon(x, factor=0.01)
                                          )
                    o['masks'] = masks

                # max number of float16 is 65536, half will lead to inf for large image.
                o['boxes'][:, [0, 2]] *= w_ori/w  # rescale to image size
                o['boxes'][:, [1, 3]] *= h_ori/h
                o['boxes'] = o['boxes'].to(torch.float32).cpu()
                o['labels'] = o['labels'].to(torch.int32).cpu()
                o['scores'] = o['scores'].to(torch.float32).cpu()

            res.append(o)

        return res

    def convert_results_to_annotations(self, output, patch_info, annotator=None, extra={}):
        output = map_coords(output, patch_info)

        ## save tables
        st = time.time()
        df = export_detections_to_table(
            output, labels_text=self.configs.labels_text,
            save_masks=True,
        )
        df['xc'] = (df['x0'] + df['x1']) / 2
        df['yc'] = (df['y0'] + df['y1']) / 2
        # df['box_area'] = (df['x1'] - df['x0']) * (df['y1'] - df['y0'])
        df['description'] = df['label'].astype(str) + '; \nscore=' + df['score'].astype(str)
        df = df.drop(columns=['score'])
        df['annotator'] = annotator or self.configs.model_name
        for k, v in extra.items():
            df[k] = v
        print(f"Export table: {time.time() - st}s.")

        return df.to_dict(orient='records')

    def test_run(self, image="http://images.cocodataset.org/val2017/000000039769.jpg"):
        import requests
        from PIL import Image

        print(f"Testing service for {image}")
        st = time.time()
        image = Image.open(requests.get(image, stream=True).raw)
        r = self.__call__([np.array(image)])

        print(f"Test service: {len(r)} ({time.time()-st}s)")
