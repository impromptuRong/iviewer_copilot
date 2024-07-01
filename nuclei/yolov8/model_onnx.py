import cv2
import time
import torch

import onnxruntime
import numpy as np
import torch.nn.functional as F

from .utils import masks2segments, process_mask, approx_poly_epsilon, non_max_suppression
from .utils import export_detections_to_table, map_coords

proto_scale_factor = 2.

class Yolov8SegmentationONNX:
    _dtype_map = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}

    def __init__(self, configs, device='cpu'):
        self.configs = configs
        self.load_model(configs.model_path, device=device)
        self.nms_params = configs.nms_params

    def load_model(self, model, device='cpu'):
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        self.session = onnxruntime.InferenceSession(
            model, providers=providers,
        )
        self.device = device

        # Get model info
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_size = model_input.shape[-2], model_input.shape[-1]
        self.model_dtype = self._dtype_map[model_input.type]

        model_outputs = self.session.get_outputs()
        self.output_names = [x.name for x in model_outputs]   

    def get_info(self):
        return {
            'input_name': self.input_name, 
            'output_names': self.outptu_names, 
            'input_size': self.input_size, 
            'model_dtype': self.model_dtype,
       }

    def __call__(self, images, patch_infos=None, nms_params={}, preprocess=True):
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
            if img.dtype == np.uint8:
                img = img / 255
            inputs.append(img)

        return np.stack(inputs).transpose(0, 3, 1, 2), np.array(image_sizes)

    def predict(self, inputs):
        # start = time.perf_counter()
        input_tensor = inputs.astype(self.model_dtype)
        preds = self.session.run(self.output_names, {self.input_name: input_tensor})
        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")

        return preds

    def postprocess(self, preds, image_sizes=None, nms_params={}):
        if isinstance(preds, list):
            preds, protos = preds
            preds = torch.from_numpy(preds)
            protos = torch.from_numpy(protos)
        else:
            preds, protos = preds, [None] * len(preds)
            preds = torch.from_numpy(preds)

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
