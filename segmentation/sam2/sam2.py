import time
import numpy as np
from ultralytics import SAM


class SAM2Segmentation:
    def __init__(self, configs, device='cpu'):
        self.configs = configs
        self.load_model(configs.model_path, device=device)

    def load_model(self, model, device='cpu'):
        self.model = SAM(model)  # sam2_b.pt
        self.model.to(device)
        self.model.info()

    def get_info(self):
        return self.configs

    def __call__(self, image, prompts=None, preprocess=True):
        s0 = time.time()
        if preprocess:
            image, _ = self.preprocess(image)

        s1 = time.time()
        if 'bboxes' in prompts:
            r = self.model(image, bboxes=prompts['bboxes'], verbose=False, save=False)[0]  # Segment with bounding box prompt
        else:
            r = self.model(image, points=prompts['points'], labels=prompts['labels'], verbose=False, save=False)[0]  # Segment with point prompt
        masks = r.masks.xy
        s2 = time.time()

        results = self.postprocess(masks)
        s3 = time.time()
        print(f"preprocess: {s1-s0}, inference: {s2-s1}, postprocess: {s3-s2}")

        return results

    def preprocess(self, image):
        img = np.array(image)
        h_ori, w_ori = img.shape[0], img.shape[1]
        if img.dtype == np.uint8:
            img = img / 255
        return img, np.array([h_ori, w_ori])

    def postprocess(self, results):
        boxes, masks = [], []
        for poly in results:
            (x0, y0), (x1, y1) = poly.min(0), poly.max(0)
            boxes.append([x0, y0, x1, y1])
            masks.append(poly)

        return {'boxes': np.array(boxes), 'masks': masks}

    def map_coords_to_records(self, output, patch_info=None, annotator=None, extra={}):
        x, y, _, _, scale_w, scale_h = patch_info or [0., 0., 0., 0., 1., 1.]
        output['boxes'] = (output['boxes'] + [x, y, x, y]) * [scale_w, scale_h, scale_w, scale_h]
        output['masks'] = [(m + [x, y]) * [scale_w, scale_h] for m in output['masks']]

        for (x0, y0, x1, y1), poly in zip(output['boxes'], output['masks']):
            xc, yc = (x0 + x1)/2, (y0 + y1)/2
            poly_x = ','.join([f'{_:.2f}' for _ in poly[:, 0]])
            poly_y = ','.join([f'{_:.2f}' for _ in poly[:, 1]])

            yield {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'xc': xc, 'yc': yc, 
                   'poly_x': poly_x, 'poly_y': poly_y, 'annotator': annotator, **extra}

    def test_run(self, image="http://images.cocodataset.org/val2017/000000039769.jpg"):
        import requests
        from PIL import Image

        print(f"Testing service for {image}")
        st = time.time()
        if image.startswith('http'):
            image = requests.get(image, stream=True).raw
        image = Image.open(image)
        box_prompts = np.array([
            [10, 50, 300, 450],
            [30, 80, 180, 110],
        ])
        r = self.__call__(image, prompts={'bboxes': box_prompts}, preprocess=False)

        print(f"Test service: segment {len(r['masks'])} objects ({time.time()-st}s)")
