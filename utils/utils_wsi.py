import os
import cv2
import datetime
import tifffile
import numpy as np

from .utils_image import get_dzi, img_as, pad, pad_pil


def load_cfg(cfg):
    if isinstance(cfg, dict):
        yaml = cfg
    else:
        import yaml
        with open(cfg, encoding='ascii', errors='ignore') as f:
            yaml = yaml.safe_load(f)  # model dict

    return yaml


def collate_fn(batch):
    return tuple(zip(*batch))


def is_image_file(x):
    ext = os.path.splitext(x)[1].lower()
    return not x.startswith('.') and ext in ['.png', '.jpeg', '.jpg', '.tif', '.tiff']


def folder_iterator(folder, keep_fn=None):
    file_idx = -1
    for root, dirs, files in os.walk(folder):
        for file in files:
            if keep_fn is not None and not keep_fn(file):
                continue
            file_idx += 1
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, folder)

            yield file_idx, rel_path, file_path


def get_slide_and_ann_file(svs_file, ann_file=None):
    folder_name, file_name = os.path.split(svs_file)
    slide_id, ext = os.path.splitext(file_name)
    if ann_file is True:
        ann_file = os.path.join(folder_name, slide_id + '.xml')
    if not (isinstance(ann_file, str) and os.path.exists(ann_file)):
        ann_file = None

    return svs_file, ann_file, slide_id


def processor(patch, info, **kwargs):
    scale = kwargs.get('scale', 1.0)
    if scale != 1.0:
        h_new, w_new = int(round(patch.shape[0] * scale)), int(round(patch.shape[1] * scale))
        patch = cv2.resize(patch, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        # patch = skimage.transform.rescale(patch, scale, order=3, multichannel=True)
        info = {
            'roi_slide': info['roi_slide'] * scale, 
            'roi_patch': info['roi_patch'] * scale,
        }

    return patch, info


def generate_roi_masks(slide, masks='tissue'):
    if isinstance(masks, str):
        if masks == 'tissue':
            res = slide.roughly_extract_tissue_region((1024, 1024), bg=255)
        elif masks == 'all':
            res = None
        elif masks == 'xml':
            res = slide.get_annotations(pattern='.*')
        else:
            res = slide.get_annotations(pattern=masks)
    elif callable(masks):
        res = masks(slide.thumbnail((1024, 1024)))
    else:
        res = masks
    
    return res


class Yolov8Generator:
    def __init__(self, osr, tile_size=512, overlap=64, mpp=0.25, masks=None):
        # Rename parameters
        patch_size, padding = tile_size, overlap
        self._osr = osr
        self.slide_id = osr.slide_id
        self.masks = masks
        
        assert padding % (patch_size/64) == 0, f"Padding {padding} should be divisible by {patch_size/64}."
        self.patch_size = patch_size
        self.padding = padding
        self.model_input_size = self.patch_size + self.padding * 2

        # Deep Zoom level
        self.dz_dimensions = osr.deepzoom_dims()
        self._dz_levels = len(self.dz_dimensions)  # Deep Zoom level count

        # Find the best page for inference
        if osr.mpp is not None:
            # We prefer to use a high resolution layer. We strict at most 1.2 rescale.
            best_page = osr.get_resize_level(osr.mpp/mpp, downsample_only=True, epsilon=0.2)
            assert best_page == 0, f"Currently we shouldn't get best_page={best_page}."

            # We force best_page is downsampled at 2**power
            power = np.log2(osr.level_downsamples[best_page])
            while best_page > 0 and abs(power-round(power)) > 1e-2:
                best_page += 1
                power = np.log2(osr.level_downsamples[best_page])

            self.page = best_page
            self.inference_dz_level = int(self._dz_levels - round(power) - 1)
            page_mpp = osr.mpp * osr.level_downsamples[self.page]
        else:
            self.page = 0
            self.inference_dz_level = 0
            page_mpp = 0.25
        # self.page_size = [osr.level_dims[self.page][1], osr.level_dims[self.page][0]]
        # self.page_size = osr.level_dims[self.page]

        # 1. Crop self.tile_size (self.overlap) from self.page (page_mpp)
        # 2. Resize to self.patch_size (self.padding), with self.run_mpp
        # 3. Result *= self.scale: rescale to page0
        self.tile_size = int(round(self.patch_size * mpp/page_mpp / 64) * 64)
        tile_patch_ratio = self.patch_size / self.tile_size  # 2 = 512/256
        self.overlap = int(self.padding / tile_patch_ratio)  # 32 = 64/2
        self.run_mpp = page_mpp / tile_patch_ratio  # 0.25 = 0.5 / 2
        self.scale = self.run_mpp / osr.mpp

        # scan whole slide and generate tile tables
        tiles, _, poly_indices, (row_id, col_id) = osr.whole_slide_scanner(
            self.tile_size, self.page, masks=self.masks, coverage_threshold=1e-8,
        )
        tiles.padding = (self.overlap, self.overlap)
        pars = (tiles.coords(), tiles.rois(), tiles.pad_width(), poly_indices, row_id, col_id)

        self.images, self.indices = [], {}
        for idx, par in enumerate(zip(*pars), 1):
            coord, roi_slide, pad_width, poly_id, row_id, col_id = par
            image_info = {
                'image_id': f'{self.slide_id}_{idx:05d}', 
                'tile_key': f'{self.page}_{col_id}_{row_id}',
                'data': {
                    'coord': coord.tolist(), 
                    'roi_slide': np.append(roi_slide, tiles.padding).tolist(),
                    'pad_width': pad_width.tolist(), 
                    'poly_id': poly_id,
                },
                'kwargs': {},
            }
            self.indices[image_info['tile_key']] = len(self.images)
            self.images.append(image_info)

    def __len__(self):
        return len(self.images)

    def load_patch_old(self, idx):
        info = self.images[idx]['data']
        pad_l, pad_r, pad_u, pad_d = info['pad_width']

        patch = self._osr.get_patch(info['coord'], self.page)
        patch = img_as('float32')(patch.convert('RGB'))
        patch = pad(patch, pad_width=[(pad_u, pad_d), (pad_l, pad_r)], 
                    mode='constant', cval=0.0)

        return patch

    def load_patch(self, idx):  # slightly faster
        info = self.images[idx]['data']
        pad_l, pad_r, pad_u, pad_d = info['pad_width']

        patch = self._osr.get_patch(info['coord'], self.page)
        # if patch.mode == 'RGBA' and format == 'jpeg':
            # patch = patch.convert('RGB')
        patch = pad_pil(patch.convert('RGB'), info['pad_width'], color=0)  # pad_l, pad_r, pad_u, pad_d
        # patch = T.ToTensor()(patch.convert('RGB'))
        # patch = T.Pad((pad_l, pad_u, pad_r, pad_d))(patch)

        return patch, info  # ['roi_slide'].astype(np.int32)

    def get_dzi(self, format='jpeg'):
        # we use level_dims[0] to make it consistent with original image
        return get_dzi(
            self._osr.level_dims[0],  # (w, h)
            tile_size=self.tile_size, 
            overlap=self.overlap, 
            format=format, 
        )

    def get_tile(self, level, address, format='jpeg'):
        if level != self.inference_dz_level:
            return None

        col, row = address
        tile_id = f"{self.page}_{col}_{row}"
        if tile_id in self.indices:
            return self.load_patch(self.indices[tile_id])
        else:
            return None


class ObjectIterator:
    def __init__(self, boxes, labels, scores=None, masks=None, keep_fn=None):
        self.boxes = boxes
        self.labels = labels
        self.scores = scores
        self.masks = masks
        self.keep_fn = keep_fn  # a function to filter out invalid entry

    def __iter__(self):
        for idx, (box, label) in enumerate(zip(self.boxes, self.labels)):
            obj = {'box': box, 'label': label}
            if self.scores:
                obj['score'] = self.scores[idx]
            if self.masks:
                obj['mask'] = self.masks[idx]
            if self.keep_fn is None or self.keep_fn(obj):
                yield obj


def wsi_imwrite(image, filename, slide_info, tiff_params, **kwargs):
    w0, h0 = image.shape[1], image.shape[0]
    tile_w, tile_h = tiff_params['tile'][-2:]   # (None/depth, w, h)
    mpp = slide_info['mpp']
    now = datetime.datetime.now()
    # software='Aperio Image Library v11.1.9'

    with tifffile.TiffWriter(filename, bigtiff=False) as tif:
        descp = f"HD-Yolo\n{w0}x{h0} ({tile_w}x{tile_h}) RGBA|{now.strftime('Date = %Y-%m-%d|Time = %H:%M:%S')}"
        for k, v in kwargs.items():
            descp += f'|{k} = {v}'
        # resolution=(mpp * 1e-4, mpp * 1e-4, 'CENTIMETER')
        tif.save(image, metadata=None, description=descp, subfiletype=0, **tiff_params,)

        for w, h in sorted(slide_info['level_dims'][1:], key=lambda x: x[0], reverse=True):
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            descp = f"{w0}x{h0} ({tile_w}x{tile_h}) -> {w}x{h} RGBA"
            # tile = (page.tilewidth, page.tilelength) if page.tilewidth > 0 and page.tilelength > 0 else None
            # resolution=(mpp * 1e-4 * w0/w, mpp * 1e-4 * h0/h, 'CENTIMETER')
            tif.save(image, metadata=None, description=descp, subfiletype=1, **tiff_params,)
