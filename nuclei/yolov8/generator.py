import numpy as np
from .utils import get_dzi, pad_pil


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
            best_page = osr.get_resize_level(mpp/osr.mpp, epsilon=0.2)
            assert best_page == 0, f"Currently we shouldn't get best_page={best_page}. Observed slide.mpp={osr.mpp}. "

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
        self.scale = osr.level_downsamples[self.page] / tile_patch_ratio
        # selr.scale = self.run_mpp / osr.mpp

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
