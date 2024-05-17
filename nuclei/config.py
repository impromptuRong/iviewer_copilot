model_name = "yolov8-lung"
model_path = "./ckpts/yolov8-lung-nuclei/best.onnx"
device = "cpu"
batch_size = 4
default_input_size = 640


labels = ['bg', 'tumor_nuclei', 'stromal_nuclei', 'immune_nuclei', 'blood_cell', 'macrophage', 'dead_nuclei', 'other_nuclei',]
labels_color = {
    -100: "#949494", 
    0: "#ffffff", 
    1: "#00ff00", 
    2: "#ff0000", 
    3: "#0000ff", 
    4: "#ff00ff", 
    5: "#ffff00",
    6: "#0094e1",
    7: "#646464",
}

labels_text = {
    0: 'bg', 1: 'tumor_nuclei', 2: 'stromal_nuclei', 3: 'immune_nuclei', 
    4: 'blood_cell', 5: 'macrophage', 6: 'dead_nuclei', 7: 'other_nuclei',
}

nms_params = {
    "conf_thres": 0.15, #  0.25,  # score_threshold, discards boxes with score < score_threshold
    "iou_thres": 0.45, #  0.7,  # iou_threshold, discards all overlapping boxes with IoU > iou_threshold
    "classes": None, 
    "agnostic": True, # False
    "multi_label": False, 
    "labels": (), 
    "nc": 7,
    "max_det": 300,  # maximum detection
}

wsi_params = {
    "default_mpp": 0.25,
    "wsi_patch_size": 512,
    "wsi_padding": 64,
    "wsi_page": 0,
    "mask_alpha": 0.3,
}

tiff_params = {
    "tile": (1, 256, 256), 
    "photometric": "RGB",
    "compression": "zlib", # compression=('jpeg', 95),  # None RGBA, requires imagecodecs
    "compressionargs": {"level": 8},
}

roi_names = {
    "tissue": True,  # use tissue region as roi
    "xml": '.*',  # use all annotations in slide_id.xml 
}
