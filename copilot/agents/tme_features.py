import math
import torch
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from . import register_agent
from .utils.utils_features import _regionprops, generate_nuclei_map, product_features, apply_filters, density_plot
from .utils.utils_image import polygon_to_binary_mask_v2


DEFAULT_MPP = 0.25
SCALE_FACTOR = 32

## Basic nuclei summary function
description = f"""\
    Summarize the nuclei information from a given dataframe. \
    This tool calculates the statistical summary of nuclei for each different type. 
    Use this tool to answer user questions about percentage, count about nuclei.
"""
@register_agent(
    name='nuclei_composition',
    type='FunctionTool',
    input_mapping={'entries': 'annotations',},
    output_mapping='nuclei_composition_statistics',
    description=description,
)
def nuclei_composition_summary(entries):
    df = pd.DataFrame(entries)
    if not df.empty:
        res = df['label'].value_counts().to_dict()
    else:
        res = {}

    return res


## nuclei feature extractor
description = f"""\
    You are a agent that can calculate the nuclei morphological features information from a given dataframe. \
    The results includes the following aspects of each nuclei: area, size, eccentricity, orientation, perimeter, solidity, pa_ratio, etc.\
"""
@register_agent(
    name='nuclei_feature_extractor',
    type=None,
    input_mapping={'entries': 'annotations',},
    output_mapping='nuclei_morphological_features',
    description=description,
)
def nuclei_features(entries, box_only=False):
    if box_only:
        boxes = np.array([(e['x0'], e['y0'], e['x1'], e['y1']) for e in entries])
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        return pd.DataFrame({'box_area': h * w, 'labels': [e['label'] for e in entries]})
    else:
        # we need a way to vectorize this, pool.starmap is super slow
        df = []
        for e in entries:
            box = np.array([e['x0'], e['y0'], e['x1'], e['y1']]).round()
            mask = None
            if e['poly_x'] and e['poly_y']:
                poly_x = [float(x) for x in e['poly_x'].split(',')]
                poly_y = [float(x) for x in e['poly_y'].split(',')]
                if len(poly_x) == len(poly_y):
                    mask = [np.stack([poly_x, poly_y], -1) - np.array([box[0], box[1]])]
            df.append(_regionprops(box, e['label'], mask))

        return pd.DataFrame(df)  # .to_dict(orient='list')


## Nuclei morphological feature summary
description = f"""\
    Summarize the nuclei morphological information from a given dataframe. \
    This tool calculates the statistical summary of nuclei for each different type. 
    Use this tool to answer user questions about percentage, count about nuclei.
"""
@register_agent(
    name='nuclei_morphological_summary',
    type='FunctionTool',
    input_mapping={'nuclei_features': 'nuclei_morphological_features',},
    output_mapping='nuclei_morphological_feature_statistics',
    description=description,
)
def nuclei_morphological_summary(nuclei_features):
    df = nuclei_features
    filtered_df = df[df['labels'].str.endswith('nuclei', na=False)]
    nuclei_f = filtered_df.groupby('labels').agg(['mean', 'std'])
    nuclei_f.columns = [f'{k}.{tag}' for k, tag in nuclei_f.columns]
    nuclei_f = nuclei_f.stack()
    nuclei_f.index = [f'{k}.{v}' for k, v in nuclei_f.index]
    nuclei_f = {
        **{f'{k}.total': v for k, v in Counter(filtered_df['labels']).items()},
        **nuclei_f.to_dict(),
    }
    return nuclei_f


## Nuclei distribution scatter plot
description = f"""\
    Generate the nuclei scatter plot from a given dataframe.
"""
@register_agent(
    name='nuclei_scatter_plot',
    type='FunctionTool',
    input_mapping={
        'entries': 'annotations',
        'roi': 'roi',
        'mpp': 'mpp',
        'classes': 'core_nuclei_types',
    },
    output_mapping=['nuclei_scatter_plot', 'average_size_for_each_nuclei_type', 'mpp_adjusted_roi_size'],
    description=description,
)
def nuclei_scatter_plot(entries, roi, mpp=DEFAULT_MPP, classes=['tumor_nuclei', 'stromal_nuclei', 'immune_nuclei']):
    ## TODO: remove dependencies from torch
    mpp_scale = mpp / DEFAULT_MPP
    labels_mapping = {val: idx+1 for idx, val in enumerate(classes)}

    x0, y0, x1, y1 = roi
    roi_size = y1 - y0, x1 - x0
    mpp_adjusted_roi_size = int(math.ceil(roi_size[0] * mpp_scale)), int(math.ceil(roi_size[1] * mpp_scale))

    boxes = torch.tensor([(e['x0'], e['y0'], e['x1'], e['y1']) for e in entries if e['label'] in labels_mapping])
    labels = torch.tensor([labels_mapping[e['label']] for e in entries if e['label'] in labels_mapping])
    res_nuclei = {'boxes': (boxes - np.array([x0, y0, x0, y0])) * mpp_scale, 'labels': labels}

    nuclei_map, r_ave = generate_nuclei_map(
        res_nuclei, slide_size=mpp_adjusted_roi_size, 
        n_classes=len(labels_mapping), use_scores=False, 
    )

    return nuclei_map, r_ave, mpp_adjusted_roi_size


## tme density and connectivity feature
description = f"""\
    The agent calculate the tumor microenvironment (TME) nuclei density image features from a given dataframe. \
    The agent provide the following results: nuclei density plot, nuclei intensity, nuclei interaction strength, nuclei distributions. \
"""
@register_agent(
    name='tme_density_feature_extractor',
    type='FunctionTool',
    input_mapping={
        'nuclei_map': 'nuclei_scatter_plot',
        'radius': 'average_size_for_each_nuclei_type',
        'classes': 'core_nuclei_types',
    },
    output_mapping=['nuclei_map_rescale', 'nuclei_density_plot', 'nuclei_density_features', 'density_map'],
    description=description,
)
def tme_density_feature_extractor(nuclei_map, radius, scale_factor=1./SCALE_FACTOR, classes=['tumor_nuclei', 'stromal_nuclei', 'immune_nuclei']):
    assert len(classes) == max(radius.keys()) + 1, f"Got classes: {classes}, but radius: {radius}."
    class_ids = sorted(radius.keys())
    nuclei_map_rescale, cloud_d = apply_filters(nuclei_map, radius, scale_factor=scale_factor, 
                                                method='gaussian', grid=4096, device='cpu')  # int(1024/scale_factor)
    df = product_features(cloud_d)  # dot products
    density_map = density_plot(cloud_d, scale_factor=scale_factor)

    ## Density based features:
    sigmoid = lambda x: 1 / (1 + np.exp(-x))  # sigmoid function to normalize logit
    # norm: 
    df2 = {}
    for idx in class_ids:
        df2[f'{idx}.norm'] = df[f'{idx}_{idx}.dot'] ** 0.5
        df2[f'{idx}.norm.logit'] = np.log(df2[f'{idx}.norm'])

    # projection (direction): df['i_j.proj'] = df['i_j.dot'] / df['j.dot']
    for i, j in itertools.permutations(class_ids, 2):
        df2[f'{i}_{j}.proj'] = df[f'{min(i, j)}_{max(i, j)}.dot'] / df[f'{j}_{j}.dot']
        df2[f'{i}_{j}.proj.logit'] = np.log(df2[f'{i}_{j}.proj'])
        df2[f'{i}_{j}.proj.prob'] = sigmoid(df2[f'{i}_{j}.proj.logit'])

    # cosine similarity: df['i_j.cos'] = df['j_j.dot']/df['i.norm']/df['j.norm']
    for i, j in itertools.combinations(class_ids, 2):
        i, j = min(i, j), max(i, j)
        df2[f'{i}_{j}.cos'] = df[f'{i}_{j}.dot']/df2[f'{i}.norm']/df2[f'{j}.norm']

    return nuclei_map_rescale, cloud_d, {**df, **df2}, density_map


## tme feature extractor
description = f"""\
    The Agent calculate nuclei environment interaction based tumor microenvironment (TME) information. \
    The agent evaluate the following TME features with probabilities or degree: 
    nuclei pleomorphism, nuclear–cytoplasmic ratio/nc-tario, tumor inflitration, immune response. 
"""
@register_agent(
    name='tme_feature_extractor',
    type='FunctionTool',
    input_mapping={
        'entries': 'annotations',
        'nuclei_morphological_feature_statistics': 'nuclei_morphological_feature_statistics', 
        'nuclei_density_features': 'nuclei_density_features',
    },
    output_mapping={ 
        'tumor_intensity': 'nuclei_intensity', 
        'stromal_intensity': 'stromal_intensity', 
        'immune_intensity': 'immune_intensity',
        'nuclei_pleomorphism_degree': 'nuclei_pleomorphism_degree',
        'til_degree': 'tumor_infiltrating_lymphocytes_degree',
        'immune_response_degree': 'immune_response_degree',
    },
    description=description,
)
def tme_summary(entries, nuclei_morphological_feature_statistics, nuclei_density_features):
    # print(nuclei_morphological_feature_statistics.keys())
    # print(nuclei_density_features.keys())
    summary = {
        'nuclei_pleomorphism_degree': np.log(nuclei_morphological_feature_statistics['tumor_nuclei.box_area.std']),
        'tumor_intensity': nuclei_density_features['0.norm'], 
        'stromal_intensity': nuclei_density_features['1.norm'], 
        'immune_intensity': nuclei_density_features['2.norm'], 
        'til_degree': nuclei_density_features['2_0.proj.prob'], 
        'immune_response_degree': nuclei_density_features['0_2.proj.prob'],
    }

    return summary


## metastasis detection
description = f"""\
    The Agent detect the existence of tumor nuclei inside lymph nodes. stromal region. \
    The agent returns the degree of metastasis for each lymph node.  
"""
@register_agent(
    name='metastasis_detection',
    type='FunctionTool',
    input_mapping={
        'entries': 'annotations',
        'roi': 'roi',
        'nuclei_map': 'nuclei_map_rescale', 
    },
    output_mapping='lymph_nodes_metastasis',
    description=description,
)
def detect_metastasis(entries, roi, nuclei_map):
    tumor_nuclei_map = np.array(nuclei_map[0])
    x0, y0, x1, y1 = roi
    h, w = tumor_nuclei_map.shape[0], tumor_nuclei_map.shape[1]
    scale_factor = np.array([h / (y1 - y0), w / (x1 - x0)])

    df = pd.DataFrame(entries)
    lymph_nodes = df[df['label'] == 'lymph_node']
    lymph_nodes_metastasis = {}
    for idx, e in lymph_nodes.iterrows():
        box = np.array([e['x0'], e['y0'], e['x1'], e['y1']])
        criteria = (df['label'] == 'tumor_nuclei') & (df['x0'] > box[0]) & (df['y0'] > box[1]) & (df['x1'] < box[2]) & (df['y1'] < box[3])
        inner_tumor = df[criteria]
        inner_tumor_size = ((inner_tumor['x1'] - inner_tumor['x0']) * (inner_tumor['y1'] - inner_tumor['y0'])).mean()
        mask = None
        if e['poly_x'] and e['poly_y']:
            poly_x = [float(x) for x in e['poly_x'].split(',')]
            poly_y = [float(x) for x in e['poly_y'].split(',')]
            if len(poly_x) == len(poly_y):
                mask = [(np.stack([poly_x, poly_y], -1) - np.array([roi[0], roi[1]])) * scale_factor]
        # mask_w, mask_h = int(math.ceil(w)), int(math.ceil(h))
        if mask:
            mask_img = polygon_to_binary_mask_v2(mask, size=(h, w), mode='xy')
        else:
            obj_x0, obj_y0, obj_x1, obj_y1 = box.round()
            mask_img = np.zeros((h, w))
            mask_img[int(obj_y0):int(obj_y1), int(obj_x0):int(obj_x1)] = 1

        lymph_nodes_metastasis[f'lymph_node_{idx}'] = np.log(inner_tumor_size) * (tumor_nuclei_map * mask_img).sum()

    lymph_nodes_metastasis = {k: float(v.item()) for k, v in lymph_nodes_metastasis.items()}

    return lymph_nodes_metastasis
