import os
import numpy as np
import random
import torch
import time
from ome_types import to_dict
import pandas as pd


def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def roi_to_dict(roi, downsample_factor=1):
    """
    Converts a single ROI into a dictionary with ROI ID as the key and the downsampled
    rectangle data (x, y, width, height) as the value.

    Args:
        roi: A single ROI object.
        downsample_factor: Factor by which to downsample the coordinates and dimensions.

    Returns:
        A dictionary with the ROI ID as the key and rectangle data as the value.
    """
    import numpy as np

    roi_data = {}
    roi_name = ""
    roi_id = roi.id  # Extract the ROI ID
    #print("ROI: ",roi_id)
    union_shapes = to_dict(roi.union)  # Convert the union shapes to a dictionary
    if 'labels' in union_shapes:
        for lbl in union_shapes['labels']:
            #print(lbl['text'])
            roi_name = lbl['text']
        
    if 'rectangles' in union_shapes:
        
        for rect in union_shapes['rectangles']:
            # Downsample rectangle dimensions
            roi_data[roi_id] = {
                'x': int(rect['x'] / downsample_factor),
                'y': int(rect['y'] / downsample_factor),
                'width': int(rect['width'] / downsample_factor),
                'height': int(rect['height'] / downsample_factor),
                'name': roi_name,
                'shape': 'rectangle',
            }

    elif 'polygons' in union_shapes:
        for polygon in union_shapes['polygons']:
            # Extract polygon points and calculate bounding box
            points = polygon['points'].split()  # Points as a space-separated string
            vertices = [tuple(map(float, point.split(','))) for point in points]  # Convert to (x, y) tuples

            # Calculate bounding box
            x_coords, y_coords = zip(*vertices)  # Separate x and y coordinates
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width = x_max - x_min
            height = y_max - y_min

            # Downsample bounding box dimensions
            roi_data[roi_id] = {
                'x': int(x_min / downsample_factor),
                'y': int(y_min / downsample_factor),
                'width': int(width / downsample_factor),
                'height': int(height / downsample_factor),
                'name': roi_name,
                'shape': 'polygon',
            }

    elif 'ellipses' in union_shapes:
        for ellipse in union_shapes['ellipses']:
            # Downsample ellipse dimensions
            roi_data[roi_id] = {
                'x': int(ellipse['x'] / downsample_factor),
                'y': int(ellipse['y'] / downsample_factor),
                'width': int(ellipse['radius_x'] / downsample_factor),
                'height': int(ellipse['radius_y'] / downsample_factor),
                'name': roi_name,
                'shape': 'ellipses',
            }

    return roi_data

def rois_to_dict(rois, downsample_factor=1):
    """
    Converts a list of ROIs into a single dictionary with all ROI IDs and their corresponding
    downsampled rectangle data (x, y, width, height).

    Args:
        rois: List of ROI objects (e.g., ome_metadata.rois).
        downsample_factor: Factor by which to downsample the coordinates and dimensions.

    Returns:
        A dictionary where each key is an ROI ID, and the value is a dictionary of rectangle data.
    """
    all_roi_data = {}
    for roi in rois:
        roi_data = roi_to_dict(roi, downsample_factor)
        all_roi_data.update(roi_data)
    return all_roi_data


def save_features_with_names(features, image_paths, output_path, model_name, format="csv"):
    """
    Save extracted features along with image names.

    Parameters:
        features (numpy.ndarray): The extracted feature array.
        image_paths (list): List of image file paths.
        output_path (str): Directory to save features.
        model_name (str): Name of the foundation model.
        format (str): File format ('csv' recommended for indexing).
    """
    os.makedirs(output_path, exist_ok=True)
    curr_datetime = time.strftime("%Y%m%d-%H%M%S")
    feature_file = os.path.join(output_path, f"{model_name}_features-{curr_datetime}.{format}")

    # Extract just the file names (without paths)
    image_names = [os.path.basename(path) for path in image_paths]

    # Combine the features and image names into a DataFrame
    df = pd.DataFrame(features)
    df.insert(0, "image_name", image_names)

    if format == "csv":
        df.to_csv(feature_file, index=False)
    elif format == "pkl":
        df.to_pickle(feature_file)
    elif format == "npy":
        np.save(feature_file, df.to_numpy())
    else:
        raise ValueError("Unsupported format. Choose 'csv', 'pkl', or 'npy'.")

    print(f"Features saved to: {feature_file}")



def load_features_from_csv(feature_file):
    """
    Load features from a CSV file.

    Parameters:
        feature_file (str): Path to the CSV file containing features.

    Returns:
        tuple: (image_names, feature_matrix)
    """
    df = pd.read_csv(feature_file)

    # Extract image names and feature values separately
    image_names = df["image_name"].tolist()
    feature_matrix = df.drop(columns=["image_name"]).values  # Convert features to NumPy array

    return image_names, feature_matrix


#### PATIENT AGGREGATION

def majority_vote_binary(s):
    counts = s.value_counts()
    if len(counts) == 1:
        return counts.index[0]          # unanimous
    if counts.iloc[0] == counts.iloc[1]:
        return np.nan                   # tie → uncertain
    return counts.idxmax()

def hard_majority(s):
    frac = s.mean()   # since s is 0/1
    if frac > 0.5:
        return 1
    if frac < 0.5:
        return 0
    return np.nan    
    


def patient_level_auc(y_true_patch, y_proba_patch, groups, pos_label=1, agg="mean"):
    """
    Compute patient-level AUC by aggregating patch-level probabilities per patient (group).

    y_true_patch: (n_patches,) array-like of 0/1
    y_proba_patch: (n_patches,) array-like of P(y=1)
    groups: (n_patches,) array-like patient IDs (e.g., LEAP_ID per patch)
    agg: 'mean' or 'median'
    """
    from sklearn.metrics import roc_auc_score
    
    df = pd.DataFrame(
        {"group": groups, "y": y_true_patch, "proba": y_proba_patch}
    )

    if agg == "mean":
        g = df.groupby("group").agg(y=("y", "max"), proba=("proba", "mean"))
    elif agg == "median":
        g = df.groupby("group").agg(y=("y", "max"), proba=("proba", "median"))
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    # Must have both classes at patient level to compute AUC
    if g["y"].nunique() < 2:
        return np.nan, g

    auc = roc_auc_score(g["y"].values, g["proba"].values)
    return auc, g

##### METRICS

def compare_patient_preds(results, patient_actual, suffix='',thresh=0.5 ):
    """
    Compare predicted patient-level response classifications with the actual labels.

    This function aggregates ROI-level predicted probabilities for each patient
    and reports the predicted class based on a threshold. It prints a summary
    showing how many ROIs were classified as responders and the overall
    patient-level prediction (majority vote via the mode).

    Prints a comparison summary for each patient.
    """
    leapids = sorted(list(results['LEAP_ID'].unique()))        
    for leap in leapids:
        meta_tmp =results.loc[results['LEAP_ID'] ==leap]
        preds = (meta_tmp[f'probs{suffix}']> thresh).astype(int)
        print(f"{leap}: {preds.sum()} / {preds.count()} \t... predicted: {'Non-Responder' if preds.mode()[0]==0 else 'Responder'}\t | actually: {patient_actual[leap]}")




def patient_level_predictions(
    results,
    patient_actual,
    suffix='',
    thresh=0.5,
    agg="majority"
):
    """
    Aggregate patch/ROI-level predictions into patient-level predictions.

    This function groups prediction results by patient (LEAP_ID) and converts
    patch-level probabilities into a single patient-level prediction using
    a specified aggregation strategy.

    Args:
        results (pd.DataFrame):
            DataFrame containing patch/ROI-level predictions. Must include:
            - 'LEAP_ID' : patient identifier
            - 'probs{suffix}' : predicted probability of response.

        patient_actual (pd.Series or dict-like):
            Mapping from LEAP_ID to the true patient-level response label
            (e.g. "Responder" or "Non-Responder").

        suffix (str, optional):
            Optional suffix used to select the probability column
            (e.g. 'probs_calibrated'). Default '' expects column 'probs'.

        thresh (float, optional):
            Probability threshold used to convert probabilities into binary
            predictions. Default is 0.5.

        agg (str, optional):
            Method used to aggregate patch-level predictions into a patient-level
            prediction. Options:
                - "majority": majority vote of binary patch predictions
                - "mean_prob": mean probability across patches compared to threshold

    Returns:
        pd.DataFrame:
            Patient-level summary table with the following columns:
                - LEAP_ID : patient identifier
                - n_patches : number of patches/ROIs for the patient
                - n_pos_patches : number of patches predicted positive
                - frac_pos_patches : fraction of positive patch predictions
                - mean_prob : mean predicted probability across patches
                - predicted : predicted patient label (0 = Non-Responder, 1 = Responder)
                - actual : true label encoded as numeric
                - actual_label : original string label
    """
    LABEL_MAP = {"Non-Responder": 0, "Responder": 1}
    rows = []
    leapids = sorted(list(results['LEAP_ID'].unique()))

    for leap in leapids:
        meta_tmp = results.loc[results['LEAP_ID'] == leap]
        probs = meta_tmp[f'probs{suffix}'].values
        preds = (probs > thresh).astype(int)

        if agg == "majority":
            patient_pred = int(preds.mean() >= 0.5)
        elif agg == "mean_prob":
            patient_pred = int(probs.mean() >= thresh)
        else:
            raise ValueError("Unknown aggregation method")

        actual_label = patient_actual.loc[leap]
        actual_numeric = LABEL_MAP[actual_label]

        rows.append({
            "LEAP_ID": leap,
            "n_patches": len(preds),
            "n_pos_patches": preds.sum(),
            "frac_pos_patches": preds.mean(),
            "mean_prob": probs.mean(),
            "predicted": patient_pred,
            "actual": actual_numeric,
            "actual_label": actual_label
        })

    return pd.DataFrame(rows)


def patient_level_metrics(
    df,
    model_name=None,
    thresh=None,
    agg=None
):
    
    from sklearn.metrics import accuracy_score, roc_auc_score,    precision_score, recall_score, f1_score, confusion_matrix
    y_true = df["actual"].values
    y_pred = df["predicted"].values
    y_score = df["mean_prob"].values  # for AUC


    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "n_patients": len(df),
        "threshold": thresh,
        "aggregation": agg,
        "confusion_matrix": cm,
        "tn": cm[0,0],
        "fp": cm[0,1],
        "fn": cm[1,0],
        "tp": cm[1,1]
    }

    return metrics



def print_duration(from_time,to_time,label="", in_mins=True):
    dur = to_time - from_time
    if in_mins:
        print(f"{label}:\t {(dur/60):.1f} mins")
    else:
        print(f"{label}:\t {dur:.1f} secs")

def print_time_since(from_time,label="",in_mins=False):
    import time
    to_time = time.perf_counter()
    print_duration(from_time,to_time,label,in_mins)
    #dur = to_time - from_time

    # if in_mins:
        
    #     print(f"{label}:\t {(dur/60):.1f} mins")
    # else:
    #     print(f"{label}:\t {dur:.1f} secs")
    return to_time


def extract_ids(filename):
    import re

    # This regex finds "LEAP" followed by digits, then later "Slide" (with an optional hyphen, underscore, or space) and digits.
    pattern = r'(Slide \d+).*?.ndpi'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        sname = match.group(0)
        #print(sname)
    else:
        sname = None
        #print("no match")

    pattern = r'Slide \d+'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        snum = match.group(0)
        #print(snum)
    else:
        snum = None
        #print("no slide num")

    pattern = r'LEAP\d+'
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        leap = match.group(0)
        #print(leap)
    else:
        leap = None
        #print("no LEAPID")
    
    return sname, snum, leap

    