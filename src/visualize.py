import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_rois_on_image(image, roi_dict, save=False, LEAPID=""):
    """
    Overlays ROI rectangles on an image and labels them with their IDs.

    Args:
        image: The background image (e.g., a 2D NumPy array or an image array).
        roi_dict: Dictionary where keys are ROI IDs and values are dictionaries
                  containing 'x', 'y', 'width', and 'height' for the rectangle.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(image, cmap='gray', interpolation='nearest')
    
    # Loop through each ROI in the dictionary

    for roi_id, rect in roi_dict.items():
        # Extract rectangle data
        x, y, width, height = rect['x'], rect['y'], rect['width'], rect['height']
        print(f"roi {roi_id} : {x}, {y}, {width}, {height}")
        # Create a rectangle patch
        rectangle = patches.Rectangle(
            (x, y),  # Bottom-left corner
            width,   # Width
            height,  # Height
            linewidth=2,
            edgecolor='blue',
            facecolor='none'
        )
        # Add the rectangle to the plot
        ax.add_patch(rectangle)

        # Add the ROI ID as a label
        ax.text(
            x + width / 2,  # Center of the rectangle (x)
            y + height / 2,  # Center of the rectangle (y)
            #roi_id,          # ROI ID label
            rect['name'],          # ROI ID label
            color='red',
            fontsize=10,
            ha='center',     # Horizontal alignment
            va='center'      # Vertical alignment
        )

    # Customize the plot
    ax.set_title('ROIs Overlaid on Image')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')  # Keep aspect ratio equal
    #plt.gca().invert_yaxis()  # Invert y-axis to align with image coordinate system
    if save:
        plt.savefig(f"ROIs_over_{LEAPID}.png")
    plt.show()
    plt.close()


def display_spatial_expression(gene, he_img, subset_metadata, expr_matrix , xlim=None, ylim=None, ax=None):
    """
    Visualise spatial gene expression overlaid on a histology image.

    This function plots ROI locations on top of an H&E image and colours them
    according to the expression level of a specified gene.

    Args:
        gene (str):
            Gene name to visualise.

        he_img (np.ndarray):
            Background H&E image (typically a downsampled RGB image).

        subset_metadata (pd.DataFrame):
            DataFrame containing ROI spatial coordinates. Must include:
            - 'ROICoordinateX_downsampled'
            - 'ROICoordinateY_downsampled'

        expr_matrix (pd.DataFrame):
            Gene expression matrix indexed by gene names with columns
            corresponding to ROIs.

        xlim (tuple, optional):
            Tuple specifying x-axis limits (xmin, xmax) for zooming.

        ylim (tuple, optional):
            Tuple specifying y-axis limits (ymin, ymax) for zooming.

        ax (matplotlib.axes.Axes, optional):
            Existing matplotlib axis to plot on. If None, a new figure
            and axis will be created.

    Returns:
        matplotlib.axes.Axes:
            Axis containing the spatial expression plot.
    """
    plt.imshow(he_img)
    plt.scatter(
        subset_metadata["ROICoordinateY_downsampled"], 
        subset_metadata["ROICoordinateX_downsampled"], 
        c=expr_matrix.loc[gene], cmap="viridis", alpha=0.8
    )
    plt.colorbar(label="Expression Level")
    plt.title(f"Spatial Expression of {gene}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.show()



def plot_classification_results(y_true, y_probs, threshold=0.5, vertical=False, labels=[0,1], 
                            title="Classification Results", save_path=None):
    """
    Plot classification performance using a confusion matrix and ROC curve.

    This function converts predicted probabilities into binary predictions using
    a specified threshold, then visualises model performance with a confusion
    matrix and ROC curve. The plots can be arranged either horizontally or
    vertically.

    Args:
        y_true (array-like):
            Ground truth binary labels.

        y_probs (array-like):
            Predicted probabilities for the positive class.

        threshold (float, optional):
            Probability threshold used to convert probabilities into binary
            predictions. Default is 0.5.

        vertical (bool, optional):
            If True, plots are stacked vertically; otherwise displayed side-by-side.

        labels (list, optional):
            Label order for the confusion matrix (default [0, 1]).

        title (str, optional):
            Title for the overall figure.

        save_path (str, optional):
            If provided, saves the figure to this file path.

    Returns:
        None
            Displays the classification performance plots.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, RocCurveDisplay
    from sklearn.metrics import f1_score

    y_preds = (y_probs>threshold).astype(int)
    f1score = f1_score(y_true, y_preds)

    if vertical:
        fig, axes = plt.subplots(2, 1, figsize=(5, 10), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), constrained_layout=True)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion matrix")

    # --- ROC curve ---
    RocCurveDisplay.from_predictions(y_true,y_probs, ax=axes[1])
    axes[1].set_title("ROC curve")

    fig.suptitle(f"{title} - threshold: {threshold} F1: {f1score:.2f}")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



def display_optimal_threshold(y_true, y_probs):
    """
    Visualise how classification metrics change across probability thresholds
    and highlight optimal thresholds according to several criteria.

    This function evaluates model performance across a dense grid of probability
    thresholds and computes the following metrics for each threshold:
        - F1 score
        - Precision
        - Recall
        - G-mean (sqrt(sensitivity * specificity))
        - Total misclassification cost

    It then identifies the threshold that optimises each metric and plots all
    metrics as functions of the decision threshold.

    Args:
        y_true (array-like):
            Ground truth binary labels (0 or 1).

        y_probs (array-like):
            Predicted probabilities for the positive class.

    Returns:
        None
            Displays a matplotlib plot showing metric behaviour across thresholds.
    """
    from sklearn.metrics import f1_score, recall_score, precision_score
    
    # Dense grid of thresholds
    dense_thresholds = np.linspace( 0 , 1 , 1000 ) 

    # Initialization of scores
    f1_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    g_means = [] 
    costs = [] 

    # Cost of errors
    cost_fn = 500   # Cost of a false negative
    cost_fp = 5     # Cost of a false positive 

    # Calculation of metrics for each threshold 
    for threshold in dense_thresholds: 
        y_pred = [ 1  if prob >= threshold else  0  for prob in y_probs] 
        
        # Calculation of metrics
        f1 = f1_score(y_true, y_pred, zero_division= 1 ) 
        precision = precision_score(y_true, y_pred, zero_division= 1 ) 
        recall = recall_score(y_true, y_pred, zero_division= 1 ) 
        
        # Calculation of G-mean
        tn = sum ((y_true[i] == 0  and y_pred[i] == 0 ) for i in  range ( len (y_true))) 
        fp = sum ((y_true[i] == 0  and y_pred[i] == 1 ) for i in  range ( len (y_true))) 
        fn = sum ((y_true[i] == 1  and y_pred[i] == 0 ) for i in  range ( len (y_true))) 
        tp = sum ((y_true[i] == 1  and y_pred[i] == 1 )for i in  range ( len (y_true))) 
        
        tpr = tp / (tp + fn) if (tp + fn) > 0  else  0   # Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0  else  0   # Specificity
        
        # G-mean (balance between sensitivity and specificity)
        g_mean = np.sqrt(tpr * tnr) if tpr * tnr > 0  else  0 
        
        # Calculation of total cost
        total_cost = (fn * cost_fn) + (fp * cost_fp) 
        
        # Storage of values
        f1_scores.append(f1) 
        precision_scores.append(precision) 
        recall_scores.append(recall) 
        g_means.append(g_mean) 
        costs.append(total_cost) 

    # Selection of optimal thresholds according to each criterion
    best_f1_threshold = dense_thresholds[np.argmax(f1_scores)] 
    best_gmean_threshold = dense_thresholds[np.argmax(g_means)] 
    best_cost_threshold = dense_thresholds[np.argmin(costs)] 

    # Plot metrics versus threshold
    plt.figure(figsize=( 10 , 6 )) 
    plt.plot(dense_thresholds, f1_scores, label= 'F1-score' , color= 'orange' ) 
    plt.plot(dense_thresholds, precision_scores, label= 'Precision' , color= 'blue' ) 
    plt.plot(dense_thresholds, recall_scores, label= 'Recall' , color= 'red' ) 
    plt.plot(dense_thresholds, g_means, label= 'G-mean' , color= 'green' ) 

    # Display of optimal thresholds
    plt.axvline(x=best_f1_threshold, color= 'orange' , linestyle= '--' , label= f'Optimal F1-score = {best_f1_threshold: .2f} ' ) 
    plt.axvline(x=best_gmean_threshold, color= 'green' , linestyle= '--' , label= f'Optimal G-mean = {best_gmean_threshold: .2f} ' ) 
    plt.axvline(x=best_cost_threshold, color= 'purple' , linestyle= '--' , label= f'Optimal Cost = {best_cost_threshold: .2f} ' ) 

    plt.xlabel( 'Threshold' ) 
    plt.ylabel( 'Score' ) 
    plt.title( 'Optimal Threshold Selection - Top 0.1% Engineer Method' ) 
    plt.legend() 
    plt.grid(alpha= 0.3 )
    plt.show() 



def plot_patient_row(fig, gs_row, image_gt, image_pred, values_gt, values_pred,
                     rois, sample_metadata, leap_id, limits, cmap='seismic',
                     show_titles=False, ds=0.2, TITLESIZE='16'):
    """
    Plot a single patient row comparing measured and predicted spatial expression.

    This function creates a row in a multi-panel figure showing:
        1) Patient identifier label
        2) Ground-truth spatial expression overlay
        3) Predicted spatial expression overlay

    Each ROI is visualised as a coloured square positioned using ROI coordinates,
    with colour indicating gene expression level. A colourbar is added beside the
    predicted panel.

    Args:
        fig (matplotlib.figure.Figure):
            Figure object the row belongs to.

        gs_row (GridSpec):
            GridSpec row defining the layout for this patient.

        image_gt (np.ndarray):
            Background H&E image used for the ground truth panel.

        image_pred (np.ndarray):
            Background H&E image used for the prediction panel.

        values_gt (array-like):
            Ground-truth expression values corresponding to ROIs.

        values_pred (array-like):
            Predicted expression values corresponding to ROIs.

        rois (list-like):
            ROI identifiers used to index spatial coordinates in `sample_metadata`.

        sample_metadata (pd.DataFrame):
            Metadata table containing ROI spatial coordinates.
            Must contain columns:
                - 'scatter_x'
                - 'scatter_y'

        leap_id (str):
            Patient identifier displayed on the left side of the row.

        limits (tuple):
            Tuple of axis limits (xlim, ylim) used to crop/zoom the images.

        cmap (str or matplotlib colormap, optional):
            Colormap used to represent expression values.

        show_titles (bool, optional):
            Whether to display "Measured" and "Predicted" panel titles.

        ds (float, optional):
            Downsampling factor used to scale ROI rectangles.

        TITLESIZE (int or str, optional):
            Font size used for panel titles.

    Returns:
        None
            Draws the patient row into the provided figure.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Width of ROI rectangles (scaled by downsampling factor)
    rect_w = 224*ds

    # Shared color scale across measured and predicted values
    vmin = min(values_gt.min(), values_pred.min())
    vmax = max(values_gt.max(), values_pred.max())+0.5

    cmap = mpl.colormaps[cmap]  if isinstance(cmap, str) else cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # Left label
    ax_label = fig.add_subplot(gs_row[0])
    ax_label.axis('off')
    ax_label.text(0.95, 0.5, leap_id, fontsize=18, ha='right', va='center', transform=ax_label.transAxes,rotation=45 )

    # Ground Truth
    ax_gt = fig.add_subplot(gs_row[1])
    ax_gt.imshow(image_gt)
    # Overlay ROI rectangles coloured by measured expression
    for x, y, val in zip(sample_metadata.loc[rois, "scatter_x"],
                         sample_metadata.loc[rois, "scatter_y"],
                         values_gt):
        
        color = cmap(norm(val)) 
        rect = patches.Rectangle((x - rect_w/2, y - rect_w/2), rect_w, rect_w, facecolor=color, edgecolor=None, alpha=0.95)
        ax_gt.add_patch(rect)
    if show_titles:
        ax_gt.set_title("Measured", fontsize=TITLESIZE)
    ax_gt.set_xlim(limits[0])
    ax_gt.set_ylim(limits[1])
    ax_gt.axis('off')
    ax_gt.set_aspect('auto')
    # Prediction
    
    # Prediction
    ax_pred = fig.add_subplot(gs_row[2])
    ax_pred.imshow(image_pred)
    # Overlay ROI rectangles coloured by predicted expression
    for x, y, val in zip(sample_metadata.loc[rois, "scatter_x"],
                         sample_metadata.loc[rois, "scatter_y"],
                         values_pred):
        color = cmap(norm(val))
        rect = patches.Rectangle((x - rect_w/2, y - rect_w/2), rect_w, rect_w, facecolor=color, edgecolor=None, alpha=0.8)
        ax_pred.add_patch(rect)
    if show_titles:
        ax_pred.set_title("Predicted", fontsize=18)
    ax_pred.set_xlim(limits[0])
    ax_pred.set_ylim(limits[1])
    ax_pred.axis('off')
    ax_pred.set_aspect('auto')

    # Colorbar
    cbar_ax = inset_axes(
        ax_pred,           # attach to the predicted image plot
        width="3%",        # % of parent width
        height="100%",     # match height of ax_pred
        loc='center left',
        bbox_to_anchor=(1.02, 0., 1, 1),
        bbox_transform=ax_pred.transAxes,
        borderpad=0
    )
    cb = plt.colorbar(sm, cax=cbar_ax)
    cb.set_label("Expression Level", fontsize=14)
    # 1) force integer tick *locations*
    cb.locator = MaxNLocator(integer=True)  
    # 2) force integer tick *formatting*
    cb.formatter = FormatStrFormatter("%d")
    cb.update_ticks()



def plot_overlay(leap_ids, gene, roi_df,y_test, y_pred, heimgs, scaled_limits,
                 suffix="", SUPTITLESIZE=18 ):
    """
    Plot spatial overlays of measured and predicted expression for multiple patients.

    This function creates a multi-row figure where each row corresponds to a patient.
    For each patient it displays:
        - A label with the patient ID
        - Ground-truth spatial expression overlay
        - Predicted spatial expression overlay
        - A colourbar

    The overlays are generated using `plot_patient_row`, which draws coloured ROI
    rectangles over H&E images.

    Args:
        leap_ids (list):
            List of patient identifiers to plot.

        gene (str):
            Gene name being visualised.

        roi_df (pd.DataFrame):
            DataFrame containing ROI metadata including:
                - 'LEAP_ID'
                - 'scatter_x'
                - 'scatter_y'

        y_test (array-like or pd.Series):
            Ground-truth expression values corresponding to ROIs.

        y_pred (array-like or pd.Series):
            Predicted expression values corresponding to ROIs.

        heimgs (dict):
            Dictionary mapping LEAP_ID → H&E image.

        scaled_limits (dict):
            Dictionary mapping LEAP_ID → (xlim, ylim) used to crop images.

        suffix (str, optional):
            Optional suffix appended to the figure title.

        SUPTITLESIZE (int, optional):
            Font size for the figure-level title.

    Returns:
        None
            Displays the spatial overlay figure.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    fig = plt.figure(figsize=(10, 2.8 * len(leap_ids)), layout='constrained')
    gs = GridSpec(len(leap_ids), 4, width_ratios=[0.1, 1, 1, 0.05], figure=fig, wspace=0.05)

    n_rows = len(leap_ids)

    row_height = 1 / n_rows
    col = sns.color_palette("YlOrRd", as_cmap=True) 

    for i, leap_id in enumerate(leap_ids):
        rois = list(roi_df[roi_df['LEAP_ID'] == leap_id].index)
        values_gt = y_test #y_test.T.loc[rois, gene]
        values_pred = y_pred #y_pred.T.loc[rois, gene]
        lim = scaled_limits[leap_id]
        image = heimgs[leap_id]

        gs_row = gs[i, :]  # This is a SubplotSpec
        sub_gs = GridSpecFromSubplotSpec(
                1, 4,
                subplot_spec=gs_row,
                width_ratios=[0.1, 1, 1, 0.05],
                wspace=0.05
            )
        plot_patient_row(
            fig,
            sub_gs,
            image, image,
            values_gt,
            values_pred,
            rois,
            roi_df,
            f"Patient {i+1}",
            lim, cmap=col,
            show_titles=(i == 0)  # Only show titles on first row
        )

        row_y = 1 - (i + 1) / n_rows
        # Adjust margins if needed (e.g., make left/right padding smaller)

    fig.suptitle(f'Expression of {gene}{suffix}', fontsize=SUPTITLESIZE)
    plt.show()



def display_spatial_expression_on_axis(gene, he_img, subset_metadata, expr_matrix , xlim=None, ylim=None, ax=None):
    """
    Plot spatial gene expression on a provided matplotlib axis.

    This function overlays ROI coordinates on top of an H&E image and colours
    them according to the expression level of a specified gene. The plot can
    either be drawn on an existing axis or a new figure/axis will be created.

    Args:
        gene (str):
            Gene name to visualise.

        he_img (np.ndarray):
            Background H&E image (typically a downsampled RGB image).

        subset_metadata (pd.DataFrame):
            DataFrame containing ROI spatial coordinates. Must include:
                - 'ROICoordinateX_downsampled'
                - 'ROICoordinateY_downsampled'

        expr_matrix (pd.DataFrame):
            Gene expression matrix indexed by gene names with columns
            corresponding to ROIs.

        xlim (tuple, optional):
            Tuple specifying x-axis limits (xmin, xmax) for zooming.

        ylim (tuple, optional):
            Tuple specifying y-axis limits (ymin, ymax) for zooming.

        ax (matplotlib.axes.Axes, optional):
            Axis to draw the plot on. If None, a new figure and axis
            will be created.

    Returns:
        matplotlib.axes.Axes:
            Axis containing the spatial expression plot.
    """

    # Create a new axis if one is not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    else:
        # Retrieve the parent figure for colorbar placement
        fig = ax.figure

    # Display the background H&E image
    ax.imshow(he_img)

    # Overlay ROI locations coloured by expression value
    sc = ax.scatter(
        subset_metadata["ROICoordinateY_downsampled"],
        subset_metadata["ROICoordinateX_downsampled"],
        c=expr_matrix.loc[gene],
        cmap="viridis",
        alpha=0.8
    )

    # Attach a colorbar to the same axis
    fig.colorbar(sc, ax=ax, label="Expression level", fraction=0.046, pad=0.04)

    # Plot annotations
    ax.set_title(f"{gene}")

    # Remove axis labels and ticks for cleaner spatial visualisation
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    # Optional zooming
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax




def display_preds(he_img, metadata, y_preds, xlim=None, ylim=None, threshold=0.5, title="Probability of Responder"):
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    # Extract expression values for the gene
    expression_values = y_preds

    # Assign colors: above threshold -> 1, below or equal -> 0
    #colors = np.where(expression_values > threshold, 1, 0)
    colors = (expression_values > threshold).astype(int)
    cmap = ListedColormap(["#440154", "#FDE725"])

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.15) 
    ax.imshow(he_img)
    sc = ax.scatter(
        metadata["ROICoordinateY_downsampled"], 
        metadata["ROICoordinateX_downsampled"], 
        c=colors, cmap=cmap, alpha=0.8
    )
    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    legend_handles = [
        mpatches.Patch(color=cmap(0), label="Non-Responder"),
        mpatches.Patch(color=cmap(1), label="Responder")
    ]
    ax.legend(handles=legend_handles,    
        loc="upper left",
        bbox_to_anchor=(1.02, 0.95),
        frameon=True)

    

    fig.text(
        0.95,          # centered horizontally
        0.02,         # near bottom
        f"threshold={threshold}",
        ha="right",
        va="bottom",
        fontsize=8,
        color="black"
    )
    plt.show()


def display_preds_subplots(
    he_img,
    metadata,
    y_preds,
    xlim=None,
    ylim=None,
    threshold=0.5,
    title="Probability of Responder",
    ax=None,
    probs=False
):
    """
    Display predicted response values spatially on an H&E image.

    This function overlays ROI locations on a histology image and colours them
    according to either:
        - predicted probabilities of response, or
        - binary predictions based on a probability threshold.

    It is designed to be used within subplot grids where the axis is passed in.

    Args:
        he_img (np.ndarray):
            Background H&E image (typically a downsampled RGB image).

        metadata (pd.DataFrame):
            DataFrame containing ROI spatial coordinates. Must include:
                - 'ROICoordinateX_downsampled'
                - 'ROICoordinateY_downsampled'

        y_preds (array-like):
            Predicted probabilities or prediction scores corresponding to ROIs.

        xlim (tuple, optional):
            Tuple specifying x-axis limits (xmin, xmax) for zooming.

        ylim (tuple, optional):
            Tuple specifying y-axis limits (ymin, ymax) for zooming.

        threshold (float, optional):
            Threshold used to convert probabilities into binary predictions
            when `probs=False`. Default is 0.5.

        title (str, optional):
            Title displayed above the subplot.

        ax (matplotlib.axes.Axes, optional):
            Axis to plot on. Should generally be provided when used in
            subplot layouts.

        probs (bool, optional):
            If True, plots predicted probabilities using a continuous colour
            scale. If False, plots binary predictions based on the threshold.

    Returns:
        matplotlib.axes.Axes:
            Axis containing the spatial prediction visualisation.
    """

    from matplotlib.colors import ListedColormap

    # Display the background H&E image
    ax.imshow(he_img)

    # --- Probability visualisation ---
    if probs:
        sc = ax.scatter(
            metadata["ROICoordinateY_downsampled"],
            metadata["ROICoordinateX_downsampled"],
            c=y_preds,
            cmap="viridis",
            alpha=0.8
        )

    # --- Binary classification visualisation ---
    else:
        # Convert probabilities to binary predictions
        colors = (y_preds > threshold).astype(int)

        # Custom two-colour colormap for binary classes
        cmap = ListedColormap(["#440154", "#FDE725"])

        sc = ax.scatter(
            metadata["ROICoordinateY_downsampled"],
            metadata["ROICoordinateX_downsampled"],
            c=colors,
            cmap=cmap,
            alpha=0.8
        )

    # Set plot title
    ax.set_title(title)

    # Optional spatial zooming
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Remove axis ticks and labels for cleaner spatial display
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax


def display_pooled_patch_distrib(df, suffix='', title="", thresh=0.5, save_path=None):
    """
    Visualise patch-level prediction distributions pooled by patient.

    This function plots the distribution of patch/ROI-level predicted
    probabilities for each patient (LEAP_ID). Individual patch predictions are
    shown as jittered points, while the mean probability per patient is
    highlighted with a red diamond marker.

    A horizontal threshold line is drawn to indicate the decision boundary
    used for binary classification.

    Args:
        df (pd.DataFrame):
            DataFrame containing patch/ROI-level predictions. Must include:
                - 'LEAP_ID' : patient identifier
                - 'Response' : ground truth patient response
                - f'probs{suffix}' : predicted probability for each patch

        suffix (str, optional):
            Optional suffix appended to the probability column name
            (e.g. 'probs_calibrated').

        title (str, optional):
            Title prefix for the figure.

        thresh (float, optional):
            Probability threshold used for classification.

        save_path (str, optional):
            If provided, the figure will be saved to this path.

    Returns:
        None
            Displays the distribution plot.
    """
    #df = test_metadata.sort_values('LEAP_ID') 
    fig, ax = plt.subplots(figsize=(7, 4))
    patient = df.groupby("LEAP_ID").agg(
        p_mean=(f"probs{suffix}","mean"),
        response=("Response","max")
    ).reset_index()


    # Show the actual patch probs (this usually explains the “spikes”)
    sns.stripplot(
        data=df, x="LEAP_ID", y=f'probs{suffix}',
        jitter=0.18, size=3, alpha=0.5, color="black", ax=ax
    )

    # Patient pooled probability as red diamonds
    sns.scatterplot(
        data=patient, x="LEAP_ID", y="p_mean",
        marker="D", s=90, color="red", edgecolor="white", linewidth=0.8, ax=ax, zorder=5
    )

    ax.axhline(thresh, ls="--", color="blue", lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("P(Responder)")
    ax.set_xlabel("LEAP_ID")
    ax.set_title(f"{title} - Patch-level probability distributions")

    # Build mapping LEAP_ID -> response label
    resp_map = dict(zip(patient["LEAP_ID"], patient["response"]))

    # Current x tick labels (order matters)
    xticks = ax.get_xticks()
    xlabels = [t.get_text() for t in ax.get_xticklabels()]

    # New labels with response underneath
    new_labels = [
        f"{lab}\n{resp_map.get(lab, '')}"
        for lab in xlabels
    ]

    ax.set_xticks(xticks)
    ax.set_xticklabels(new_labels, fontsize=6)
    ax.tick_params(axis="x", labelrotation=45)


    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()




def pred_scatterplot(
    y_measured,
    y_pred,
    scores,
    genelist,
    title="Predicted vs Measured",
    num_genes=10
):
    """
    Plot predicted vs measured expression for multiple genes.

    This function generates a grid of scatter plots comparing predicted and
    measured gene expression values. Each subplot corresponds to a gene and
    includes an identity line (y=x) to indicate perfect prediction.

    The Pearson q-value for each gene is displayed in the panel.

    Args:
        y_measured (pd.DataFrame):
            Measured expression matrix with genes as rows and samples/ROIs as columns.

        y_pred (pd.DataFrame):
            Predicted expression matrix with the same structure as `y_measured`.

        scores (pd.DataFrame):
            DataFrame containing gene-level evaluation statistics.
            Must include columns:
                - 'Gene'
                - 'PearsonQVals'
                - 'PearsonPVals'

        genelist (list):
            List of genes to plot.

        title (str, optional):
            Figure title.

        num_genes (int, optional):
            Number of genes to plot. Determines the number of subplot rows.

    Returns:
        None
            Displays the scatter plot grid.
    """

    # Define subplot grid
    ncols = 5
    nrows = int(num_genes / ncols)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(10, nrows * 2),
        sharex=False,
        sharey=False
    )

    # Flatten axes array for easier indexing
    axs = axs.flatten()

    # Iterate over genes
    for i, genename in enumerate(genelist):

        # Retrieve significance values for the gene
        qval = scores.loc[scores['Gene'] == genename, 'PearsonQVals'].values[0]
        pval = scores.loc[scores['Gene'] == genename, 'PearsonPVals'].values[0]

        # Scatter plot: measured vs predicted expression
        sns.scatterplot(
            x=y_measured.loc[genename, :],
            y=y_pred.loc[genename, :],
            ax=axs[i]
        )

        # --- Identity line (perfect prediction reference) ---
        lims = [
            min(axs[i].get_xlim()[0], axs[i].get_ylim()[0]),
            max(axs[i].get_xlim()[1], axs[i].get_ylim()[1])
        ]
        axs[i].plot(lims, lims, '--', color='black', linewidth=1)

        # --- Display q-value annotation ---
        axs[i].text(
            0.95,
            0.05,
            f"p={qval:.2e}",
            transform=axs[i].transAxes,
            fontsize=8,
            ha='right',
            va='bottom',
            bbox=dict(facecolor='grey', edgecolor='none', alpha=0.6)
        )

        # Remove ticks for cleaner grid layout
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        # Axis labels
        axs[i].set_xlabel("Measured Expression", fontsize=9)
        axs[i].set_ylabel("Predicted Expression", fontsize=9)

        # Italicised gene name
        axs[i].set_title(r"$\it{" + genename + "}$")

    # Figure title
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()

    