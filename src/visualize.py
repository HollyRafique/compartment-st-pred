import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualise_feature_vector_heatmap(feature_vector):
    """
    Visualizes the feature vector using a heatmap.

    Parameters:
    - feature_vector: 1D torch.Tensor of shape [n_features].
    """
    #feature_vector = feature_vector.squeeze().detach().numpy()  # Remove the batch dimension
    feature_vector = feature_vector.reshape(1, -1)  # Reshape for heatmap

    plt.figure(figsize=(12, 2))
    sns.heatmap(feature_vector, cmap='viridis', annot=False, cbar=True, xticklabels=False)
    plt.title("Feature Vector Heatmap")
    plt.show()

def viz_side_by_side(original_image, feature_map, index=0, cmap='viridis'):
    try:
        # Extract and process the selected original image from the batch
        original_image = original_image

        # Extract and process the selected feature map from the batch
        feature_img = feature_map[index].transpose(0, 2).sum(-1).detach().numpy()

        # Plotting
        plt.figure(figsize=(12, 6))  # Adjust figure size as needed

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image {index}")

        # Plot feature map
        plt.subplot(1, 2, 2)
        plt.imshow(feature_img, cmap=cmap)
        plt.title(f"Feature Map {index}")

        plt.show()

    except IndexError:
        print(f"Index {index} is out of bounds for the batch size {original_image.shape[0]}.")
    except Exception as e:
        print(f"An error occurred: {e}")



def visualise_feature_matrix_heatmap(feature_matrix):
    """
    Visualizes a feature matrix using a heatmap.

    Parameters:
    - feature_matrix: NumPy array of shape (n_samples, n_features), where each row is a sample and each column is a feature.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the heatmap
    sns.heatmap(feature_matrix, cmap='viridis', annot=False, cbar=True)
    
    plt.title("Feature Matrix Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    
    plt.show()




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


    
def overlay_coords_on_img(img, subset_metadata, img_title=""):
    xcoords = subset_metadata.loc[:,"ROICoordinateY_downsampled"] #/ 4).astype(int)

    # Assign the corrected Y coordinate
    ycoords = subset_metadata.loc[:,"ROICoordinateX_downsampled"] #/ 4).astype(int)

    # Overlay corrected coordinates

    plt.scatter(
        xcoords,
        ycoords,
        label="Corrected FOVs",
        alpha=0.8,
        c="red",
        #edgecolors="black",
        s=20
    )
    # Add labels for each point using ROI
    for i, txt in enumerate(subset_metadata['Roi']):
        plt.text(
            xcoords.iloc[i]+20,
            ycoords.iloc[i],
            str(int(txt)),
            fontsize=6,
            ha='left', 
            va='center',  # Center the text vertically
            color="black"
        )

    # Set plot parameters
    plt.title(f"Corrected FOVs for {img_title}")
    plt.legend()

    #plt.axis("equal")

    plt.imshow(img) 
    plt.show()

def display_spatial_threshold(gene, he_img, metadata, expr_matrix, threshold ):

    # Extract expression values for the gene
    expression_values = expr_matrix.loc[gene]

    # Assign colors: above threshold -> 1, below or equal -> 0
    colors = np.where(expression_values > threshold, 1, 0)

    plt.imshow(he_img)
    plt.scatter(
        metadata["ROICoordinateY_downsampled"], 
        metadata["ROICoordinateX_downsampled"], 
        c=colors, cmap="viridis", alpha=0.8
    )
    plt.colorbar(label="Above (1) vs Below (0) Threshold")
    plt.title(f"Spatial Expression of {gene}(Threshold: {threshold})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def display_spatial_expression(gene, he_img, subset_metadata, expr_matrix , xlim=None, ylim=None, ax=None):

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

def single_violin(expression_df,gene):
    # Create a violin plot of expression values
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=expression_df["Expression"], color="skyblue", inner="box")
    # Calculate statistics
    mean_expression = expression_df["Expression"].mean()
    median_expression = expression_df["Expression"].median()


    # Add mean and median lines
    plt.axhline(mean_expression, color="red",  linewidth=1)
    plt.axhline(median_expression, color="green",  linewidth=1)
    plt.text(0, mean_expression, f"Mean: {mean_expression:.2f}", color="red", fontsize=8)
    plt.text(0, median_expression, f"Median: {median_expression:.2f}", color="green", fontsize=8)

    # Customize the plot
    plt.title(f"Distribution of Expression Values for {gene}")
    plt.ylabel("Expression Value")
    plt.xlabel("All Samples (ROIs)")
    plt.show()


def ometiff_to_rgb(img):
    from skimage import exposure
    # Load your IF image (e.g., 4-channel image with shape (4, height, width))

    # Initialize an empty RGB image
    rgb_image = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)

    # Map each channel to RGB components (example: Red, Green, Blue, and Yellow)
    rgb_image[:, :, 0] = exposure.rescale_intensity(img[0], in_range='image', out_range=(0, 1))  # Red
    rgb_image[:, :, 1] = exposure.rescale_intensity(img[1], in_range='image', out_range=(0, 1))  # Green
    rgb_image[:, :, 2] = exposure.rescale_intensity(img[2], in_range='image', out_range=(0, 1))  # Blue

    # If you have a fourth channel and want to add it as yellow (mixing red and green)
    rgb_image[:, :, 0] += exposure.rescale_intensity(img[3], in_range='image', out_range=(0, 0.5))
    rgb_image[:, :, 1] += exposure.rescale_intensity(img[3], in_range='image', out_range=(0, 0.5))

    # Clip values to ensure they stay within the 0-1 range
    rgb_image = np.clip(rgb_image, 0, 1)
    gamma = 0.45  # Choose a value less than 1 to make the image brighter
    brightened_channels = np.stack([exposure.adjust_gamma(channel, gamma) for channel in rgb_image.transpose(2, 0, 1)], axis=2)

    return brightened_channels



def plot_classification_results(y_true, y_probs, threshold=0.5, vertical=False, labels=[0,1], 
                            title="Classification Results", save_path=None):
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
    from sklearn.metrics import f1_score, recall_score, precision_score
    
    # Dense grid of thresholds
    dense_thresholds = np.linspace( 0 , 1 , 1000 ) 

    # Initialization of scores
    f1_scores = [] 
    precision_scores = [] 
    recall_scores = [] 
    g_means = [] 
    costs = [] 

    # Cost of errors (e.g., finance, medicine)
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

    # Displaying the optimal results
    #best_f1_threshold, best_gmean_threshold, best_cost_threshold


def plot_patient_row(fig, gs_row, image_gt, image_pred, values_gt, values_pred,
                     rois, sample_metadata, leap_id, limits, cmap='seismic',
                     show_titles=False, ds=0.2, TITLESIZE='16'):
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    rect_w = 224*ds
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
    if ax is None:
        print("none")
        fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    else:
        fig = ax.figure  # needed for colorbar

    ax.imshow(he_img)
    sc=ax.scatter(
        subset_metadata["ROICoordinateY_downsampled"], 
        subset_metadata["ROICoordinateX_downsampled"], 
        c=expr_matrix.loc[gene], cmap="viridis", alpha=0.8
    )
    # attach colorbar to THIS axes
    fig.colorbar(sc, ax=ax, label="Expression level", fraction=0.046, pad=0.04)
    ax.set_title(f"{gene}")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    
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


    

def display_preds_subplots(he_img, metadata, y_preds, xlim=None, ylim=None, 
                                threshold=0.5, title="Probability of Responder", ax=None, probs=False):
    from matplotlib.colors import ListedColormap
    # Extract expression values for the gene
    expression_values = y_preds

    # Assign colors: above threshold -> 1, below or equal -> 0
    #colors = np.where(expression_values > threshold, 1, 0)

    #fig, ax = plt.subplots()
    ax.imshow(he_img)

    if probs:
        sc = ax.scatter(
            metadata["ROICoordinateY_downsampled"], 
            metadata["ROICoordinateX_downsampled"], 
            c=y_preds, cmap="viridis", alpha=0.8
        )
        #ax.colorbar(label="Probability")
    else:
        
        colors = (expression_values > threshold).astype(int)
        cmap = ListedColormap(["#440154", "#FDE725"])
        sc = ax.scatter(
            metadata["ROICoordinateY_downsampled"], 
            metadata["ROICoordinateX_downsampled"], 
            c=colors, cmap=cmap, alpha=0.8
        )
    ax.set_title(title)
    #ax.set_xlabel("X Coordinate")
    #ax.set_ylabel("Y Coordinate")
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    #ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    return ax

def display_probs(he_img, metadata, y_probs , xlim=None, ylim=None, title="Probability of Responder"):
    plt.imshow(he_img)
    plt.scatter(
        metadata["ROICoordinateY_downsampled"], 
        metadata["ROICoordinateX_downsampled"], 
        c=y_probs, cmap="viridis", alpha=0.8
    )
    plt.colorbar(label="Probability")
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.show()

def display_pooled_patch_distrib(df, suffix='', title="", thresh=0.5, save_path=None):
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





def pred_scatterplot(y_measured, y_pred, scores, genelist, title="Predicted vs Measured", num_genes=10):

    ncols=5
    nrows=int(num_genes/ncols)

    fig, axs = plt.subplots( nrows,ncols, figsize=(10, nrows*2), sharex=False, sharey=False)
    axs = axs.flatten()


    for i, genename in enumerate(genelist):
        qval = scores.loc[scores['Gene']==genename,'PearsonQVals'].values[0]
        pval = scores.loc[scores['Gene']==genename,'PearsonPVals'].values[0]
        sns.scatterplot(x=y_measured.loc[genename,:], y=y_pred.loc[genename,:], ax=axs[i])

        #identity line
        lims = [
            min(axs[i].get_xlim()[0], axs[i].get_ylim()[0]),
            max(axs[i].get_xlim()[1], axs[i].get_ylim()[1])
        ]
        axs[i].plot(lims, lims, '--', color='black', linewidth=1)

        axs[i].text(
            0.95, 0.05, f"p={qval:.2e}",
            transform=axs[i].transAxes,
            fontsize=8, ha='right', va='bottom',
            bbox=dict(facecolor='grey', edgecolor='none', alpha=0.6)
        )


        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel("Measured Expression",fontsize=9)
        axs[i].set_ylabel("Predicted Expression",fontsize=9)
        axs[i].set_title(r"$\it{" + genename + "}$") 


    fig.suptitle(title)
    plt.tight_layout()
    plt.show()