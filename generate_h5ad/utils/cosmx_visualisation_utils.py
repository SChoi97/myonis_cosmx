#---
# Helper functions
#--- 
import re
import cv2
import numpy as np
import pandas as pd
patch_layout = {
                0: (0, 0),       1: (1024, 0),       2: (2048, 0),       3: (3072, 0),
                4: (0, 1024),    5: (1024, 1024),    6: (2048, 1024),    7: (3072, 1024),
                8: (0, 2048),    9: (1024, 2048),   10: (2048, 2048),   11: (3072, 2048),
                12: (0, 3072),   13: (1024, 3072),   14: (2048, 3072),   15: (3072, 3072)
                }

def numericalSort(value):
    """
    Helper function to sort strings with groups of digits numerically.
    """
    numbers = re.findall(r'\d+', value.name)
    parts = re.split(r'\d+', value.name)
    result = []
    for i in range(max(len(parts), len(numbers))):
        if i < len(parts):
            result.append(parts[i])
        if i < len(numbers):
            result.append(int(numbers[i]))
    return result

def create_fiji_lut(color='cyan', n_colors=256):
    """
    Create a Fiji-style lookup table (LUT) colormap.
    
    Args:
        color: str, one of 'cyan', 'magenta', 'yellow', 'red', 'green', 'blue', 'gray', 'grays'
        n_colors: int, number of color steps (default 256)
    
    Returns:
        matplotlib.colors.LinearSegmentedColormap
    """
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define LUT color mappings (black to target color)
    lut_colors = {
        'cyan':    (0.0, 1.0, 1.0),  # RGB for cyan
        'magenta': (1.0, 0.0, 1.0),  # RGB for magenta
        'yellow':  (1.0, 1.0, 0.0),  # RGB for yellow
        'red':     (1.0, 0.0, 0.0),  # RGB for red
        'green':   (0.0, 1.0, 0.0),  # RGB for green
        'blue':    (0.0, 0.0, 1.0),  # RGB for blue
        'gray':    (1.0, 1.0, 1.0),  # RGB for white (grayscale)
        'grays':   (1.0, 1.0, 1.0),  # Alias for gray
    }
    
    if color.lower() not in lut_colors:
        raise ValueError(f"Color '{color}' not supported. Choose from: {list(lut_colors.keys())}")
    
    target_rgb = lut_colors[color.lower()]
    
    # Create color dictionary for LinearSegmentedColormap
    # Goes from black (0,0,0) to target color
    cdict = {
        'red':   [(0.0, 0.0, 0.0), (1.0, target_rgb[0], target_rgb[0])],
        'green': [(0.0, 0.0, 0.0), (1.0, target_rgb[1], target_rgb[1])],
        'blue':  [(0.0, 0.0, 0.0), (1.0, target_rgb[2], target_rgb[2])]
    }
    
    return LinearSegmentedColormap(f'fiji_{color.lower()}', cdict, N=n_colors)

def unpack_object_contours(obj_contours):
    """
    obj_contours: adata.uns['Object Contours']
      {
        'Contours': np.ndarray (n_cells, n_points, 2) OR
                    {'points': (sum_n_points, 2), 'offsets': (n_cells+1,)},
        'Contour offsets': np.ndarray (n_cells, 2)
      }
    Returns:
      contours_list: list of (n_points, 2) float32 arrays (local coords)
      offsets_list:  list of (dx, dy) floats per object 
    """
    contours = obj_contours['Contours']

    if isinstance(contours, dict):
        points = np.asarray(contours['points'], dtype=np.float32)       # (sum_n_points, 2)
        offsets = np.asarray(contours['offsets'], dtype=np.int32)       # (n_cells + 1,)
        contours_list = [points[offsets[i]:offsets[i+1]] for i in range(len(offsets) - 1)]
    else:
        arr = np.asarray(contours, dtype=np.float32)                     # (n_cells, n_points, 2)
        contours_list = [arr[i] for i in range(arr.shape[0])]

    offsets_arr = np.asarray(obj_contours.get('Contour offsets', []), dtype=np.float32)  # (n_cells, 2)
    offsets_list = [tuple(off) for off in offsets_arr]

    return contours_list, offsets_list

#Fetch patch transcripts

def filter_transcript_coordinates(transcript_df: pd.DataFrame, target_patch: int, patch_size: int = 1024, n_cols: int = 4, contours: list = None, contour_offsets: list = None):
    """
    Filter transcript coordinates to only include those within the target patch.
    Returns local coordinates (relative to patch origin) as list of (x, y) tuples.
    Optionally also filters and converts contours to local coordinates.

    Args:
    - transcript_df: DataFrame with 'x' and 'y' columns containing global coordinates
    - target_patch: Patch ID (0-indexed) to filter transcripts for
    - patch_size: Size of each patch in pixels (default: 1024)
    - n_cols: Number of columns in patch grid (default: 4)
    - contours: Optional list of contours (each with shape (n_points, 2)) in global coordinates
    - contour_offsets: Optional list of (x_offset, y_offset) tuples for each contour

    Returns:
    - If contours is None: List of (x, y) tuples with local coordinates
    - If contours provided: Tuple of (transcript_coords, filtered_contours)
    """
    # Create explicit patch layout mapping (patch_id -> (x_offset, y_offset))
    patch_layout = {
        0: (0, 0),       1: (1024, 0),       2: (2048, 0),       3: (3072, 0),
        4: (0, 1024),    5: (1024, 1024),    6: (2048, 1024),    7: (3072, 1024),
        8: (0, 2048),    9: (1024, 2048),   10: (2048, 2048),   11: (3072, 2048),
       12: (0, 3072),   13: (1024, 3072),   14: (2048, 3072),   15: (3072, 3072)
    }
    
    # Get bounds for target patch
    patch_x_offset, patch_y_offset = patch_layout[target_patch]
    x_min = patch_x_offset
    x_max = patch_x_offset + patch_size
    y_min = patch_y_offset
    y_max = patch_y_offset + patch_size
    
    # Filter transcripts within patch bounds
    mask = (
        (transcript_df['x'] >= x_min) & 
        (transcript_df['x'] < x_max) &
        (transcript_df['y'] >= y_min) & 
        (transcript_df['y'] < y_max)
    )
    filtered_df = transcript_df[mask]
    
    # Convert to local coordinates and return as list of tuples
    local_coords = [
        (x - patch_x_offset, y - patch_y_offset) 
        for x, y in zip(filtered_df['x'].values, filtered_df['y'].values)
    ]
    
    # If contours not provided, return only transcript coordinates
    if contours is None:
        return local_coords
    
    # Filter and convert contours to local coordinates
    filtered_contours = []
    for i, (contour, offset) in enumerate(zip(contours, contour_offsets)):
        # Get contour center in global coordinates
        contour_x_offset, contour_y_offset = offset
        
        # Check if contour is within patch bounds (using offset as reference point)
        if (x_min <= contour_x_offset < x_max and y_min <= contour_y_offset < y_max):
            # Convert contour to global coordinates first
            global_contour = contour + np.array([contour_x_offset, contour_y_offset])
            
            # Convert to patch-local coordinates
            local_contour = global_contour - np.array([patch_x_offset, patch_y_offset])
            filtered_contours.append(local_contour)
    
    return local_coords, filtered_contours


def visualise_labels(image: np.ndarray,
                     contours: list,
                     alpha: float = 0.35,
                     linewidth: int = 4,
                     target_gene_counts: list = None,
                     cmap: str = 'RdBu',
                     myotube_contours: list = None,
                     myotube_alpha: float = 0.25,
                     myotube_linewidth: int = 4,
                     myotube_line_color = (0, 0, 0),
                     myotube_fill_color = (200, 200, 200),
                     assignment_labels: list = None,
                     unassigned_color = '#B0B0B0',
                     nuclei_color = None,
                     myotube_color = None,
                     image_edge_outline: bool = True,
                     image_edge_border_width: int = 10):
    """
    Visualize training image with instances from contours.

    Args:
    - image: Training input image as numpy array (H, W, C), (H, W, 2), or (H, W)
    - contours: List of nuclei contours where each contour has shape (n_points, 2)
    - alpha: Opacity of the nuclei masks when overlaid on the image.
    - linewidth: Width of the nuclei contour.
    - target_gene_counts: Optional list of counts for each nucleus to determine color intensity
    - cmap: Colormap to use when target_gene_counts is provided (default: 'RdBu')
    - myotube_contours: Optional list of myotube contours (same format as nuclei)
    - myotube_alpha: Opacity of myotube fill overlay
    - myotube_linewidth: Width of myotube contour outline
    - myotube_line_color: Hex string or (R, G, B) for myotube outline color
    - myotube_fill_color: Hex string or (R, G, B) for myotube fill color
    - assignment_labels: Optional list of ints of length == len(contours). Value is index of
                         assigned myotube for each nucleus. -1 means unassigned.
                         If provided, nuclei i and myotube assignment_labels[i] share the same color.
    - unassigned_color: Hex string or (R, G, B) color for nuclei with label -1
    - nuclei_color: Hex string or (R, G, B) uniform color for nuclei when no labels are passed
    - myotube_color: Hex string or (R, G, B) uniform color for myotubes when no labels are passed
    - image_edge_outline: When False, suppress only the outline pixels that lie near
      the image border; fills remain. Default: True.
    - image_edge_border_width: Extra border (in pixels) to suppress beyond the line
      width when image_edge_outline=False. Default: 10.

    Returns:
    - overlay_image: Image with nuclei and optional myotube overlays.
    """
    # Handle different image formats
    if image.ndim == 2:
        # Single channel grayscale -> RGB
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 2:
        # 2-channel image (e.g., immunofluorescence)
        # Channel 0: cyan, Channel 1: gray
        ch0 = image[:, :, 0].astype(np.float32)
        ch1 = image[:, :, 1].astype(np.float32)
        
        # Normalize each channel to 0-1 range for better visualization
        if ch0.max() > 0:
            ch0 = (ch0 - ch0.min()) / (ch0.max() - ch0.min())
        if ch1.max() > 0:
            ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min())
        
        # Create RGB image: Channel 0=cyan (G+B), Channel 1=gray (R+G+B)
        image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        image[:, :, 0] = ch1           # Red from gray (ch1)
        image[:, :, 1] = ch0 + ch1     # Green from cyan (ch0) + gray (ch1)
        image[:, :, 2] = ch0 + ch1     # Blue from cyan (ch0) + gray (ch1)
        
        # Normalize and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    image = np.ascontiguousarray(image, dtype=np.uint8)

    def _to_rgb(color):
        if isinstance(color, str):
            hs = color.lstrip('#')
            if len(hs) == 6:
                return tuple(int(hs[i:i+2], 16) for i in (0, 2, 4))
            raise ValueError('Hex color must be 6 characters like #RRGGBB')
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) == 3:
            return tuple(int(c) for c in color)
        raise TypeError('Color must be a hex string or an RGB tuple/list/array')

    # Start from base image and optionally blend myotube fill first
    base_image = image.copy()

    # If myotube contours are provided, overlay their filled regions first
    if myotube_contours is not None and len(myotube_contours) > 0:
        myotube_fill_layer = base_image.copy()

        if assignment_labels is not None:
            # Build a palette sized to myotubes
            default_palette = [(255, 123, 156), (96, 113, 150), (255, 199, 89), (150, 230, 179),
                               (217, 61, 72), (242, 146, 72), (255, 221, 117), (228, 253, 225),
                               (187, 214, 134), (244, 144, 151), (245, 100, 118), (243, 217, 177),
                               (230, 230, 234), (238, 200, 224), (133, 199, 242), (167, 202, 177)]
            num_mt = len(myotube_contours)
            if num_mt <= len(default_palette):
                myotube_palette = default_palette[:num_mt]
            else:
                np.random.seed(0)
                rand_cols = np.random.randint(0, 255, (num_mt, 3))
                myotube_palette = [tuple(map(int, c)) for c in rand_cols]

            for j, mt_contour in enumerate(myotube_contours):
                mt_color = myotube_palette[j]
                contour_int = np.array(mt_contour, dtype=np.int32)
                cv2.drawContours(myotube_fill_layer, [contour_int], -1, mt_color, thickness=cv2.FILLED)
        else:
            # Uniform myotube fill color
            if myotube_color is not None:
                mt_fill_rgb = _to_rgb(myotube_color)
            else:
                mt_fill_rgb = _to_rgb(myotube_fill_color)
            for mt_contour in myotube_contours:
                contour_int = np.array(mt_contour, dtype=np.int32)
                cv2.drawContours(myotube_fill_layer, [contour_int], -1, mt_fill_rgb, thickness=cv2.FILLED)

        # Blend myotube fill onto the base image
        base_image = cv2.addWeighted(myotube_fill_layer, myotube_alpha, base_image, 1 - myotube_alpha, 0)

    # Prepare nuclei layers
    mask_image = np.zeros_like(base_image, dtype=np.uint8)
    outline_image = np.zeros_like(base_image, dtype=np.uint8)

    # Determine nuclei colors from gene counts if provided (used only when no labels and no nuclei_color)
    colors_from_counts = None
    if assignment_labels is None and nuclei_color is None and target_gene_counts is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        counts_array = np.array(target_gene_counts)
        norm = Normalize(vmin=counts_array.min(), vmax=counts_array.max())
        cmap_obj = plt.get_cmap(cmap)
        colors_from_counts = []
        for count in target_gene_counts:
            rgba = cmap_obj(norm(count))
            colors_from_counts.append((int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)))

    # Default palette for nuclei when needed
    default_palette = [(255, 123, 156), (96, 113, 150), (255, 199, 89), (150, 230, 179),
                       (217, 61, 72), (242, 146, 72), (255, 221, 117), (228, 253, 225),
                       (187, 214, 134), (244, 144, 151), (245, 100, 118), (243, 217, 177),
                       (230, 230, 234), (238, 200, 224), (133, 199, 242), (167, 202, 177)]
    random_palette = None

    # If assignment labels provided and myotubes exist, precompute palette to share colors
    shared_palette = None
    if assignment_labels is not None and myotube_contours is not None and len(myotube_contours) > 0:
        num_mt = len(myotube_contours)
        if num_mt <= len(default_palette):
            shared_palette = default_palette[:num_mt]
        else:
            np.random.seed(0)
            rand_cols = np.random.randint(0, 255, (num_mt, 3))
            shared_palette = [tuple(map(int, c)) for c in rand_cols]

    # Draw nuclei
    for i, contour in enumerate(contours):
        # Decide color for this nucleus
        if assignment_labels is not None and shared_palette is not None:
            label = assignment_labels[i]
            if label is None or label < 0 or label >= len(shared_palette):
                color = _to_rgb(unassigned_color)
            else:
                color = shared_palette[label]
        elif nuclei_color is not None:
            color = _to_rgb(nuclei_color)
        elif colors_from_counts is not None:
            color = colors_from_counts[i]
        else:
            if i < len(default_palette):
                color = default_palette[i]
            else:
                if random_palette is None:
                    np.random.seed(0)
                    rand_cols = np.random.randint(0, 255, (len(contours), 3))
                    random_palette = [tuple(map(int, c)) for c in rand_cols]
                color = random_palette[i]

        # Convert contour to int32 format required by OpenCV
        contour_int = np.array(contour, dtype=np.int32)
        cv2.drawContours(mask_image, [contour_int], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(outline_image, [contour_int], -1, color, thickness=linewidth)

    # Optionally suppress outline pixels near the image edges (keep fills)
    if not image_edge_outline:
        border = max(0, int(image_edge_border_width))
        margin = max(1, int(linewidth)) + border
        margin = min(margin, image.shape[0] // 2, image.shape[1] // 2)
        if margin > 0:
            outline_image[0:margin, :, :] = 0
            outline_image[image.shape[0] - margin:image.shape[0], :, :] = 0
            outline_image[:, 0:margin, :] = 0
            outline_image[:, image.shape[1] - margin:image.shape[1], :] = 0

    combined_mask = cv2.addWeighted(mask_image, 0.2, outline_image, 1.2, 0)
    overlay_image = cv2.addWeighted(base_image, 1, combined_mask, alpha, 0)

    # Draw myotube outlines on top (if provided)
    if myotube_contours is not None and len(myotube_contours) > 0:
        myotube_outline_layer = np.zeros_like(base_image, dtype=np.uint8)
        if assignment_labels is not None and shared_palette is not None:
            for j, mt_contour in enumerate(myotube_contours):
                line_color = shared_palette[j]
                contour_int = np.array(mt_contour, dtype=np.int32)
                cv2.drawContours(myotube_outline_layer, [contour_int], -1, line_color, thickness=myotube_linewidth)
        else:
            # Always use myotube_line_color for the outline
            mt_line_rgb = _to_rgb(myotube_line_color)
            for mt_contour in myotube_contours:
                contour_int = np.array(mt_contour, dtype=np.int32)
                cv2.drawContours(myotube_outline_layer, [contour_int], -1, mt_line_rgb, thickness=myotube_linewidth)

        # Optionally suppress myotube outline pixels near the edges
        if not image_edge_outline:
            border = max(0, int(image_edge_border_width))
            mt_margin = max(1, int(myotube_linewidth)) + border
            mt_margin = min(mt_margin, image.shape[0] // 2, image.shape[1] // 2)
            if mt_margin > 0:
                myotube_outline_layer[0:mt_margin, :, :] = 0
                myotube_outline_layer[image.shape[0] - mt_margin:image.shape[0], :, :] = 0
                myotube_outline_layer[:, 0:mt_margin, :] = 0
                myotube_outline_layer[:, image.shape[1] - mt_margin:image.shape[1], :] = 0

        mt_mask = myotube_outline_layer.sum(axis=2) > 0
        overlay_image[mt_mask] = myotube_outline_layer[mt_mask]

    return overlay_image

def visualise_transcripts(image: np.ndarray,
                          spots: list,
                          contours: list = None,
                          spot_color: str = '#FF0000',
                          spot_size: int = 3,
                          spot_outline_width: int = 1,
                          contour_alpha: float = 0.5,
                          linewidth: int = 2,
                          contour_color: str = '#C8C8C8',
                          myotube_contours: list = None,
                          myotube_linewidth: int = 2,
                          myotube_alpha: float = 0.25,
                          myotube_line_color: tuple = (0, 0, 0),
                          myotube_fill_color: tuple = (200, 200, 200),
                          image_edge_outline: bool = True,
                          image_edge_border_width: int = 10):
    """
    Visualize transcript spots on image, optionally with cell contours.

    Args:
    - image: Training input image as numpy array (H, W, C), (H, W, 2), or (H, W)
    - spots: List of (x, y) coordinates for transcript spots
    - contours: Optional list of contours where each contour has shape (n_points, 2)
    - spot_color: Hex color code or (R, G, B) tuple for spots (default: '#FF0000')
    - spot_size: Radius of the spots in pixels (default: 3)
    - spot_outline_width: Width of black outline around spots (default: 1)
    - contour_alpha: Opacity of the contour outlines (0=transparent, 1=opaque, default: 0.5)
    - linewidth: Width of the contour outline (default: 2)
    - contour_color: Hex color code or (R, G, B) tuple for contours (default: '#C8C8C8')
    - myotube_contours: Optional list of myotube contours (same format as nuclei)
    - myotube_linewidth: Width of the myotube contour outline (default: 2)
    - myotube_alpha: Transparency for myotube fill overlay (default: 0.25)
    - myotube_line_color: Line color for myotube contours as hex or (R, G, B)
    - myotube_fill_color: Fill color for myotube regions as hex or (R, G, B)
    - image_edge_outline: When False, suppress only the outline pixels that lie near
      the image border; fills remain. Default: True.
    - image_edge_border_width: Extra border (in pixels) to suppress beyond the line
      width when image_edge_outline=False. Default: 10.

    Returns:
    - overlay_image: Image with spots and optionally contours overlaid.
    """
    # Handle different image formats
    if image.ndim == 2:
        # Single channel grayscale -> RGB
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 2:
        # 2-channel image (e.g., immunofluorescence)
        # Channel 0: cyan, Channel 1: gray
        ch0 = image[:, :, 0].astype(np.float32)
        ch1 = image[:, :, 1].astype(np.float32)
        
        # Normalize each channel to 0-1 range for better visualization
        if ch0.max() > 0:
            ch0 = (ch0 - ch0.min()) / (ch0.max() - ch0.min())
        if ch1.max() > 0:
            ch1 = (ch1 - ch1.min()) / (ch1.max() - ch1.min())
        
        # Create RGB image: Channel 0=cyan (G+B), Channel 1=gray (R+G+B)
        image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        image[:, :, 0] = ch1           # Red from gray (ch1)
        image[:, :, 1] = ch0 + ch1     # Green from cyan (ch0) + gray (ch1)
        image[:, :, 2] = ch0 + ch1     # Blue from cyan (ch0) + gray (ch1)
        
        # Normalize and convert to uint8
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    
    image = np.ascontiguousarray(image, dtype=np.uint8)
    overlay_image = image.copy()

    def _to_rgb(color):
        if isinstance(color, str):
            hs = color.lstrip('#')
            if len(hs) == 6:
                return tuple(int(hs[i:i+2], 16) for i in (0, 2, 4))
            raise ValueError('Hex color must be 6 characters like #RRGGBB')
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) == 3:
            return tuple(int(c) for c in color)
        raise TypeError('Color must be a hex string or an RGB tuple/list/array')

    # If myotube contours are provided, overlay their filled regions first
    if myotube_contours is not None and len(myotube_contours) > 0:
        mt_fill_rgb = _to_rgb(myotube_fill_color)
        myotube_layer = overlay_image.copy()
        for contour in myotube_contours:
            contour_int = np.array(contour, dtype=np.int32)
            cv2.drawContours(myotube_layer, [contour_int], -1, mt_fill_rgb, thickness=cv2.FILLED)
        # Blend myotube fill onto the image
        overlay_image = cv2.addWeighted(myotube_layer, myotube_alpha, overlay_image, 1 - myotube_alpha, 0)

    # Draw contours if provided
    if contours is not None and len(contours) > 0:
        mask_image = np.zeros_like(image, dtype=np.uint8)
        outline_image = np.zeros_like(image, dtype=np.uint8)
        
        rgb_contour = _to_rgb(contour_color)
        
        for contour in contours:
            # Convert contour to int32 format required by OpenCV
            contour_int = np.array(contour, dtype=np.int32)
            cv2.drawContours(outline_image, [contour_int], -1, rgb_contour, thickness=linewidth)

        # Optionally suppress contour outline pixels near the image edges (keep fills)
        if not image_edge_outline:
            border = max(0, int(image_edge_border_width))
            margin = max(1, int(linewidth)) + border
            margin = min(margin, image.shape[0] // 2, image.shape[1] // 2)
            if margin > 0:
                outline_image[0:margin, :, :] = 0
                outline_image[image.shape[0] - margin:image.shape[0], :, :] = 0
                outline_image[:, 0:margin, :] = 0
                outline_image[:, image.shape[1] - margin:image.shape[1], :] = 0
        
        # Blend contour outlines with image
        overlay_image = cv2.addWeighted(overlay_image, 1, outline_image, contour_alpha, 0)

    # Draw myotube outlines on top (if provided)
    if myotube_contours is not None and len(myotube_contours) > 0:
        mt_line_rgb = _to_rgb(myotube_line_color)
        mt_outline_layer = np.zeros_like(image, dtype=np.uint8)
        for contour in myotube_contours:
            contour_int = np.array(contour, dtype=np.int32)
            cv2.drawContours(mt_outline_layer, [contour_int], -1, mt_line_rgb, thickness=myotube_linewidth)

        # Optionally suppress myotube outline pixels near the edges
        if not image_edge_outline:
            border = max(0, int(image_edge_border_width))
            mt_margin = max(1, int(myotube_linewidth)) + border
            mt_margin = min(mt_margin, image.shape[0] // 2, image.shape[1] // 2)
            if mt_margin > 0:
                mt_outline_layer[0:mt_margin, :, :] = 0
                mt_outline_layer[image.shape[0] - mt_margin:image.shape[0], :, :] = 0
                mt_outline_layer[:, 0:mt_margin, :] = 0
                mt_outline_layer[:, image.shape[1] - mt_margin:image.shape[1], :] = 0

        mt_mask = mt_outline_layer.sum(axis=2) > 0
        overlay_image[mt_mask] = mt_outline_layer[mt_mask]
    
    # Draw transcript spots
    if spots is not None and len(spots) > 0:
        rgb_color = _to_rgb(spot_color)
        
        # Draw each spot
        for spot in spots:
            x, y = int(spot[0]), int(spot[1])
            # Draw filled circle with spot color
            cv2.circle(overlay_image, (x, y), spot_size, rgb_color, -1)
            # Draw black outline
            if spot_outline_width > 0:
                cv2.circle(overlay_image, (x, y), spot_size, (0, 0, 0), spot_outline_width)

    return overlay_image

def visualize_voronoi(contours: list,
                       image_shape: tuple,
                       target_gene_counts: list = None,
                       cmap: str = 'RdBu',
                       linewidth: int = 2,
                       alpha: float = 0.7,
                       morphology_class_list: list = None,
                       morphology_class_colors: dict = None,
                       myotube_contours: list = None,
                       myotube_linewidth: int = 2,
                       myotube_alpha: float = 0.25,
                       myotube_line_color: tuple = (0, 0, 0),
                       myotube_fill_color: tuple = (200, 200, 200),
                       background_color: str = 'white',
                       image_edge_outline: bool = True,
                       image_edge_border_width: int = 10):
    """
    Visualize contours as a segmentation map with colored fills based on gene counts OR use to visualise class labels.

    Args:
    - contours: List of contours where each contour has shape (n_points, 2)
    - image_shape: Tuple (height, width) for the output image size
    - target_gene_counts: Optional list of counts for each contour to determine color intensity
    - cmap: Colormap to use when target_gene_counts is provided (default: 'RdBu')
    - linewidth: Width of the black contour outline (default: 2)
    - alpha: Transparency of colored fills (0=transparent, 1=opaque, default: 0.7)
    - morphology_class_list: Optional list of integer class labels for each nucleus
    - morphology_class_colors: Optional dict mapping class labels to colors (hex or RGB)
                               e.g., {-1: '#B5B5B5', 0: '#6BD6D6', 1: '#6BD6D6', 2: '#E7355B'}
    - myotube_contours: Optional list of myotube contours (same format as nuclei). If provided,
                        myotubes are filled and outlined using the parameters below.
    - myotube_linewidth: Width of the myotube contour outline (default: 2)
    - myotube_alpha: Transparency for myotube fill overlay (default: 0.25)
    - myotube_line_color: Line color for myotube contours as hex or (R, G, B) (default: black)
    - myotube_fill_color: Fill color for myotube regions as hex or (R, G, B) (default: light gray)
    - background_color: Background color, either 'white' or 'black' (default: 'white')
    - image_edge_outline: When False, suppress only the outline pixels that lie near
      the image border; fills remain. Default: True.
    - image_edge_border_width: Extra border (in pixels) to suppress beyond the line
      width when image_edge_outline=False. Default: 10.

    Returns:
    - voronoi_image: RGB image with contours on white background
    """
    # Create background
    height, width = image_shape
    if background_color == 'white':
        background = np.ones((height, width, 3), dtype=np.uint8) * 255
    elif background_color == 'black':
        background = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        raise ValueError(f"Invalid background color: {background_color}")
    # Initialize colored layer with same background color
    colored_layer = background.copy()

    def _to_rgb(color):
        if isinstance(color, str):
            hs = color.lstrip('#')
            if len(hs) == 6:
                return tuple(int(hs[i:i+2], 16) for i in (0, 2, 4))
            raise ValueError('Hex color must be 6 characters like #RRGGBB')
        if isinstance(color, (tuple, list, np.ndarray)) and len(color) == 3:
            return tuple(int(c) for c in color)
        raise TypeError('Color must be a hex string or an RGB tuple/list/array')
    
    # Determine colors based on morphology_class_list, target_gene_counts, or defaults
    if morphology_class_list is not None and morphology_class_colors is not None:
        # Use morphology class colors from dict
        class_color_map = {k: _to_rgb(v) for k, v in morphology_class_colors.items()}
        default_color = (128, 128, 128)  # fallback gray for unknown classes
        colors = [class_color_map.get(cls, default_color) for cls in morphology_class_list]
    elif target_gene_counts is not None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        
        # Normalize counts to [0, 1]
        counts_array = np.array(target_gene_counts)
        norm = Normalize(vmin=counts_array.min(), vmax=counts_array.max())
        cmap_obj = plt.get_cmap(cmap)
        
        # Generate colors from colormap
        colors = []
        for count in target_gene_counts:
            rgba = cmap_obj(norm(count))
            # Convert RGBA (0-1) to RGB (0-255)
            color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
            colors.append(color)
    else:
        # Default colors if no gene counts provided
        colors = [(255, 123, 156),
                  (96, 113, 150),
                  (255, 199, 89),
                  (150, 230, 179),
                  (217, 61, 72),
                  (242, 146, 72),
                  (255, 221, 117),
                  (228, 253, 225),
                  (187, 214, 134),
                  (244, 144, 151),
                  (245, 100, 118),
                  (243, 217, 177),
                  (230, 230, 234),
                  (238, 200, 224),
                  (133, 199, 242),
                  (167, 202, 177)]
    
    # Draw filled contours on colored layer
    for i, contour in enumerate(contours):
        if morphology_class_list is None and target_gene_counts is None:
            if i < len(colors):
                color = colors[i]
            else:
                np.random.seed(0)
                colors = np.random.randint(0, 255, (len(contours), 3))
                colors = [tuple(map(int, color)) for color in colors]
                color = colors[i]
        else:
            color = colors[i]
        
        # Convert contour to int32 format required by OpenCV
        contour_int = np.array(contour, dtype=np.int32)
        
        # Fill contour with color
        cv2.drawContours(colored_layer, [contour_int], -1, color, thickness=cv2.FILLED)
    
    # Blend colored layer with white background using alpha
    voronoi_image = cv2.addWeighted(colored_layer, alpha, background, 1 - alpha, 0)

    # If myotube contours are provided, overlay their filled regions with separate styling
    if myotube_contours is not None and len(myotube_contours) > 0:
        mt_fill_rgb = _to_rgb(myotube_fill_color)
        # Start from current image to ensure blending only affects myotube regions
        myotube_layer = voronoi_image.copy()
        for contour in myotube_contours:
            contour_int = np.array(contour, dtype=np.int32)
            cv2.drawContours(myotube_layer, [contour_int], -1, mt_fill_rgb, thickness=cv2.FILLED)
        # Blend myotube fill onto the image
        voronoi_image = cv2.addWeighted(myotube_layer, myotube_alpha, voronoi_image, 1 - myotube_alpha, 0)
    
    # Draw outlines for nuclei on top (white for black background, black for white background)
    nuclei_line_color = (255, 255, 255) if background_color == 'black' else (0, 0, 0)
    
    # Create outline layer for nuclei to support edge suppression
    nuclei_outline_layer = np.zeros((height, width, 3), dtype=np.uint8)
    for contour in contours:
        contour_int = np.array(contour, dtype=np.int32)
        cv2.drawContours(nuclei_outline_layer, [contour_int], -1, nuclei_line_color, thickness=linewidth)
    
    # Suppress edge-adjacent outline pixels if requested
    if not image_edge_outline:
        border = max(0, int(image_edge_border_width))
        margin = max(1, int(linewidth)) + border
        margin = min(margin, height // 2, width // 2)  # safety clamp
        if margin > 0:
            nuclei_outline_layer[0:margin, :, :] = 0          # top
            nuclei_outline_layer[height - margin:height, :, :] = 0  # bottom
            nuclei_outline_layer[:, 0:margin, :] = 0          # left
            nuclei_outline_layer[:, width - margin:width, :] = 0    # right
    
    # Composite nuclei outlines onto image
    nuclei_mask = cv2.cvtColor(nuclei_outline_layer, cv2.COLOR_BGR2GRAY)
    voronoi_image[nuclei_mask > 0] = nuclei_line_color

    # Draw myotube outlines on top (if provided)
    if myotube_contours is not None and len(myotube_contours) > 0:
        mt_line_rgb = _to_rgb(myotube_line_color)
        mt_outline_layer = np.zeros((height, width, 3), dtype=np.uint8)
        for contour in myotube_contours:
            contour_int = np.array(contour, dtype=np.int32)
            cv2.drawContours(mt_outline_layer, [contour_int], -1, mt_line_rgb, thickness=myotube_linewidth)
        
        # Suppress edge-adjacent outline pixels if requested
        if not image_edge_outline:
            border = max(0, int(image_edge_border_width))
            mt_margin = max(1, int(myotube_linewidth)) + border
            mt_margin = min(mt_margin, height // 2, width // 2)  # safety clamp
            if mt_margin > 0:
                mt_outline_layer[0:mt_margin, :, :] = 0
                mt_outline_layer[height - mt_margin:height, :, :] = 0
                mt_outline_layer[:, 0:mt_margin, :] = 0
                mt_outline_layer[:, width - mt_margin:width, :] = 0
        
        # Composite myotube outlines onto image
        mt_mask = cv2.cvtColor(mt_outline_layer, cv2.COLOR_BGR2GRAY)
        voronoi_image[mt_mask > 0] = mt_line_rgb
    
    return voronoi_image

default_palette = [(255, 123, 156), (96, 113, 150), (255, 199, 89), (150, 230, 179),
                       (217, 61, 72), (242, 146, 72), (255, 221, 117), (228, 253, 225),
                       (187, 214, 134), (244, 144, 151), (245, 100, 118), (243, 217, 177),
                       (230, 230, 234), (238, 200, 224), (133, 199, 242), (167, 202, 177)]
