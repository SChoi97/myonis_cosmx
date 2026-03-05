
"""
Script that takes target_call_coords.csv files and segmentaton masks (nuclei and myotubes) to assign transcripts to objects.
The output is two h5ad files (nuclei and myotubes) with count matrices, metadata, and linkage info.

Example csv file name: Run_fe397e89-03f0-4249-932e-8eba33cb77ee_FOV00001__target_call_coord.csv
Example segmentation file name: 20250724_142445_S1_C902_P99_N99_F00001_patch_0.txt

Original image size 4256x4256
Pixel size: 2.74um

Patch layout (1024x1024):
┌─────────────────────┐
│  0    1    2    3   │  ← Row 0 (y: 0-1023)
│                     │
│  4    5    6    7   │  ← Row 1 (y: 1024-2047)
│                     │
│  8    9   10   11   │  ← Row 2 (y: 2048-3071)
│                     │
│ 12   13   14   15   │  ← Row 3 (y: 3072-4095)
│                     │
└─────────────────────┘

160 pixels lost on the right and bottom of the image.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, issparse
import anndata as ad
from tqdm import tqdm
import time

from utils.cosmx_utils import (
    extract_fov_token_from_target_csv,
    load_polygons_for_fov,
    assign_counts,
    assign_counts_fast,
    load_gene_list,
    load_field_to_cell_line,
    pack_contours,
    make_contours_storage,
    print_adata_summary,
    extract_fov_token_from_name,
    make_cell_line_resolver,
    is_edge_polygon,
    compute_morphology_features,
    group_mask_paths_by_fov,
    load_polygons_from_paths,
    assign_nuclei_to_myotubes,  # Newly added
    assign_counts_raster,       # Newly added (for myotubes)
)

def parse_args():
    parser = argparse.ArgumentParser(description='CosMx Count Matrix Generation (Nuclei + Myotubes)')
    
    # Updated input paths
    parser.add_argument('--nuclei_contour_path', metavar='NUCLEI_PATH', required=True, help='Path to nuclei segmentation masks (yolov8 format)')
    parser.add_argument('--myotube_contour_path', metavar='MYOTUBE_PATH', required=True, help='Path to myotube segmentation masks (yolov8 format)')
    parser.add_argument('--overlap_threshold', type=float, default=0.8, help='Overlap threshold (0.0-1.0) for assigning nucleus to myotube')
    parser.add_argument(
        '--assignment_rule',
        choices=['greedy', 'shared', 'null', 'random'],
        default='greedy',
        help=(
            "Nuclei transcript assignment behavior in overlapping regions: "
            "'greedy' (first match), 'shared' (count for all matches), "
            "'null' (count only if exactly one match), "
            "'random' (balanced random sharing among overlapping matches)"
        ),
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=0,
        help='Random seed used when --assignment_rule=random (for reproducible sharing)',
    )
    parser.add_argument('--myhc_metadata', metavar='MYHC_METADATA', default=None, help='Optional CSV with myotube metadata for intensity filtering')
    parser.add_argument('--myhc_intensity_threshold', type=float, default=15, help='Minimum average_intensity to keep myotubes (8-bit scale)')
    
    parser.add_argument('--target_coords_path', metavar='TARGET_COORDS_PATH', help='Path to directory containing per-field subfolders (e.g., FOV00001) with target_call_coord.csv')
    parser.add_argument('--savepath', metavar='SAVEPATH', help='path to save .h5ad files')
    parser.add_argument('--genes_csv', metavar='GENES_CSV', required=True, help='CSV containing gene names to include')
    parser.add_argument('--patch_size', type=int, default=1024, help='Patch size in pixels used to denormalize segmentation coordinates')
    parser.add_argument('--grid_cols', type=int, default=4, help='Number of patch columns in the grid (used for offsets)')
    parser.add_argument('--field_ids_csv', metavar='FIELD_IDS_CSV', required=True, help='CSV mapping FOV indices to cell line names')
    parser.add_argument('--slide_name', metavar='SLIDE_NAME', default=None, help='Column name in field_ids_csv to use for FOV-to-cell-line mapping (e.g., T5R5)')
    parser.add_argument('--edge_threshold', type=int, default=0, help='Pixels from patch edge to consider an object an edge object (in original patch coords)')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers across fields (use >1 for speed)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose progress/timing logs')

    return parser.parse_args()

def _build_myotube_intensity_drop_map(metadata_path: Path, intensity_threshold: float):
    """Build image-stem -> set(object_idx) for myotubes to remove by intensity."""
    meta = pd.read_csv(metadata_path)
    required_cols = {'image_name', 'average_intensity'}
    missing = required_cols - set(meta.columns)
    if missing:
        raise ValueError(f"Missing required columns in --myhc_metadata: {missing}")

    idx_col = None
    for candidate in ('contour_idx', 'assignment_idx', 'myotube_id'):
        if candidate in meta.columns:
            idx_col = candidate
            break
    if idx_col is None:
        raise ValueError(
            "Could not map metadata rows to contour objects: expected one of "
            "['contour_idx', 'assignment_idx', 'myotube_id'] in --myhc_metadata"
        )

    work = meta.copy()
    work['_image_stem'] = work['image_name'].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    intensity_vals = pd.to_numeric(work['average_intensity'], errors='coerce')
    idx_vals = pd.to_numeric(work[idx_col], errors='coerce')
    fail_mask = intensity_vals < float(intensity_threshold)
    valid_mask = fail_mask & idx_vals.notna() & work['_image_stem'].notna()

    to_drop = work.loc[valid_mask, ['_image_stem']].copy()
    to_drop['obj_idx'] = idx_vals.loc[valid_mask].astype(np.int64).to_numpy()

    drop_map = {}
    if not to_drop.empty:
        for image_stem, sub_df in to_drop.groupby('_image_stem', sort=False):
            drop_map[str(image_stem)] = set(int(v) for v in sub_df['obj_idx'].tolist())

    summary = {
        'rows_total': int(work.shape[0]),
        'rows_below_threshold': int(fail_mask.sum()),
        'rows_mapped': int(valid_mask.sum()),
        'index_column': idx_col,
    }
    return drop_map, summary

def _filter_myotubes_by_intensity(myotube_objs, drop_map):
    """Remove objects whose (image_name-stem, per-image index) appears in drop_map."""
    if not drop_map:
        return myotube_objs, 0

    kept = []
    removed = 0
    image_to_next_idx = {}
    for obj in myotube_objs:
        image_stem = Path(str(obj.get('image_name', ''))).stem
        local_idx = image_to_next_idx.get(image_stem, 0)
        image_to_next_idx[image_stem] = local_idx + 1

        drop_set = drop_map.get(image_stem)
        if drop_set is not None and local_idx in drop_set:
            removed += 1
            continue
        kept.append(obj)
    return kept, removed

def create_anndata_from_objects(objects, counts, genes_out, fov_token, cell_line, patch_size, edge_threshold, prefix="obj", verbose=False):
    """Helper to create AnnData for a set of objects."""
    t0 = time.perf_counter()
    is_edge = [is_edge_polygon(obj['local_polygon'], patch_size, edge_threshold) for obj in objects]
    image_names = [obj['image_name'] for obj in objects]
    if verbose:
        print(f"[{fov_token}] computed edge flags for {len(objects)} objs in {time.perf_counter()-t0:.2f}s", flush=True)
    
    obs = pd.DataFrame({
        'object_id': [f"{fov_token}_{prefix}_{i}" for i in range(len(objects))],
        'field': [fov_token] * len(objects),
        'patch_idx': [obj['patch_idx'] for obj in objects],
        'Cell Line': [cell_line] * len(objects),
        'is_edge': is_edge,
        'image_name': image_names,
    })
    
    var = pd.DataFrame(index=pd.Index(genes_out, name='gene'))
    X = csr_matrix(counts)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    
    t2 = time.perf_counter()
    local_contours = [obj['local_polygon'] for obj in objects]
    contours_store = make_contours_storage(local_contours)
    offsets_arr = np.asarray([obj['offset'] for obj in objects], dtype=np.float32)
    
    adata.uns['Object Contours'] = {
        'Contours': contours_store,
        'Contour offsets': offsets_arr,
    }
    if verbose:
        print(f"[{fov_token}] stored contours ({len(local_contours)} objects) in {time.perf_counter()-t2:.2f}s", flush=True)
    
    t3 = time.perf_counter()
    morph_arr, morph_cols = compute_morphology_features(local_contours)
    adata.obsm['morphology_features'] = morph_arr
    adata.uns['morphology_feature_columns'] = morph_cols
    if verbose:
        print(f"[{fov_token}] computed morphology for {len(local_contours)} objs in {time.perf_counter()-t3:.2f}s", flush=True)
    
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
        
    return adata

def _process_field(field: Path, target_coords_path: Path, nuclei_paths, myotube_paths, fov_token: str, genes: list, patch_size: int, grid_cols: int, cell_line: str, edge_threshold: int, overlap_threshold: float, out_dir: Path, myotube_drop_map=None, myhc_intensity_threshold: float = 15, assignment_rule: str = 'greedy', random_seed: int = 0, verbose: bool = False):
    t_all = time.perf_counter()
    target_coords = pd.read_csv(target_coords_path)
    if verbose:
        print(f"[{fov_token}] loaded target coords: {len(target_coords)} rows", flush=True)
    
    # 1. Load objects
    # Load in nuclei contours
    t0 = time.perf_counter()
    nuclei_objs = load_polygons_from_paths(nuclei_paths, patch_size, grid_cols)
    if verbose:
        print(f"[{fov_token}] loaded nuclei objs: {len(nuclei_objs)} in {time.perf_counter()-t0:.2f}s", flush=True)
    # Load in myotube contours
    t1 = time.perf_counter()
    myotube_objs = load_polygons_from_paths(myotube_paths, patch_size, grid_cols)
    if myotube_drop_map is not None:
        myotube_before = len(myotube_objs)
        myotube_objs, myotube_removed = _filter_myotubes_by_intensity(myotube_objs, myotube_drop_map)
        if verbose:
            print(
                f"[{fov_token}] myotube intensity filter removed {myotube_removed}/{myotube_before} "
                f"objects (avg_intensity < {myhc_intensity_threshold})",
                flush=True,
            )
    if verbose:
        print(f"[{fov_token}] loaded myotube objs: {len(myotube_objs)} in {time.perf_counter()-t1:.2f}s", flush=True)
    # Compute per-patch local indices for nuclei and myotubes (0..n-1 within each patch)
    def _compute_patch_local_ids(objects):
        patch_to_next = {}
        local_ids = []
        for obj in objects:
            pid = int(obj['patch_idx'])
            lid = patch_to_next.get(pid, 0)
            obj['patch_local_id'] = lid
            local_ids.append(lid)
            patch_to_next[pid] = lid + 1
        return local_ids
    nuc_local_ids = _compute_patch_local_ids(nuclei_objs)
    myo_local_ids = _compute_patch_local_ids(myotube_objs)
    if verbose:
        try:
            num_patches_with_myo = len(set(int(o['patch_idx']) for o in myotube_objs))
            num_patches_with_nuc = len(set(int(o['patch_idx']) for o in nuclei_objs))
        except Exception:
            num_patches_with_myo = 0
            num_patches_with_nuc = 0
        print(f"[{fov_token}] indexed local IDs (patches: myo={num_patches_with_myo}, nuc={num_patches_with_nuc})", flush=True)
    
    print(f"FOV {fov_token}: nuclei={len(nuclei_objs)}, myotubes={len(myotube_objs)}, spots={len(target_coords)}", flush=True)
    
    # 2. Assign transcripts
    # Nuclei count matrix
    t2 = time.perf_counter()
    n_counts, n_genes = assign_counts_fast(
        nuclei_objs,
        genes,
        target_coords,
        patch_size,
        grid_cols,
        assignment_rule=assignment_rule,
        random_seed=random_seed,
    )
    if verbose:
        print(f"[{fov_token}] nuclei counts shape={n_counts.shape} in {time.perf_counter()-t2:.2f}s", flush=True)
    # Myotube count matrix (prefer rasterization for speed; fallback to polygon method)
    t3 = time.perf_counter()
    try:
        m_counts, m_genes = assign_counts_raster(myotube_objs, genes, target_coords, patch_size, grid_cols, verbose=verbose)
        if verbose:
            print(f"[{fov_token}] myotube counts (raster) shape={m_counts.shape} in {time.perf_counter()-t3:.2f}s", flush=True)
    except ImportError as e:
        if verbose:
            print(f"[{fov_token}] rasterization not available ({e}); falling back to polygon method", flush=True)
        m_counts, m_genes = assign_counts_fast(myotube_objs, genes, target_coords, patch_size, grid_cols)
        if verbose:
            print(f"[{fov_token}] myotube counts (polygon) shape={m_counts.shape} in {time.perf_counter()-t3:.2f}s", flush=True)
    
    # 3. Assign nuclei to myotubes
    # Returns {nuclei_idx: myotube_idx}
    t4 = time.perf_counter()
    assignment_map = assign_nuclei_to_myotubes(nuclei_objs, myotube_objs, overlap_threshold, verbose=verbose)
    if verbose:
        print(f"[{fov_token}] nuclei→myotube mapping {len(assignment_map)} assignments in {time.perf_counter()-t4:.2f}s", flush=True)
    
    # 4. Create AnnData objects
    t5 = time.perf_counter()
    adata_nuclei = create_anndata_from_objects(nuclei_objs, n_counts, n_genes, fov_token, cell_line, patch_size, edge_threshold, prefix="nuc", verbose=verbose)
    adata_myotubes = create_anndata_from_objects(myotube_objs, m_counts, m_genes, fov_token, cell_line, patch_size, edge_threshold, prefix="myo", verbose=verbose)
    # Add local IDs to obs for convenience
    adata_nuclei.obs['local_id'] = np.array(nuc_local_ids, dtype=np.int32)
    adata_myotubes.obs['local_id'] = np.array(myo_local_ids, dtype=np.int32)
    if verbose:
        print(f"[{fov_token}] built AnnData objects in {time.perf_counter()-t5:.2f}s", flush=True)
    
    # 5. Add assignment info
    # Add myotube_id to nuclei obs
    myotube_ids_int = []
    myotube_patch_idxs = []
    for i in range(len(nuclei_objs)):
        if i in assignment_map:
            m_idx = assignment_map[i]
            myotube_ids_int.append(int(myo_local_ids[m_idx]))
            myotube_patch_idxs.append(int(myotube_objs[m_idx]['patch_idx']))
        else:
            myotube_ids_int.append(-1)
            myotube_patch_idxs.append(-1)
    # Store as integer per-patch myotube ID; add myotube_patch_idx for disambiguation
    adata_nuclei.obs['myotube_id'] = np.array(myotube_ids_int, dtype=np.int32)
    adata_nuclei.obs['myotube_patch_idx'] = np.array(myotube_patch_idxs, dtype=np.int32)
    
    # Store map in myotubes uns
    # We use string keys (Object IDs) for the dictionary
    id_map = {}
    for n_idx, m_idx in assignment_map.items():
        n_patch = int(nuclei_objs[n_idx]['patch_idx'])
        m_patch = int(myotube_objs[m_idx]['patch_idx'])
        n_local = int(nuc_local_ids[n_idx])
        m_local = int(myo_local_ids[m_idx])
        n_id = f"{fov_token}_patch_{n_patch}_nuc_{n_local}"
        m_id = f"{fov_token}_patch_{m_patch}_myotube_{m_local}"
        id_map[n_id] = m_id
    adata_myotubes.uns['nuclei_to_myotube_map'] = id_map
    
    # 6. Save separate files
    path_nuclei = out_dir / f"{fov_token}_nuclei.h5ad"
    path_myotubes = out_dir / f"{fov_token}_myotubes.h5ad"
    
    t6 = time.perf_counter()
    adata_nuclei.write_h5ad(path_nuclei)
    adata_myotubes.write_h5ad(path_myotubes)
    
    # Explicitly confirm write completion
    print(f"Saved FOV {fov_token} to {path_nuclei} and {path_myotubes} (write {time.perf_counter()-t6:.2f}s, total {time.perf_counter()-t_all:.2f}s)", flush=True)

    return str(path_nuclei), str(path_myotubes)

def main(args):
    nuclei_dir = Path(args.nuclei_contour_path)
    myotube_dir = Path(args.myotube_contour_path)
    target_dir = Path(args.target_coords_path)
    save_dir = Path(args.savepath)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    field_dir = save_dir / 'field_data'
    combined_dir = save_dir / 'combined_data'
    field_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    nuclei_paths = list(nuclei_dir.glob('*.txt'))
    myotube_paths = list(myotube_dir.glob('*.txt'))
    
    fov_to_nuclei = group_mask_paths_by_fov(nuclei_paths)
    fov_to_myotubes = group_mask_paths_by_fov(myotube_paths)
    
    field_dirs = sorted([p for p in target_dir.iterdir() if p.is_dir()])
    
    print(f"Found {len(nuclei_paths)} nuclei masks, {len(myotube_paths)} myotube masks, {len(field_dirs)} field folders")

    genes = load_gene_list(Path(args.genes_csv))
    print(f"Loaded {len(genes)} genes")

    resolve_cell_line = make_cell_line_resolver(Path(args.field_ids_csv), slide_name=args.slide_name)

    myotube_drop_map = None
    if args.myhc_metadata:
        metadata_path = Path(args.myhc_metadata)
        if not metadata_path.exists():
            raise FileNotFoundError(f"--myhc_metadata does not exist: {metadata_path}")
        myotube_drop_map, filter_summary = _build_myotube_intensity_drop_map(
            metadata_path,
            args.myhc_intensity_threshold,
        )
        print(
            f"Loaded myhc metadata from {metadata_path} "
            f"(rows={filter_summary['rows_total']}, "
            f"below_threshold={filter_summary['rows_below_threshold']}, "
            f"mapped={filter_summary['rows_mapped']}, "
            f"index_col={filter_summary['index_column']}, "
            f"threshold={args.myhc_intensity_threshold})",
            flush=True,
        )
    
    # Prepare tasks
    tasks = []
    for field in field_dirs:
        csv_candidates = sorted(field.glob('*_target_call_coord.csv')) or sorted(field.glob('*.csv'))
        if not csv_candidates:
            print(f"Warning: No CSV found in {field}, skipping")
            continue
        target_coords_path = csv_candidates[0]
        fov_token = extract_fov_token_from_name(field.name)
        
        n_paths = fov_to_nuclei.get(fov_token, [])
        m_paths = fov_to_myotubes.get(fov_token, [])
        
        cell_line = resolve_cell_line(fov_token)
        
        tasks.append((
            field, 
            target_coords_path, 
            n_paths, 
            m_paths, 
            fov_token, 
            genes, 
            args.patch_size, 
            args.grid_cols, 
            cell_line, 
            args.edge_threshold, 
            args.overlap_threshold, 
            field_dir,
            myotube_drop_map,
            args.myhc_intensity_threshold,
            args.assignment_rule,
            args.random_seed,
            args.verbose
        ))
        
    nuclei_res_paths = []
    myotube_res_paths = []

    # Execute
    if args.n_workers and args.n_workers > 1:
        print(f"Parallel processing with {args.n_workers} workers… (tasks={len(tasks)})", flush=True)
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futures = [ex.submit(_process_field, *t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                try:
                    p_n, p_m = fut.result()
                    nuclei_res_paths.append(p_n)
                    myotube_res_paths.append(p_m)
                except Exception as e:
                    # Capture full traceback if possible
                    import traceback
                    print(f"Error in worker: {e}", flush=True)
                    traceback.print_exc()
    else:
        for t in tqdm(tasks):
            try:
                p_n, p_m = _process_field(*t)
                nuclei_res_paths.append(p_n)
                myotube_res_paths.append(p_m)
            except Exception as e:
                print(f"Error processing {t[4]}: {e}", flush=True)

    # Combine
    for name, paths in [('nuclei', nuclei_res_paths), ('myotubes', myotube_res_paths)]:
        if not paths:
            continue
        adatas = []
        for p in paths:
            try:
                adatas.append(ad.read_h5ad(p))
            except Exception:
                pass
        
        if adatas:
            combined = ad.concat(adatas, axis=0, join='outer')
            if not issparse(combined.X):
                combined.X = csr_matrix(combined.X)
            out_path = combined_dir / f'combined_{name}.h5ad'
            combined.write_h5ad(out_path)
            print(f"Saved {out_path}", flush=True)
            print_adata_summary(str(out_path), combined)

if __name__ == "__main__":
    main(parse_args())
