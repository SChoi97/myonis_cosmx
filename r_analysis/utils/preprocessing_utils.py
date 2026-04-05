import pandas as pd

def to_target_format(df, slide_name, include_labels=True):
    # Normalize filename (remove path + extension)
    base = (
        df["Image Name"]
        .astype(str)
        .str.rsplit("/", n=1).str[-1]
        .str.replace(r"\.[^.]+$", "", regex=True)
    )

    # cellline can be alphanumeric (e.g., 1174, NCRM1, etc.)
    parsed = base.str.extract(
        r"^field_(?P<field>[^_]+)_patch_(?P<patch_idx>\d+)_cellline_(?P<cell_line>.+?)_localid_(?P<local_id>\d+)$"
    )

    bad = parsed.isna().any(axis=1)
    if bad.any():
        examples = df.loc[bad, "Image Name"].head(5).tolist()
        raise ValueError(f"{slide_name}: {bad.sum()} image names failed parsing. Examples: {examples}")

    out = pd.DataFrame({
        "Slide Name": slide_name,
        "field": parsed["field"],
        "patch_idx": parsed["patch_idx"].astype(int),
        "Cell Line": parsed["cell_line"],   # keep as string to support mixed types
        "local_id": parsed["local_id"].astype(int),
        "Image Name": df["Image Name"].values,
        "Predicted Class": df["Predicted Class"].values,
        "Sigmoid Logits": df["Sigmoid Logits"].values,
    })

    if include_labels:
        out["Labels"] = out["Cell Line"]

    # column order
    cols = ["Slide Name", "field", "patch_idx", "Cell Line", "local_id"]
    if include_labels:
        cols += ["Labels"]
    cols += ["Image Name", "Predicted Class", "Sigmoid Logits"]

    return out[cols]
