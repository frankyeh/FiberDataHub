import requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, matplotlib.colors as mcolors, math

def get_all_repos_for_account(account):
    return ["frankyeh/FiberDataHub"]

# Uses the correct repository name passed as 'repo' and the correct tag
def get_qc_asset_links(repo):
    # Use the repo argument correctly, and the confirmed working tag
    release_tag = "qc-data"
    url = f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}"
    try:
        resp = requests.get(url); resp.raise_for_status()
        return {a["name"].replace("_qc.tsv", ""): a["browser_download_url"]
                for a in resp.json().get("assets", []) if a["name"].endswith("_qc.tsv")}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching release data from {url}: {e}"); return {}

def process_qc_data(qc_url):
    """Download, filter qc.tsv, return DataFrame (None if error)."""
    try:
        with requests.get(qc_url, stream=True, allow_redirects=True) as r:
            r.raise_for_status(); df = pd.read_csv(r.raw, sep="\t")
    except requests.exceptions.RequestException as e: print(f"Failed to download {qc_url}: {e}"); return None
    except Exception as e: print(f"Failed to read/process data from {qc_url}: {e}"); return None

    if 'neighboring DWI correlation(masked)' in df.columns: df = df[df['neighboring DWI correlation(masked)'] <= 1]
    if 'DWI contrast' in df.columns: df = df[df['DWI contrast'] != 1]

    req_cols = ['neighboring DWI correlation(masked)', 'DWI contrast', 'dwi count(b0/dwi)']
    present_cols = [col for col in req_cols if col in df.columns]
    if not present_cols: print(f"Warning: No required columns found in {qc_url}. Skipping."); return None

    df = df[present_cols]
    # Corrected syntax for the loop and conditional assignment
    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df

def load_data_by_tag(repo, account=None):
    qc_links = get_qc_asset_links(repo); data_by_tag = {}
    if not qc_links: print(f"Warning: No QC asset links for '{repo}'."); return {}
    for tag, url in qc_links.items():
        if account and not (tag.startswith(account + '_') or tag == account): continue
        df = process_qc_data(url)
        if df is not None and not df.empty and not df.isnull().all().all(): data_by_tag[tag] = df
        else: print(f"Skipping asset '{tag}' due to issues or empty data.")
    return data_by_tag

def load_all_releases(account):
    all_data = {}; print("--- Loading Data ---")
    for repo in get_all_repos_for_account(account):
        print(f"Loading for repo: {repo}, account: '{account}'")
        data_by_tag = load_data_by_tag(repo, account=account)
        print(f"Found {len(data_by_tag)} matching tags in {repo} for account '{account}'.")
        short_repo_name = repo.split('/')[-1]
        for tag, df in data_by_tag.items(): all_data[f"{tag}({short_repo_name})"] = df
    print(f"Total releases loaded for account '{account}': {len(all_data)}")
    if not all_data: print(f"Warning: No data loaded for account '{account}'.")
    return all_data

def darken_color(color, factor=0.7):
    try: return tuple(np.array(mcolors.to_rgb(color)) * factor)
    except ValueError: print(f"Warning: Could not convert color {color}."); return color

def _plot_single_chunk(ax_scatter, ax_kde, keys, data_dict, account_y_range):
    n_colors = len(keys)
    if n_colors == 0: ax_scatter.set_visible(False); ax_kde.set_visible(False); return
    palette = sns.color_palette("tab20", max(n_colors, 20))[:n_colors]
    cluster_labels = {} # {(rx, ry): [ {x, y, tag, color}, ... ] }
    ndwic_col = 'neighboring DWI correlation(masked)'; dwic_col = 'DWI contrast'; count_col = 'dwi count(b0/dwi)'
    plot_xlim = (0.5, 1.0); plot_ylim = (0.9, 2.0)

    for i, key in enumerate(keys):
        df = data_dict.get(key)
        if df is None or ndwic_col not in df.columns or dwic_col not in df.columns:
            print(f"Warning: Skipping key '{key}' due to missing data/cols."); continue
        ndwic, dwic = df[ndwic_col].values, df[dwic_col].values
        valid_indices = np.isfinite(ndwic) & np.isfinite(dwic)
        if not np.any(valid_indices): continue
        ndwic, dwic = ndwic[valid_indices], dwic[valid_indices]

        marker_sizes = np.maximum(pd.to_numeric(df[count_col].astype(str).str.split('/').str[-1], errors='coerce').fillna(1).iloc[valid_indices] / 25.0, 5.0) \
                       if count_col in df.columns else np.ones_like(ndwic) * 10

        tag_only = key.split("(")[0]; simplified_tag = tag_only
        for prefix in ["data-hcp_", "abcd_", "dsi_", "HCP_"]:
            if simplified_tag.startswith(prefix): simplified_tag = simplified_tag[len(prefix):]; break
        if '_' in simplified_tag: simplified_tag = simplified_tag.split('_')[-1]
        if '-' in simplified_tag:
            parts = simplified_tag.split('-');
            if len(parts) > 1 and parts[0].lower() in ('hcp', 'abcd'): simplified_tag = '-'.join(parts[1:])
            elif parts[0].lower().startswith("ds00"): simplified_tag = simplified_tag[len("ds00"):]
        if not simplified_tag: simplified_tag = tag_only

        ax_scatter.scatter(ndwic, dwic, color=palette[i], label=simplified_tag, alpha=0.5, s=marker_sizes)
        color = palette[i]; dark_color = darken_color(color)

        if len(ndwic) > 1:
            try: sns.kdeplot(x=ndwic, y=dwic, fill=False, n_levels=1, thresh=0.5, color=dark_color, alpha=0.8, ax=ax_kde, zorder=10, warn_singular=False)
            except Exception as e: print(f"Warning: Could not generate KDE for '{key}': {e}")
        elif len(ndwic) == 1: ax_kde.scatter(ndwic, dwic, color=dark_color, s=30, marker='o', zorder=10)

        if len(ndwic) > 0:
            x_med, y_med = np.median(ndwic), np.median(dwic)
            if np.isfinite(x_med) and np.isfinite(y_med):
                cluster_key = (round(x_med, 1), round(y_med, 1))
                if cluster_key not in cluster_labels: cluster_labels[cluster_key] = []
                cluster_labels[cluster_key].append({"x": x_med, "y": y_med, "tag": simplified_tag, "color": dark_color})

    for center, labels in cluster_labels.items():
        if not labels: continue
        if len(labels) == 1:
            lab = labels[0]; label_x, label_y = np.clip(lab["x"], plot_xlim[0] + 0.02, plot_xlim[1] - 0.02), np.clip(lab["y"], plot_ylim[0] + 0.02, plot_ylim[1] - 0.02)
            ax_kde.text(label_x, label_y, lab["tag"], ha='center', va='center', fontsize=8, color=lab["color"], zorder=20, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
        else:
            n = len(labels); r = 0.03 * account_y_range if account_y_range > 0 else 0.01
            for idx, lab in enumerate(labels):
                angle = 2 * math.pi * idx / n; delta_x, delta_y = r * math.cos(angle) * 0.5, r * math.sin(angle)
                new_x, new_y = np.clip(lab["x"] + delta_x, plot_xlim[0] + 0.02, plot_xlim[1] - 0.02), np.clip(lab["y"] + delta_y, plot_ylim[0] + 0.02, plot_ylim[1] - 0.02)
                ax_kde.text(new_x, new_y, lab["tag"], ha='center', va='center', fontsize=8, color=lab["color"], zorder=20, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))

    for ax in [ax_scatter, ax_kde]:
        ax.set_xlabel("Neighboring DWI correlation (masked)", fontsize=9); ax.set_ylabel("DWI contrast", fontsize=9)
        ax.set_xlim(plot_xlim); ax.set_ylim(plot_ylim); ax.tick_params(axis='both', which='major', labelsize=8); ax.grid(True, linestyle=':', alpha=0.6)
    handles, labels = ax_scatter.get_legend_handles_labels()
    if handles: ax_scatter.legend(handles, labels, fontsize=7)

def plot_multi_account_releases(account_configs):
    if not account_configs: print("Error: No account configs."); return

    all_data_by_account = {}; print("--- Loading Data ---")
    for account, chunk_size in account_configs.items():
        if not isinstance(account, str) or not isinstance(chunk_size, int) or chunk_size <= 0:
            print(f"Warning: Invalid config for '{account}' ({chunk_size}). Skipping."); continue
        print(f"Processing account: {account} (chunk size: {chunk_size})")
        account_data = load_all_releases(account)
        if account_data:
            all_data_by_account[account] = {"data": account_data, "keys": sorted(account_data.keys()), "chunk_size": chunk_size}
        else: print(f"No data loaded for '{account}'. Not plotted.")

    if not all_data_by_account: print("--- Plotting Aborted: No data loaded. ---"); return
    total_keys_loaded = sum(len(info["keys"]) for info in all_data_by_account.values())
    print(f"\n--- Data Loading Complete: {len(all_data_by_account)} account(s), {total_keys_loaded} datasets. ---")

    total_plot_rows = 0; account_plot_info = {}; plot_ylim = (0.9, 2.0)
    print("--- Calculating Layout and Ranges ---")
    for account, info in all_data_by_account.items():
        keys, chunk_size, data_dict = info["keys"], info["chunk_size"], info["data"]
        num_account_chunks = math.ceil(len(keys) / chunk_size) if keys else 0
        account_plot_rows = math.ceil(num_account_chunks / 2)
        total_plot_rows += account_plot_rows

        account_y_vals = []; dwic_col = 'DWI contrast'
        for key in keys:
            df = data_dict.get(key);
            if df is not None and dwic_col in df.columns:
                valid_y = df[dwic_col].dropna();
                if not valid_y.empty: account_y_vals.extend(valid_y.values)

        min_y = max(np.min(account_y_vals) if account_y_vals else plot_ylim[0], plot_ylim[0])
        max_y = min(np.max(account_y_vals) if account_y_vals else plot_ylim[1], plot_ylim[1])
        account_y_range = max_y - min_y if max_y > min_y else 1.0

        account_plot_info[account] = {"plot_rows": account_plot_rows, "y_range": account_y_range, "num_chunks": num_account_chunks}
        print(f"Account '{account}': {len(keys)} datasets, {num_account_chunks} chunks, {account_plot_rows} plot rows, y-range ~{account_y_range:.2f}")

    if total_plot_rows == 0: print("--- Plotting Aborted: No valid data chunks. ---"); return

    fig, axes = plt.subplots(nrows=total_plot_rows, ncols=4, figsize=(20, total_plot_rows * 5), dpi=120, squeeze=False)
    print(f"\n--- Creating Plot Figure ({total_plot_rows} rows x 4 columns) ---")

    current_row_offset = 0; account_list = list(all_data_by_account.keys())
    for account in account_list:
        info, plot_info = all_data_by_account[account], account_plot_info[account]
        account_keys, account_data_dict, chunk_size = info["keys"], info["data"], info["chunk_size"]
        account_plot_rows, account_y_range, num_account_chunks = plot_info["plot_rows"], plot_info["y_range"], plot_info["num_chunks"]
        if account_plot_rows == 0: print(f"Skipping plotting for '{account}'."); continue
        print(f"\n--- Plotting Account: {account} ({account_plot_rows} rows assigned) ---")

        account_chunks = [account_keys[i:i + chunk_size] for i in range(0, len(account_keys), chunk_size)]

        for acc_plot_row_idx in range(account_plot_rows):
            fig_row_idx = current_row_offset + acc_plot_row_idx; chunk_idx1, chunk_idx2 = acc_plot_row_idx * 2, acc_plot_row_idx * 2 + 1
            if chunk_idx1 < num_account_chunks:
                keys1 = account_chunks[chunk_idx1]; ax_scatter1, ax_kde1 = axes[fig_row_idx, 0], axes[fig_row_idx, 1]
                print(f"  Row {fig_row_idx+1}, Group 1 ({account}, Chunk {chunk_idx1+1}, Keys: {len(keys1)})")
                _plot_single_chunk(ax_scatter1, ax_kde1, keys1, account_data_dict, account_y_range)
                ax_scatter1.set_title(f"{account} - Scatter", fontsize=10); ax_kde1.set_title(f"{account} - KDE & Labels", fontsize=10)
            else: axes[fig_row_idx, 0].set_visible(False); axes[fig_row_idx, 1].set_visible(False)
            if chunk_idx2 < num_account_chunks:
                keys2 = account_chunks[chunk_idx2]; ax_scatter2, ax_kde2 = axes[fig_row_idx, 2], axes[fig_row_idx, 3]
                print(f"  Row {fig_row_idx+1}, Group 2 ({account}, Chunk {chunk_idx2+1}, Keys: {len(keys2)})")
                _plot_single_chunk(ax_scatter2, ax_kde2, keys2, account_data_dict, account_y_range)
                ax_scatter2.set_title(f"{account} - Group {chunk_idx2+1} Scatter", fontsize=10); ax_kde2.set_title(f"{account} - Group {chunk_idx2+1} KDE & Labels", fontsize=10)
            else: axes[fig_row_idx, 2].set_visible(False); axes[fig_row_idx, 3].set_visible(False)
        current_row_offset += account_plot_rows

    plt.tight_layout(rect=[0.02, 0.02, 0.95, 0.97], h_pad=3, w_pad=3)
    plt.savefig('qc_plots.png')

# --- Example Usage ---
accounts_to_plot = {
    "data-hcp": 4,
    "data-abcd": 5,
    "data-openneuro": 10,
    "data-indi": 10,
    "data-others": 7
}

plot_multi_account_releases(accounts_to_plot)
