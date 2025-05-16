import requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, matplotlib.colors as mcolors, math
import io # Added for StringIO

def get_qc_asset_links(repo):
    """Fetches download links for QC TSV assets from a specific release tag."""
    release_tag = "qc-data"
    url = f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}"
    try:
        resp = requests.get(url)
        resp.raise_for_status() # Raise an exception for bad status codes
        # Return a dictionary mapping asset name (without _qc.tsv) to download URL
        return {a["name"].replace("_qc.tsv", ""): a["browser_download_url"]
                for a in resp.json().get("assets", []) if a["name"].endswith("_qc.tsv")}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching release data from {url}: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while fetching release data from {url}: {e}")
        return {}

def process_qc_data(qc_url):
    """Download, check line count, filter qc.tsv, return DataFrame (None if error or too few lines)."""
    try:
        r = requests.get(qc_url, allow_redirects=True)
        r.raise_for_status()
        content = r.content.decode('utf-8') # Decode content to string
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {qc_url}: {e}"); return None
    except Exception as e: # Catch other potential errors like decoding errors
        print(f"Error during download/decoding for {qc_url}: {e}"); return None

    # Check line count (header and data rows)
    lines = content.strip().splitlines()
    if len(lines) < 16: # Assuming at least 1 header row + 15 data rows needed
        print(f"Skipping {qc_url}: File has {len(lines)} lines (less than 16).")
        return None

    try:
        # Use io.StringIO to read the string content with pandas
        df = pd.read_csv(io.StringIO(content), sep="\t")
    except Exception as e:
        print(f"Failed to read/process data from {qc_url} with pandas: {e}"); return None

    # Apply filtering based on column values if columns exist
    if 'neighboring DWI correlation(masked)' in df.columns:
        df = df[df['neighboring DWI correlation(masked)'] <= 1]
    if 'DWI contrast' in df.columns:
        df = df[df['DWI contrast'] != 1]

    # Select required columns, add NaN for missing ones
    req_cols = ['neighboring DWI correlation(masked)', 'DWI contrast', 'dwi count(b0/dwi)']
    present_cols = [col for col in req_cols if col in df.columns]

    if not present_cols:
         print(f"Warning: No required columns found in {qc_url}. Skipping."); return None

    df = df[present_cols].copy() # Use .copy() to avoid SettingWithCopyWarning

    for col in req_cols:
        if col not in df.columns:
            df[col] = np.nan # Add missing required columns with NaN

    # Check if the dataframe is still valid after filtering/column selection
    if df.empty or df.isnull().all().all():
        print(f"Skipping {qc_url}: Data is empty or all NaN after processing.")
        return None

    return df

def load_data_for_account(account):
    """
    Loads QC data for a specific account, grouping by 'repo' extracted from the filename.
    Returns {repo_from_filename: {tag_suffix: df, ...}, ...}
    """
    all_data_grouped = {} # Structure: {repo_from_filename: {tag_suffix: df, ...}, ...}
    # Assume all data is in 'frankyeh/FiberDataHub' based on the example
    full_repo_name = "frankyeh/FiberDataHub"
    print(f"Loading assets from repo: {full_repo_name}, filtering for account prefix: '{account}'")

    # Get all asset links first from the single source repo
    qc_links = get_qc_asset_links(full_repo_name) # get_qc_asset_links already fetches from the specified repo
    if not qc_links:
        print(f"Warning: No QC asset links found in '{full_repo_name}'.")
        return {}

    # Process links, filter by account prefix, and group by repo_from_filename
    processed_tags_count = 0
    skipped_tags_count = 0

    for full_tag_key, url in qc_links.items(): # full_tag_key is like 'data-hcp_lifespan_hcp-ya' or 'abcd_release3'
        # Check if the tag starts with the target account prefix
        # Allow exact match for cases like 'abcd' if an asset was named 'abcd_qc.tsv'
        if not full_tag_key.startswith(account + '_') and full_tag_key != account:
            skipped_tags_count += 1
            continue

        # Parse the tag key to extract repo_from_filename and tag_suffix
        parts = full_tag_key.split('_')
        if len(parts) < 2:
            # Handle cases like 'abcd_qc.tsv' -> full_tag_key='abcd'
            if full_tag_key == account:
                 repo_from_filename = "misc" # Assign a default repo name if no underscore
                 tag_suffix = full_tag_key # The tag suffix is the full key in this case
            else:
                print(f"Warning: Skipping tag '{full_tag_key}' with unexpected format (less than 2 parts after split).")
                skipped_tags_count += 1
                continue
        else:
             # Assuming format is account_repo_..._tag
             # The account is the first part. The repo_from_filename is the second part.
             # The tag_suffix is the rest joined by underscores.
             # Example: 'data-hcp_lifespan_hcp-ya' -> parts = ['data-hcp', 'lifespan', 'hcp-ya']
             # Example: 'data-openneuro_ds001_R1' -> parts = ['data-openneuro', 'ds001', 'R1']
             account_prefix_found = parts[0]
             if account_prefix_found != account:
                 # Should be caught by startswith check, but defensive check
                 print(f"Logic error: Tag '{full_tag_key}' prefix '{account_prefix_found}' doesn't match target '{account}'.")
                 skipped_tags_count += 1
                 continue

             repo_from_filename = parts[1]
             tag_suffix_parts = parts[2:]
             if tag_suffix_parts:
                 tag_suffix = '_'.join(tag_suffix_parts)
             else:
                 # Format was just account_repo, maybe the repo name is the tag?
                 # E.g., 'data-hcp_lifespan'. Use the repo name as the tag_suffix in this case.
                 tag_suffix = repo_from_filename


        # Ensure the repo key exists in our grouping dictionary
        if repo_from_filename not in all_data_grouped:
            all_data_grouped[repo_from_filename] = {} # {tag_suffix: df}

        # Process the actual data file
        df = process_qc_data(url)

        if df is not None: # process_qc_data returns None if skipping
            all_data_grouped[repo_from_filename][tag_suffix] = df # Store df under repo and tag_suffix
            processed_tags_count += 1
        else:
            # process_qc_data already printed a message if it skipped the file
            skipped_tags_count += 1


    total_tags_loaded = sum(len(tags) for tags in all_data_grouped.values())
    print(f"Finished loading for account '{account}' from '{full_repo_name}'. Loaded {total_tags_loaded} tags ({processed_tags_count} processed, {skipped_tags_count} skipped).")
    if not all_data_grouped: print(f"Warning: No data loaded for account '{account}'.")
    return all_data_grouped # Returns {repo_from_filename: {tag_suffix: df, ...}, ...} for this account


def darken_color(color, factor=0.7):
    """Darkens a given color."""
    try:
        # Ensure input is a valid color string or tuple
        rgb = mcolors.to_rgb(color)
        return tuple(np.array(rgb) * factor)
    except ValueError:
        print(f"Warning: Could not convert color {color}. Using original color.")
        return color
    except Exception as e:
        print(f"An unexpected error occurred while darkening color {color}: {e}")
        return color


def _plot_single_chunk(ax_scatter, ax_kde, tag_suffixes, data_dict_for_group, current_group_y_range):
    """Plots a single chunk of tag_suffixes on the provided axes."""
    # tag_suffixes are the keys (e.g., 'hcp-ya', 'release3') for the current group
    # data_dict_for_group is {tag_suffix: df} for the current (account, repo_from_filename) group

    n_colors = len(tag_suffixes)
    if n_colors == 0:
        ax_scatter.set_visible(False)
        ax_kde.set_visible(False)
        return

    palette = sns.color_palette("tab20", max(n_colors, 20))[:n_colors]
    cluster_labels = {} # {(rx, ry): [ {x, y, tag, color}, ... ] }

    ndwic_col = 'neighboring DWI correlation(masked)'
    dwic_col = 'DWI contrast'
    count_col = 'dwi count(b0/dwi)'
    plot_xlim = (0.5, 1.0)
    plot_ylim = (0.9, 2.0) # Fixed plot limits for consistency

    for i, tag_suffix in enumerate(tag_suffixes): # Iterate through tag_suffixes
        df = data_dict_for_group.get(tag_suffix) # Lookup using tag_suffix

        # Basic check if df is valid for plotting
        if df is None or df.empty or ndwic_col not in df.columns or dwic_col not in df.columns:
             print(f"Warning: Skipping plot for tag_suffix '{tag_suffix}' due to missing data, empty DataFrame, or missing required columns.")
             continue

        ndwic = df[ndwic_col].values
        dwic = df[dwic_col].values

        # Ensure both columns have valid data for plotting
        valid_indices = np.isfinite(ndwic) & np.isfinite(dwic)
        if not np.any(valid_indices):
            print(f"Warning: Skipping plot for tag_suffix '{tag_suffix}': No valid (finite) data points in required columns.")
            continue

        ndwic_valid, dwic_valid = ndwic[valid_indices], dwic[valid_indices]

        # Calculate marker sizes based on dwi count if available
        marker_sizes = np.ones_like(ndwic_valid) * 10 # Default size
        if count_col in df.columns:
            try:
                 # Extract DWI count from the last part after '/' and convert to numeric
                 dwi_counts = pd.to_numeric(df[count_col].astype(str).str.split('/').str[-1], errors='coerce').fillna(1)
                 marker_sizes = np.maximum(dwi_counts.iloc[valid_indices] / 25.0, 5.0) # Scale and set min size
            except Exception as e:
                 print(f"Warning: Could not calculate marker sizes for tag_suffix '{tag_suffix}': {e}. Using default size.")
                 marker_sizes = np.ones_like(ndwic_valid) * 10


        # Simplify the tag_suffix for plot labels
        simplified_tag = tag_suffix
        # Keep the simplification logic, it operates on the tag_suffix now
        # Attempt to remove common prefixes from the tag suffix itself
        for prefix in ["hcp_", "abcd_", "dsi_", "HCP_"]:
             if simplified_tag.startswith(prefix):
                  simplified_tag = simplified_tag[len(prefix):]
                  break # Apply only the first matching prefix

        # If there's still an underscore, take the part after the last one
        if '_' in simplified_tag:
             simplified_tag = simplified_tag.split('_')[-1]

        # If there's a hyphen, try to simplify based on common patterns
        if '-' in simplified_tag:
             parts = simplified_tag.split('-')
             if len(parts) > 1:
                 # Handle patterns like hcp-ya, abcd-releaseN, ds001-R1
                 if parts[0].lower() in ('hcp', 'abcd'):
                     simplified_tag = '-'.join(parts[1:]) # Keep parts after the first common prefix
                 elif parts[0].lower().startswith("ds00"):
                      simplified_tag = '-'.join(parts[1:]) # Keep parts after ds00 prefix
                 # Add other patterns if needed
             # else: # If only one part or no clear pattern, keep as is

        if not simplified_tag: # Fallback if simplification results in empty string
             simplified_tag = tag_suffix


        ax_scatter.scatter(ndwic_valid, dwic_valid, color=palette[i], label=simplified_tag, alpha=0.5, s=marker_sizes)
        color = palette[i]
        dark_color = darken_color(color)

        # Plot KDE if enough data points, otherwise plot median point
        if len(ndwic_valid) > 1:
             try:
                 # Use valid data points for KDE
                 sns.kdeplot(x=ndwic_valid, y=dwic_valid, fill=False, n_levels=1, thresh=0.5, color=dark_color, alpha=0.8, ax=ax_kde, zorder=10, warn_singular=False)
             except Exception as e:
                 print(f"Warning: Could not generate KDE for tag_suffix '{tag_suffix}': {e}")
        elif len(ndwic_valid) == 1: # Only one data point
             ax_kde.scatter(ndwic_valid, dwic_valid, color=dark_color, s=30, marker='o', zorder=10)


        # Store median coordinates and label info for text annotations
        if len(ndwic_valid) > 0:
             x_med, y_med = np.median(ndwic_valid), np.median(dwic_valid)
             if np.isfinite(x_med) and np.isfinite(y_med):
                 # Cluster labels by rounded median coordinates
                 cluster_key = (round(x_med, 1), round(y_med, 1))
                 if cluster_key not in cluster_labels:
                      cluster_labels[cluster_key] = []
                 cluster_labels[cluster_key].append({"x": x_med, "y": y_med, "tag": simplified_tag, "color": dark_color})

    # Plot text labels for clusters of median points
    for center, labels in cluster_labels.items():
        if not labels: continue

        if len(labels) == 1:
             # Plot a single label directly at the median position (clipped)
             lab = labels[0]
             # Ensure labels are within the *fixed* plot limits (0.5, 1.0) and (0.9, 2.0)
             label_x = np.clip(lab["x"], plot_xlim[0] + 0.02, plot_xlim[1] - 0.02)
             label_y = np.clip(lab["y"], plot_ylim[0] + 0.02, plot_ylim[1] - 0.02)
             ax_kde.text(label_x, label_y, lab["tag"], ha='center', va='center', fontsize=8, color=lab["color"], zorder=20,
                         bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
        else:
             # Spread multiple labels in a cluster around the median center
             n = len(labels)
             # Adjust radius based on the *current group's* y_range for better spacing in that specific subplot
             # Use current_group_y_range passed from the calling function
             r = 0.03 * current_group_y_range if current_group_y_range > 0 else 0.01
             r = min(r, 0.05) # Cap the radius to avoid excessively large label clusters

             for idx, lab in enumerate(labels):
                 angle = 2 * math.pi * idx / n # Angle for positioning
                 # Calculate offset from the median center
                 delta_x, delta_y = r * math.cos(angle) * 0.5, r * math.sin(angle) # Adjust delta_x for aspect ratio

                 # Calculate new position and clip to stay within the fixed plot limits
                 new_x = np.clip(lab["x"] + delta_x, plot_xlim[0] + 0.02, plot_xlim[1] - 0.02)
                 new_y = np.clip(lab["y"] + delta_y, plot_ylim[0] + 0.02, plot_ylim[1] - 0.02)

                 ax_kde.text(new_x, new_y, lab["tag"], ha='center', va='center', fontsize=8, color=lab["color"], zorder=20,
                             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))


    # Set consistent plot limits and labels for both scatter and KDE axes
    for ax in [ax_scatter, ax_kde]:
        ax.set_xlabel("Neighboring DWI correlation (masked)", fontsize=9)
        ax.set_ylabel("DWI contrast", fontsize=9)
        ax.set_xlim(plot_xlim)
        ax.set_ylim(plot_ylim)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)

    # Add legend to the scatter plot axis
    handles, labels = ax_scatter.get_legend_handles_labels()
    if handles:
        ax_scatter.legend(handles, labels, fontsize=7)

def plot_multi_account_releases(account_configs):
    """
    Loads data for specified accounts and plots QC metrics grouped by account and repo (from filename).
    Each (account, repo) group is chunked based on the account's chunk_size.
    """
    if not account_configs:
        print("Error: No account configurations provided."); return

    # Structure to hold loaded data: {account: {repo_from_filename: {tag_suffix: df, ...}, ...}, ...}
    all_data_by_account_repo = {}
    print("--- Loading Data ---")

    # Load data for each specified account
    for account, chunk_size in account_configs.items():
        if not isinstance(account, str) or not isinstance(chunk_size, int) or chunk_size <= 0:
            print(f"Warning: Invalid config for account '{account}' (chunk size: {chunk_size}). Skipping.")
            continue

        account_repo_data = load_data_for_account(account) # Returns {repo_from_filename: {tag_suffix: df, ...}, ...}
        if account_repo_data:
            all_data_by_account_repo[account] = {"data": account_repo_data, "chunk_size": chunk_size}
        else:
             print(f"No data loaded for account '{account}'. This account will not be plotted.")

    # Check if any data was loaded for plotting
    if not all_data_by_account_repo:
        print("--- Plotting Aborted: No data loaded for any specified account. ---"); return

    # Calculate the total number of rows needed for the figure
    total_plot_rows = 0
    # Store info needed for plotting each (account, repo_from_filename) group
    plot_group_info = {} # { (account, repo_from_filename): {"plot_rows": N, "y_range": R, "num_chunks": M} }
    plot_ylim = (0.9, 2.0) # Fixed plot limits for calculating y-range if no data

    print("\n--- Calculating Layout and Ranges ---")

    # Iterate through accounts and repos_from_filename to determine layout and y-ranges
    # This loop calculates how many rows each (account, repo) group will take
    for account, account_info in all_data_by_account_repo.items():
        account_repo_data = account_info["data"] # {repo_from_filename: {tag_suffix: df, ...}, ...}
        account_chunk_size = account_info["chunk_size"]

        # Sort repos for consistent plot order within an account
        sorted_repos = sorted(account_repo_data.keys())

        for repo_from_filename in sorted_repos:
            repo_data = account_repo_data[repo_from_filename] # {tag_suffix: df, ...}
            repo_keys = sorted(repo_data.keys()) # These are the tag_suffixes

            if not repo_keys:
                print(f"Warning: No tags found for account '{account}', repo '{repo_from_filename}'. Skipping layout calculation for this group.")
                continue

            # Calculate number of chunks and plot rows for this specific (account, repo) group
            num_repo_chunks = math.ceil(len(repo_keys) / account_chunk_size)
            repo_plot_rows = math.ceil(num_repo_chunks / 2) # 2 plot columns (scatter + kde) per row

            # Calculate y-range for this (account, repo_from_filename) group's data
            current_group_y_vals = []
            dwic_col = 'DWI contrast'
            for tag_suffix in repo_keys: # Iterate through tag_suffixes in this group
                 df = repo_data.get(tag_suffix);
                 if df is not None and dwic_col in df.columns:
                     valid_y = df[dwic_col].dropna();
                     if not valid_y.empty:
                         current_group_y_vals.extend(valid_y.values)

            # Determine the y-range for label spreading within _plot_single_chunk
            # Use a default range if no valid data, or calculate from min/max of actual data within fixed plot_ylim
            min_y_actual = np.min(current_group_y_vals) if current_group_y_vals else plot_ylim[0]
            max_y_actual = np.max(current_group_y_vals) if current_group_y_vals else plot_ylim[1]
            # Ensure the range is within the fixed plot limits
            min_y_clipped = max(min_y_actual, plot_ylim[0])
            max_y_clipped = min(max_y_actual, plot_ylim[1])
            current_group_y_range = max_y_clipped - min_y_clipped if max_y_clipped > min_y_clipped else (plot_ylim[1] - plot_ylim[0]) # Default to full plot range if data range is zero or invalid


            plot_group_info[(account, repo_from_filename)] = {"plot_rows": repo_plot_rows, "y_range": current_group_y_range, "num_chunks": num_repo_chunks}
            total_plot_rows += repo_plot_rows # Add rows for this group to the total
            print(f"Group ('{account}', '{repo_from_filename}'): {len(repo_keys)} tags, {num_repo_chunks} chunks, {repo_plot_rows} plot rows, estimated data y-range ~{current_group_y_range:.2f}")


    if total_plot_rows == 0:
        print("--- Plotting Aborted: No valid data groups found for plotting. ---"); return

    # Create the figure with the calculated total number of rows
    # Each row will have 4 columns: Scatter (Chunk 1), KDE (Chunk 1), Scatter (Chunk 2), KDE (Chunk 2)
    fig, axes = plt.subplots(nrows=total_plot_rows, ncols=4, figsize=(20, total_plot_rows * 5), dpi=120, squeeze=False)
    print(f"\n--- Creating Plot Figure ({total_plot_rows} rows x 4 columns) ---")

    current_row_offset = 0 # Keep track of the current row position in the figure
    # Iterate through the plot_group_info in a sorted manner for consistent layout
    # Sorting by account first, then repo_from_filename
    sorted_plot_groups = sorted(plot_group_info.keys())

    for (account, repo_from_filename) in sorted_plot_groups:
        info = plot_group_info[(account, repo_from_filename)]
        repo_plot_rows, current_group_y_range, num_repo_chunks = info["plot_rows"], info["y_range"], info["num_chunks"]

        # Get the actual data for this group {tag_suffix: df}
        # Need to check if the account and repo still exist in case of unforeseen issues
        if account not in all_data_by_account_repo or repo_from_filename not in all_data_by_account_repo[account]["data"]:
             print(f"Error: Data for group ('{account}', '{repo_from_filename}') not found during plotting phase. Skipping.")
             continue

        repo_data_dict = all_data_by_account_repo[account]["data"][repo_from_filename]
        repo_keys = sorted(repo_data_dict.keys()) # Ensure keys (tag_suffixes) are sorted
        account_chunk_size = all_data_by_account_repo[account]["chunk_size"] # Get the chunk size for the account

        if repo_plot_rows == 0:
            print(f"Skipping plotting for group ('{account}', '{repo_from_filename}') as it requires 0 rows."); continue
        print(f"\n--- Plotting Group: ('{account}', '{repo_from_filename}') ({repo_plot_rows} rows assigned) ---")

        # Chunk the tag_suffixes for this repo group based on the account's chunk size
        repo_chunks = [repo_keys[i:i + account_chunk_size] for i in range(0, len(repo_keys), account_chunk_size)]

        # Iterate through the chunks for this group and plot them in pairs of axes (scatter/kde)
        for repo_plot_row_idx in range(repo_plot_rows):
            fig_row_idx = current_row_offset + repo_plot_row_idx # The absolute row index in the figure
            chunk_idx1, chunk_idx2 = repo_plot_row_idx * 2, repo_plot_row_idx * 2 + 1 # Indices of chunks for the left and right plots in this row

            # Plot left column (Chunk 1)
            if chunk_idx1 < num_repo_chunks:
                keys1 = repo_chunks[chunk_idx1] # Get the tag_suffixes for this chunk
                ax_scatter1, ax_kde1 = axes[fig_row_idx, 0], axes[fig_row_idx, 1]
                print(f"  Plotting Row {fig_row_idx+1}, Column 1 (Group ('{account}', '{repo_from_filename}'), Chunk {chunk_idx1+1}, Tags: {len(keys1)})")
                # Call _plot_single_chunk with the tags, the group's data dictionary, and the group's y-range
                _plot_single_chunk(ax_scatter1, ax_kde1, keys1, repo_data_dict, current_group_y_range)
                # Set titles for the scatter and KDE plots
                ax_scatter1.set_title(f"{account}/{repo_from_filename} - Scatter (Chunk {chunk_idx1+1})", fontsize=10)
                ax_kde1.set_title(f"{account}/{repo_from_filename} - KDE & Labels (Chunk {chunk_idx1+1})", fontsize=10)
            else: # Hide axes if no chunk for this position in the grid
                 axes[fig_row_idx, 0].set_visible(False)
                 axes[fig_row_idx, 1].set_visible(False)


            # Plot right column (Chunk 2)
            if chunk_idx2 < num_repo_chunks:
                keys2 = repo_chunks[chunk_idx2] # Get the tag_suffixes for this chunk
                ax_scatter2, ax_kde2 = axes[fig_row_idx, 2], axes[fig_row_idx, 3]
                print(f"  Plotting Row {fig_row_idx+1}, Column 2 (Group ('{account}', '{repo_from_filename}'), Chunk {chunk_idx2+1}, Tags: {len(keys2)})")
                 # Call _plot_single_chunk with the tags, the group's data dictionary, and the group's y-range
                _plot_single_chunk(ax_scatter2, ax_kde2, keys2, repo_data_dict, current_group_y_range)
                 # Set titles for the scatter and KDE plots
                ax_scatter2.set_title(f"{account}/{repo_from_filename} - Scatter (Chunk {chunk_idx2+1})", fontsize=10)
                ax_kde2.set_title(f"{account}/{repo_from_filename} - KDE & Labels (Chunk {chunk_idx2+1})", fontsize=10)
            else: # Hide axes if no chunk for this position in the grid
                 axes[fig_row_idx, 2].set_visible(False)
                 axes[fig_row_idx, 3].set_visible(False)

        # Move the row offset down by the number of rows used by this group
        current_row_offset += repo_plot_rows

    # Adjust layout to prevent titles/labels overlapping and save the figure
    plt.tight_layout(rect=[0.02, 0.02, 0.95, 0.97], h_pad=3, w_pad=3)
    plt.savefig('qc_plots.png')
    print("\n--- QC plots saved to 'qc_plots.png' ---")


# --- Example Usage ---
# Define the accounts to plot and the desired chunk size (number of tags per plot panel) for each account.
# The chunk size applies to the number of tag_suffixes plotted together within an (account, repo) group chunk.
accounts_to_plot = {
    "data-hcp": 4, # Plot max 4 tags per panel for data-hcp groups (e.g., data-hcp/lifespan)
    "data-abcd": 11, # Plot max 5 tags per panel for data-abcd groups (e.g., data-abcd/release3)
    "data-openneuro": 10, # Plot max 10 tags per panel for data-openneuro groups (e.g., data-openneuro/ds001)
    "data-indi": 10,
    "data-others": 7
}

plot_multi_account_releases(accounts_to_plot)
