#!/usr/bin/env python3
import os, io, requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from collections import defaultdict
import colorsys

# --- Configuration ---
# token, headers = os.getenv('PAT_TOKEN'), {'Authorization': f'token {os.getenv("PAT_TOKEN")}'} # Get token & setup headers
RELEASE_URL = "https://api.github.com/repos/frankyeh/FiberDataHub/releases/tags/qc-data"
OUTPUT_FILENAME = 'qc_counts_combined.png'

# --- Helper Functions ---
def adjust_lightness(color, amount=1.3):
    """Adjusts the lightness of an RGB color. >1 is lighter, <1 is darker."""
    try:
        r, g, b, *a = color
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, min(1, l * amount)) # Adjust and clamp lightness to the [0, 1] range
        new_rgb = colorsys.hls_to_rgb(h, l, s)
        return (*new_rgb, *a) if a else new_rgb
    except (TypeError, ValueError):
        return color # Fallback in case of an error

def get_base_color(x_pos, color_map):
    """Finds the background color from the map based on an x-position."""
    for segment in color_map:
        if x_pos < segment['end']:
            return segment['color']
    return color_map[-1]['color'] if color_map else "black" # Fallback for the last point

# --- Data Fetching and Processing ---
# Group original data by the merge key to ensure alignment
grouped_data = defaultdict(list)

try:
    #    resp = requests.get(RELEASE_URL, headers=headers); resp.raise_for_status()
    resp = requests.get(RELEASE_URL); resp.raise_for_status()
    assets = resp.json().get('assets', [])
except requests.exceptions.RequestException as e:
    print(f"Error fetching {RELEASE_URL}: {e}"); assets = []

for asset in assets:
    name = asset.get('name')
    if not name or not name.endswith('_qc.tsv'): continue

    base_name = name.replace('_qc.tsv', '')
    parts = base_name.split('_')
    if len(parts) < 2: continue
    key = f"{parts[0]}_{parts[1]}" # e.g., "data-hcp_lifespan"

    try:
        r = requests.get(asset['browser_download_url']); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep='\t')
        if 'neighboring DWI correlation(masked)' in df.columns:
            df = df[df['neighboring DWI correlation(masked)'] <= 1]
        if 'DWI contrast' in df.columns:
            df = df[df['DWI contrast'] != 1]
        
        if not df.empty:
            grouped_data[key].append({'count': len(df), 'label': base_name})
    except Exception as e:
        print(f"Error processing {name}: {e}")

# --- Prepare Aligned Data for Plotting ---
if grouped_data:
    # Sort groups by key (e.g., 'data-hcp_disease') for consistent order
    sorted_groups = sorted(grouped_data.items())

    # Data for BACKGROUND bars (merged groups)
    labels_bg = [key for key, group in sorted_groups]
    counts_bg = [sum(item['count'] for item in group) for key, group in sorted_groups]
    colors_bg = sns.color_palette("tab20c", len(counts_bg))

    # Data for FOREGROUND bars (original files, aligned with background)
    counts_fg, labels_fg = [], []
    for key, group in sorted_groups:
        for item in sorted(group, key=lambda x: x['label']): # Sort within group for consistency
            counts_fg.append(item['count'])
            labels_fg.append(item['label'])

# --- Plotting ---
if grouped_data:
    # Sort and Prep Data (Same as before)
    sorted_groups = sorted(grouped_data.items())
    labels_bg = [key for key, group in sorted_groups]
    counts_bg = [sum(item['count'] for item in group) for key, group in sorted_groups]
    colors_bg = sns.color_palette("tab20c", len(counts_bg))
    
    counts_fg, labels_fg = [], []
    for key, group in sorted_groups:
        for item in sorted(group, key=lambda x: x['label']):
            counts_fg.append(item['count'])
            labels_fg.append(item['label'])

    grand_total_bg = sum(counts_bg)

    # Increase figure height and ylim to accommodate staggered labels
    fig, ax = plt.subplots(figsize=(24, 5), dpi=100) 
    ax.set(ylim=(0, 4)); ax.axis("off") # Y-limits expanded

    # 1. Draw BACKGROUND bars (Groups) with STAGGERING
    x_pos_bg = 0
    color_map = []
    
    # Staggering Logic
    # We cycle through 3 distinct height levels to avoid overlap
    for i, (ct, col, lbl) in enumerate(zip(counts_bg, colors_bg, labels_bg)):
        color_map.append({'end': x_pos_bg + ct, 'color': col})
        
        # Only label if the group isn't microscopic (> 0.1% of total)
        if (ct / grand_total_bg) > 0.001:
            center_x = x_pos_bg + ct / 2
            
            # Logic: Alternate Top/Bottom, then cycle through 3 height levels (0, 1, 2)
            is_top = (i % 2 == 0)
            level = (i // 2) % 3  # Groups items into levels: [0,0], [1,1], [2,2]...
            
            # Calculate Y positions
            base_y = 2
            height_step = 0.6
            dist_from_bar = 0.5 + (level * height_step)
            
            y_txt = base_y + dist_from_bar if is_top else base_y - dist_from_bar
            y_conn = base_y + (0.2 if is_top else -0.2) # Small connector nub near bar

            # Draw Bracket
            # Vertical lines
            ax.plot([x_pos_bg, x_pos_bg], [y_conn, y_txt], "k", lw=0.5, alpha=0.4)
            ax.plot([x_pos_bg + ct, x_pos_bg + ct], [y_conn, y_txt], "k", lw=0.5, alpha=0.4)
            # Horizontal cap at text level
            ax.plot([x_pos_bg, x_pos_bg + ct], [y_txt, y_txt], "k", lw=0.5, alpha=0.4)

            # Format Text
            sample_name = lbl.replace('data-hcp_', '').replace('data-', '').replace('_','\n')
            txt_str = f"{sample_name}\n(n={ct})" if is_top else f"(n={ct})\n{sample_name}"
            
            # Determine Text Anchor position (slightly above/below the bracket line)
            y_text_final = y_txt + 0.15 if is_top else y_txt - 0.15
            
            ax.text(center_x, y_text_final, txt_str, ha="center", va="center", 
                    fontsize=11, linespacing=0.9)

        x_pos_bg += ct

    # Total count label on the far right
    ax.text(x_pos_bg * 1.01, 2, f"Total: {grand_total_bg}", va="center", fontsize=12, fontweight="bold")

    # 2. Draw FOREGROUND bars (Sub-items)
    x_pos_fg = 0
    last_base_color = None
    sub_item_counter = 0

    for ct, lbl in zip(counts_fg, labels_fg):
        center_x_fg = x_pos_fg + ct / 2
        base_color = get_base_color(center_x_fg, color_map)

        if base_color != last_base_color:
            sub_item_counter = 0
            last_base_color = base_color

        amount = 1.0 + ((sub_item_counter % 6) * 0.03)
        new_color = adjust_lightness(base_color, amount=amount)
        
        ax.barh(2, ct, left=x_pos_fg, height=0.4, color=new_color, 
                edgecolor=adjust_lightness(base_color, amount=0.9), linewidth=0.5)
        
        # Only label sub-items if wide enough (>0.2%)
        if (ct / grand_total_bg) > 0.002:
            parts = lbl.split('_')
            s_name = parts[-1] if len(parts) > 1 else lbl.split('-')[-1]
            ax.text(center_x_fg, 2.0, s_name, ha="center", va="center", 
                    fontsize=7, rotation=90, color='white' if sum(base_color[:3])<1.5 else 'black')
            
        x_pos_fg += ct
        sub_item_counter += 1
        
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', pad_inches=0.1)
    print(f"Chart saved to {OUTPUT_FILENAME}")
