#!/usr/bin/env python3
import os, io, requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from collections import defaultdict
import colorsys

# --- Configuration ---
# token, headers = os.getenv('PAT_TOKEN'), {'Authorization': f'token {os.getenv("PAT_TOKEN")}'} 
RELEASE_URL = "https://api.github.com/repos/frankyeh/FiberDataHub/releases/tags/qc-data"
OUTPUT_FILENAME = 'qc_counts_combined.png'

# --- Helper Functions ---
def adjust_lightness(color, amount=1.3):
    """Adjusts the lightness of an RGB color. >1 is lighter, <1 is darker."""
    try:
        r, g, b, *a = color
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = max(0, min(1, l * amount)) 
        new_rgb = colorsys.hls_to_rgb(h, l, s)
        return (*new_rgb, *a) if a else new_rgb
    except (TypeError, ValueError):
        return color 

def get_base_color(x_pos, color_map):
    for segment in color_map:
        if x_pos < segment['end']:
            return segment['color']
    return color_map[-1]['color'] if color_map else "black" 

def resolve_overlaps(nodes, min_dist):
    """
    Adjusts x-coordinates to ensure nodes are at least min_dist apart.
    """
    nodes.sort(key=lambda k: k['x'])
    # INCREASED ITERATIONS: Run 50 times to ensure they push far enough apart
    for _ in range(50): 
        stable = True
        for i in range(len(nodes) - 1):
            a = nodes[i]
            b = nodes[i+1]
            diff = b['x'] - a['x']
            if diff < min_dist:
                stable = False
                move = (min_dist - diff) / 2
                a['x'] -= move
                b['x'] += move
        if stable: break
    return nodes

# --- Data Fetching and Processing ---
grouped_data = defaultdict(list)

try:
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
    key = f"{parts[0]}_{parts[1]}" 

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

# --- Plotting ---
if grouped_data:
    sorted_groups = sorted(grouped_data.items())
    labels_bg = [key for key, group in sorted_groups]
    counts_bg = [sum(item['count'] for item in group) for key, group in sorted_groups]
    colors_bg = sns.color_palette("tab20c", len(counts_bg))

    counts_fg, labels_fg = [], []
    for key, group in sorted_groups:
        for item in sorted(group, key=lambda x: x['label']):
            counts_fg.append(item['count'])
            labels_fg.append(item['label'])

    fig, ax = plt.subplots(figsize=(24, 3), dpi=100)
    ax.set(ylim=(1.5, 2.5)); ax.axis("off")

    # 1. Draw BACKGROUND bars and create the color map
    x_pos_bg, grand_total_bg = 0, sum(counts_bg)
    color_map = []
    label_toggle = False

    text_candidates = []

    for ct, col, lbl in zip(counts_bg, colors_bg, labels_bg):
        color_map.append({'end': x_pos_bg + ct, 'color': col})
        
        if ct > 100:
            center_x = x_pos_bg + ct / 2
            y_offset = 0.45
            y_txt, is_top = (2 + y_offset, False) if label_toggle else (2 - y_offset, True)
            label_toggle = not label_toggle
            y_pos, y_pos2 = y_txt + (2 - y_txt) * 0.3, y_txt + (2 - y_txt) * 0.4
            
            # --- ORIGINAL DRAWING ---
            ax.plot([x_pos_bg, x_pos_bg], [y_pos, y_pos2], "k", lw=1)
            ax.plot([x_pos_bg + ct, x_pos_bg + ct], [y_pos, y_pos2], "k", lw=1)
            ax.plot([x_pos_bg, x_pos_bg + ct], [y_pos, y_pos], "k", lw=1)
            
            sample_name = lbl.replace('data-', '').replace('_','\n')
            text = f"{sample_name}\n(n={ct})" if is_top else f"(n={ct})\n{sample_name}"
            
            text_candidates.append({
                'x': center_x,
                'y': y_txt,
                'text': text,
                'orig_x': center_x
            })
            
        x_pos_bg += ct
    
    # --- ADJUST TEXT POSITIONS ---
    upper_row = [n for n in text_candidates if n['y'] > 2]
    lower_row = [n for n in text_candidates if n['y'] < 2]
    
    # INCREASED DISTANCE: 9% of total width (was 5%)
    min_dist = grand_total_bg * 0.09 
    
    upper_row = resolve_overlaps(upper_row, min_dist)
    lower_row = resolve_overlaps(lower_row, min_dist)
    
    for node in upper_row + lower_row:
        ax.text(node['x'], node['y'], node['text'], ha="center", va="center", fontsize=12)
        
        # Draw connecting line if moved significantly
        if abs(node['x'] - node['orig_x']) > (grand_total_bg * 0.01):
             ax.plot([node['orig_x'], node['x']], 
                     [2 + (0.3 if node['y']>2 else -0.3), node['y']], 
                     color='black', lw=0.5, alpha=0.3, linestyle=':')

    # Moved Total Label slightly further right to avoid collision with widely spread text
    ax.text(x_pos_bg + max(counts_bg, default=0) * 0.05, 2, f"Total samples: {grand_total_bg}", ha="left", va="center", fontsize=10, fontweight="bold")

    # 2. Draw FOREGROUND bars
    x_pos_fg = 0
    last_base_color = None
    sub_item_counter = 0

    for ct, lbl in zip(counts_fg, labels_fg):
        center_x_fg = x_pos_fg + ct / 2
        base_color = get_base_color(center_x_fg, color_map)

        if base_color != last_base_color:
            sub_item_counter = 0
            last_base_color = base_color

        variation = (sub_item_counter % 6) * 0.03
        amount = 1.0 + variation
        new_color = adjust_lightness(base_color, amount=amount)
        
        ax.barh(2, ct, left=x_pos_fg, height=0.4, color=new_color, edgecolor=adjust_lightness(base_color, amount=0.9))
        
        if ct > 100:
            sample_parts = lbl.split('_')
            sample_name = sample_parts[-1] if len(sample_parts) > 1 else lbl.split('-')[-1]
            ax.text(center_x_fg, 2.0, sample_name, ha="center", va="center", fontsize=8, rotation=90)
            
        x_pos_fg += ct
        sub_item_counter += 1
        
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', pad_inches=0.1)
    print(f"Chart saved to {OUTPUT_FILENAME}")
