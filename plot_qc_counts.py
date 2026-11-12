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

# Merge numbered repository parts (e.g., brain2 into brain) when the base exists.
for key in list(grouped_data):
    account, repo = key.split('_', 1)
    base = f"{account}_{repo.rstrip('0123456789').rstrip('_')}"
    if base != key and base in grouped_data:
        grouped_data[base].extend(grouped_data.pop(key))

# --- Prepare Aligned Data for Plotting ---
def draw_counts(ax, grouped_data, palette='tab20c'):
    sorted_groups = sorted(grouped_data.items())
    labels_bg = [key for key, group in sorted_groups]
    counts_bg = [sum(item['count'] for item in group) for key, group in sorted_groups]
    colors_bg = sns.color_palette(palette, len(counts_bg))

    counts_fg, labels_fg = [], []
    for key, group in sorted_groups:
        for item in sorted(group, key=lambda x: x['label']):
            counts_fg.append(item['count'])
            labels_fg.append(item['label'])

    # Adjusted Y-limit slightly to ensure bottom text is visible
    ax.set(ylim=(1.4, 2.5)); ax.axis("off") 

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
            
            # --- ORIGINAL LINE DRAWING (UNCHANGED) ---
            ax.plot([x_pos_bg, x_pos_bg], [y_pos, y_pos2], "k", lw=1)
            ax.plot([x_pos_bg + ct, x_pos_bg + ct], [y_pos, y_pos2], "k", lw=1)
            ax.plot([x_pos_bg, x_pos_bg + ct], [y_pos, y_pos], "k", lw=1)
            
            sample_name = lbl.replace('data-hcp_', '').replace('data-', '').replace('_','\n')
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
    
    # Set minimum distance to 5.5% of total width
    min_dist = grand_total_bg * 0.055 
    
    upper_row = resolve_overlaps(upper_row, min_dist)
    lower_row = resolve_overlaps(lower_row, min_dist)
    
    for node in upper_row + lower_row:
        ax.text(node['x'], node['y'], node['text'], ha="center", va="center", fontsize=12)
        
        # Draw connecting line if moved significantly
        if abs(node['x'] - node['orig_x']) > (grand_total_bg * 0.01):
             ax.plot([node['orig_x'], node['x']], 
                     [2 + (0.3 if node['y']>2 else -0.3), node['y']], 
                     color='black', lw=0.5, alpha=0.3, linestyle=':')

    # Moved Total Samples text to the bottom center (X=midpoint, Y=1.45)
    ax.text(grand_total_bg / 2, 1.35, f"Total samples: {grand_total_bg}", 
            ha="center", va="top", fontsize=12, fontweight="bold")

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
        



def draw_compact_counts(ax, grouped_data):
    groups = sorted(grouped_data.items())
    total = sum(item['count'] for _, entries in groups for item in entries)
    if not total:
        ax.axis('off')
        return

    x, labels = 0, []
    for (key, entries), color in zip(groups, sns.color_palette('husl', len(groups))):
        width = sum(item['count'] for item in entries)
        for j, item in enumerate(sorted(entries, key=lambda v: v['label'])):
            ax.barh(0, item['count'], left=x, height=0.24,
                    color=adjust_lightness(color, 1 + (j % 4) * 0.04),
                    edgecolor='white', linewidth=0.2)
            x += item['count']
        if width >= total * 0.10:
            labels.append((width, key, (x - width / 2) / total))

    positions = []
    for count, key, pos in sorted(labels, reverse=True):
        if len(positions) == 3:
            break
        if any(abs(pos - prev) < 0.20 for prev in positions):
            continue
        positions.append(pos)
        ax.text(pos, 0.78, f"{key.replace('data-', '').replace('_', '/')}\n(n={count:,})",
                ha='center', va='center', fontsize=11, fontweight='medium',
                transform=ax.transAxes)
        ax.plot([pos, pos], [0.60, 0.48], color='black', lw=0.6,
                transform=ax.transAxes)

    ax.text(0.5, 0.07, f"Total samples: {total:,}", ha='center',
            va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.set(xlim=(0, total), ylim=(-0.5, 0.8))
    ax.axis('off')


def group_type(key):
    account, repo = key.split('_', 1)
    if account == 'data-dandi':
        return 2
    if repo in ('lifespan', 'brain', 'brain2'):
        return 0
    if repo in ('disease', 'disease2', 'tcia'):
        return 1
    return 2


if grouped_data:
    fig, ax = plt.subplots(figsize=(24, 3.5), dpi=100)
    draw_counts(ax, grouped_data)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', pad_inches=0.1)
    print(f"Chart saved to {OUTPUT_FILENAME}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 6), dpi=100)
    names = ('Normal human', 'Human patient', 'Animal / phantom / other')
    for i, (ax, name) in enumerate(zip(axes, names)):
        subset = {key: value for key, value in grouped_data.items() if group_type(key) == i}
        if subset:
            draw_compact_counts(ax, subset)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No QC datasets', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(name, fontsize=16, fontweight='bold', pad=3)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.05, hspace=0.38)
    fig.savefig('qc_counts_by_type.png', dpi=100)
    print("Chart saved to qc_counts_by_type.png")
