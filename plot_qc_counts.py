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
def draw_counts(ax, grouped_data):
    sorted_groups = sorted(grouped_data.items())
    labels_bg = [key for key, group in sorted_groups]
    counts_bg = [sum(item['count'] for item in group) for key, group in sorted_groups]
    colors_bg = sns.color_palette("tab20c", len(counts_bg))

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
        


def group_type(key):
    account, repo = key.split('_', 1)
    if account == 'data-dandi':
        return 2
    if repo in ('lifespan', 'brain', 'brain2'):
        return 0
    if repo in ('disease', 'disease2', 'tcia'):
        return 1
    return 2


# --- Dataset-type chart (website hero) ---
# one row per dataset type on a common scale; each segment is one repository group,
# so bar lengths compare the types truthfully and segments show where the samples live
TYPE_NAMES = ('Healthy human', 'Patient / disease', 'Animal / phantom / other')
TYPE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a')  # categorical, validated for color-vision deficiency
INK, MUTED = '#16181d', '#5b616e'


def type_rows(grouped_data):
    """[(type name, color, total, [(repository label, count) largest first])] for the three dataset types"""
    rows = []
    for i, (name, color) in enumerate(zip(TYPE_NAMES, TYPE_COLORS)):
        repos = [(key.replace('data-', '').replace('_', '/'), sum(item['count'] for item in group))
                 for key, group in grouped_data.items() if group_type(key) == i]
        repos.sort(key=lambda r: -r[1])
        rows.append((name, color, sum(c for _, c in repos), repos))
    return rows


def draw_type_bar(ax, y, color, repos, scale, height, label_min):
    """one bar of repository segments separated by surface gaps; returns the repositories too narrow to label"""
    x, unlabeled = 0, []
    for label, count in repos:
        ax.barh(y, count, left=x, height=height, color=color, edgecolor='white', linewidth=1.5)
        if count >= label_min * scale:
            ax.text(x + scale * 0.004, y + height / 2 + 0.06, f"{label}\n{count:,}", ha='left', va='bottom',
                    fontsize=11, color=INK, linespacing=1.25)
        else:
            unlabeled.append((label, count))
        x += count
    return x, unlabeled


def save_type_hero(rows, total, stamp):
    """wide composition for desktop pages"""
    fig = plt.figure(figsize=(12, 4.8), dpi=200)
    fig.patch.set_facecolor('white')
    fig.text(0.02, 0.95, 'QC-passed diffusion MRI samples by dataset type', fontsize=13, fontweight='bold', color=INK, va='top')
    fig.text(0.98, 0.95, f'Live count from the qc-data release · {stamp}', fontsize=10, color=MUTED, ha='right', va='top')
    ax = fig.add_axes([0.25, 0.13, 0.73, 0.72])
    ax.axis('off')
    scale = max(r[2] for r in rows)
    ax.set(xlim=(0, scale), ylim=(-0.5, len(rows) - 0.25))
    for i, (name, color, row_total, repos) in enumerate(rows):
        y = len(rows) - 1 - i
        fy = 0.13 + 0.72 * (y + 0.5) / (len(rows) + 0.25)  # row center in figure coordinates
        fig.text(0.02, fy + 0.005, f"{row_total:,}", fontsize=24, fontweight='bold', color=INK, va='bottom')
        fig.patches.append(plt.Rectangle((0.02, fy - 0.052), 0.011, 0.0275, color=color, transform=fig.transFigure))
        fig.text(0.036, fy - 0.005, name, fontsize=12, color=MUTED, va='top')
        end, rest = draw_type_bar(ax, y - 0.1, color, repos, scale, 0.34, 0.07)
        if rest:
            ax.text(end + scale * 0.008, y - 0.1, '  ·  '.join(f"{label} {count:,}" for label, count in rest),
                    ha='left', va='center', fontsize=10, color=MUTED)
    fig.text(0.02, 0.03, f"Total samples: {total:,} QC-passed scans in {sum(len(r[3]) for r in rows)} repository groups. "
             "Counts change as datasets are added and quality-controlled.", fontsize=10, color=MUTED)
    fig.savefig('qc_counts_by_type.png', facecolor='white')
    fig.savefig('qc_counts_by_type.svg', facecolor='white')
    plt.close(fig)


def save_type_mobile(rows, total, stamp):
    """narrow composition for phones: the same rows stacked, largest repositories listed under each bar"""
    fig = plt.figure(figsize=(4.2, 7.4), dpi=200)
    fig.patch.set_facecolor('white')
    fig.text(0.06, 0.975, 'QC-passed samples\nby dataset type', fontsize=13, fontweight='bold', color=INK, va='top', linespacing=1.2)
    scale = max(r[2] for r in rows)
    top, step = 0.875, 0.27
    for i, (name, color, row_total, repos) in enumerate(rows):
        y0 = top - i * step
        fig.text(0.06, y0, f"{row_total:,}", fontsize=20, fontweight='bold', color=INK, va='top')
        fig.text(0.06, y0 - 0.058, name, fontsize=11, color=MUTED, va='top')
        ax = fig.add_axes([0.06, y0 - 0.125, 0.88, 0.04])
        ax.axis('off')
        ax.set(xlim=(0, scale), ylim=(0, 1))
        x = 0
        for _, count in repos:
            ax.barh(0.5, count, left=x, height=1, color=color, edgecolor='white', linewidth=1.2)
            x += count
        shown = repos[:3]
        more = len(repos) - len(shown)
        listing = '\n'.join(f"{label}  {count:,}" for label, count in shown) + (f"\n+{more} more" if more else '')
        fig.text(0.06, y0 - 0.14, listing, fontsize=10, color=MUTED, va='top', linespacing=1.35)
    fig.text(0.06, 0.035, f"Total samples: {total:,}\nLive count · {stamp}", fontsize=9, color=MUTED, va='bottom', linespacing=1.3)
    fig.savefig('qc_counts_by_type_mobile.png', facecolor='white')
    plt.close(fig)


if grouped_data:
    fig, ax = plt.subplots(figsize=(24, 3.5), dpi=100)
    draw_counts(ax, grouped_data)
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, bbox_inches='tight', pad_inches=0.1)
    print(f"Chart saved to {OUTPUT_FILENAME}")

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    rows = type_rows(grouped_data)
    total = sum(r[2] for r in rows)
    assert total == sum(item['count'] for group in grouped_data.values() for item in group)  # every sample has one type
    stamp = pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')
    save_type_hero(rows, total, stamp)
    save_type_mobile(rows, total, stamp)
    print("Charts saved to qc_counts_by_type.png, qc_counts_by_type.svg and qc_counts_by_type_mobile.png")
