#!/usr/bin/env python3
import os, io, requests, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, matplotlib.colors as mcolors, math

token, headers = os.getenv('PAT_TOKEN'), {'Authorization': f'token {os.getenv("PAT_TOKEN")}'} # Get token & setup headers
release_url = "https://api.github.com/repos/frankyeh/FiberDataHub/releases/tags/qc-data"

try:
    resp = requests.get(release_url, headers=headers); resp.raise_for_status()
    assets = resp.json().get('assets', [])
except requests.exceptions.RequestException as e:
    print(f"Error fetching {release_url}: {e}"); assets = []

counts, labels = [], []
for a in assets:
    name = a.get('name')
    if name and name.endswith('_qc.tsv'):
        try:
            r = requests.get(a['browser_download_url'], headers=headers); r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep='\t')
            if 'neighboring DWI correlation(masked)' in df.columns: df = df[df['neighboring DWI correlation(masked)'] <= 1]
            if 'DWI contrast' in df.columns: df = df[df['DWI contrast'] != 1]
            counts.append(len(df)); labels.append(name.replace('_qc.tsv','')) # Append count & label
        except Exception as e: print(f"Error processing {name}: {e}") # Catch any error during download/processing

if counts:
    colors = sns.color_palette("tab20", len(counts))
    accounts = sorted({lbl.split('_')[0] for lbl in labels})

    fig, ax = plt.subplots(figsize=(22, 3), dpi=100)
    ax.set(ylim=(1.5, 2.5)); ax.axis("off") # Set limits & turn off axis
    grand_total, x = sum(counts), 0
    prev_ct, prev_ct2, prev_offset, prev_offset2 = 1000, 1000, 0.25, 0.25 # Init label positioning vars
    label_toggle = False

    for ct, col, lbl in zip(counts, colors, labels):
        ax.barh(2, ct, left=x, height=0.2, color=col, edgecolor="white") # Draw bar
        cx = x + ct / 2

        y_offset = 0.15; y_offset = 0.35 if prev_ct + prev_ct2 < 1000 and prev_offset2 == 0.25 else (0.25 if prev_ct + prev_ct2 < 1000 else y_offset) # Determine y-offset

        if ct > 100: # Draw label for sizeable segments
            y_txt, is_top = (2 + y_offset, False) if label_toggle else (2 - y_offset, True)
            label_toggle = not label_toggle

            delta = 2 - y_txt; ax.plot([cx, cx], [y_txt + delta * 0.3, 2], color="black", lw=1) # Draw pointer line

            sample_parts = lbl.split('_'); sample_name = sample_parts[-1] if len(sample_parts) > 1 else lbl.split('-')[-1] # Simplify name
            text = f"{sample_name}\n(n={ct})" if is_top else f"(n={ct})\n{sample_name}"
            ax.text(cx, y_txt, text, ha="center", va="center", fontsize=8) # Draw label

            prev_offset2, prev_offset, prev_ct2, prev_ct = prev_offset, y_offset, prev_ct, ct # Update positioning vars

        x += ct # Increment x position

    ax.text(x + max(counts) * 0.02, 2, f"Total samples: {grand_total}", ha="left", va="center", fontsize=10, fontweight="bold") # Draw total annotation
    plt.tight_layout();
    plt.savefig('qc_counts.png')
