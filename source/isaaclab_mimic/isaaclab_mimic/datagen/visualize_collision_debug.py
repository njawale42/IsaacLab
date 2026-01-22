#!/usr/bin/env python3
"""
Visualize collision debug data saved by the scheduling module.

Usage:
    # First, run data generation with debug enabled:
    COLLISION_DEBUG=1 python your_datagen_script.py
    
    # Then visualize:
    python -m isaaclab_mimic.datagen.visualize_collision_debug /tmp/collision_debug.pkl
    
    # Or with custom options:
    python -m isaaclab_mimic.datagen.visualize_collision_debug /tmp/collision_debug.pkl --sample 5 --show-links
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


def load_debug_data(path: str) -> dict:
    """Load collision debug data from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def print_summary(data: dict) -> None:
    """Print summary of collision debug data."""
    print("\n" + "=" * 60)
    print("COLLISION DEBUG SUMMARY")
    print("=" * 60)
    print(f"Collision margin: {data['collision_margin']:.4f} m")
    print(f"Densify factor: {data['densify_factor']}")
    print(f"Right trajectory points: {data['Nr']}")
    print(f"Left trajectory points: {data['Nl']}")
    print(f"Total pairs checked: {data['Nr'] * data['Nl']}")
    print(f"Sample pairs saved: {len(data['sample_pairs'])}")
    
    if data['min_distances']:
        min_d = min(data['min_distances'])
        max_d = max(data['min_distances'])
        avg_d = sum(data['min_distances']) / len(data['min_distances'])
        print(f"\nPenetration depths (negative = collision):")
        print(f"  Min: {min_d:.4f} m")
        print(f"  Max: {max_d:.4f} m")
        print(f"  Avg: {avg_d:.4f} m")
        
        colliding = sum(1 for d in data['min_distances'] if d <= 0)
        print(f"\nColliding samples: {colliding}/{len(data['min_distances'])}")
    
    # Show link names
    print(f"\nRight arm links ({len(data['idx_to_name_r'])} total):")
    for idx, name in sorted(data['idx_to_name_r'].items())[:10]:
        spheres_for_link = (data['link_idx_map_r'] == idx).sum()
        print(f"  [{idx}] {name}: {spheres_for_link} spheres")
    if len(data['idx_to_name_r']) > 10:
        print(f"  ... and {len(data['idx_to_name_r']) - 10} more")
    
    print(f"\nLeft arm links ({len(data['idx_to_name_l'])} total):")
    for idx, name in sorted(data['idx_to_name_l'].items())[:10]:
        spheres_for_link = (data['link_idx_map_l'] == idx).sum()
        print(f"  [{idx}] {name}: {spheres_for_link} spheres")
    if len(data['idx_to_name_l']) > 10:
        print(f"  ... and {len(data['idx_to_name_l']) - 10} more")


def visualize_sample_matplotlib(data: dict, sample_idx: int = 0, show_links: bool = False) -> None:
    """Visualize a sample pair using matplotlib 3D."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    if sample_idx >= len(data['sample_pairs']):
        print(f"Sample index {sample_idx} out of range (max: {len(data['sample_pairs']) - 1})")
        return
    
    sample = data['sample_pairs'][sample_idx]
    spheres_r = sample['spheres_r']  # [N, 4] - x, y, z, radius
    spheres_l = sample['spheres_l']  # [M, 4]
    
    # Filter to valid spheres (radius > 0)
    valid_r = spheres_r[:, 3] > 0
    valid_l = spheres_l[:, 3] > 0
    spheres_r = spheres_r[valid_r]
    spheres_l = spheres_l[valid_l]
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw right arm spheres (red)
    if len(spheres_r) > 0:
        ax.scatter(spheres_r[:, 0], spheres_r[:, 1], spheres_r[:, 2],
                   s=spheres_r[:, 3] * 5000, c='red', alpha=0.4, label='Right arm')

    # Draw left arm spheres (blue)
    if len(spheres_l) > 0:
        ax.scatter(spheres_l[:, 0], spheres_l[:, 1], spheres_l[:, 2],
                   s=spheres_l[:, 3] * 5000, c='blue', alpha=0.4, label='Left arm')

    # Find closest pair
    if len(spheres_r) > 0 and len(spheres_l) > 0:
        min_dist = float('inf')
        min_pair = None
        for sph_r in spheres_r:
            for sph_l in spheres_l:
                dist = np.linalg.norm(sph_r[:3] - sph_l[:3]) - sph_r[3] - sph_l[3] - data['collision_margin']
                if dist < min_dist:
                    min_dist = dist
                    min_pair = (sph_r, sph_l)

        if min_pair is not None:
            closest_r, closest_l = min_pair
            ax.plot([closest_r[0], closest_l[0]], [closest_r[1], closest_l[1]], [closest_r[2], closest_l[2]],
                    'g-', linewidth=2, label=f'Closest pair (d={min_dist:.4f}m)')
            ax.scatter([closest_r[0], closest_l[0]], [closest_r[1], closest_l[1]], [closest_r[2], closest_l[2]],
                       s=100, c='green', marker='x')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    title = f"Sample {sample_idx}: R[{sample['r_idx']}] vs L[{sample['l_idx']}]"
    title += f"\nMin penetration: {sample['min_penetration']:.4f}m"
    title += f" ({'COLLISION' if sample['is_colliding'] else 'no collision'})"
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = 0
    for spheres in [spheres_r, spheres_l]:
        if len(spheres) > 0:
            for i in range(3):
                max_range = max(max_range, spheres[:, i].max() - spheres[:, i].min())
    
    if max_range > 0:
        mid_x = (spheres_r[:, 0].mean() + spheres_l[:, 0].mean()) / 2 if len(spheres_r) > 0 and len(spheres_l) > 0 else 0
        mid_y = (spheres_r[:, 1].mean() + spheres_l[:, 1].mean()) / 2 if len(spheres_r) > 0 and len(spheres_l) > 0 else 0
        mid_z = (spheres_r[:, 2].mean() + spheres_l[:, 2].mean()) / 2 if len(spheres_r) > 0 and len(spheres_l) > 0 else 0
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    plt.tight_layout()
    plt.show()


def visualize_distance_histogram(data: dict) -> None:
    """Show histogram of minimum distances across samples."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return
    
    if not data['min_distances']:
        print("No distance data available")
        return
    
    distances = np.array(data['min_distances'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    bins = 50
    n, bins_edges, patches = ax.hist(distances, bins=bins, edgecolor='black', alpha=0.7)
    
    # Color bars based on collision
    for i, patch in enumerate(patches):
        if bins_edges[i] <= 0:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Collision threshold')
    ax.axvline(x=data['collision_margin'], color='orange', linestyle=':', linewidth=2, 
               label=f"Margin ({data['collision_margin']:.3f}m)")
    
    ax.set_xlabel('Minimum penetration depth (m)\n(negative = collision)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Closest Approach Distances')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def visualize_trajectory_distances(data: dict) -> None:
    """Show distance over trajectory progression."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if not data['sample_pairs']:
        print("No sample data available")
        return

    r_indices = [s['r_idx'] for s in data['sample_pairs']]
    l_indices = [s['l_idx'] for s in data['sample_pairs']]
    distances = [s['min_penetration'] for s in data['sample_pairs']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Distance vs right arm progress
    ax1 = axes[0]
    colors = ['red' if d <= 0 else 'green' for d in distances]
    ax1.scatter(r_indices, distances, c=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Right arm trajectory index')
    ax1.set_ylabel('Min penetration (m)')
    ax1.set_title('Distance vs Right Arm Progress')

    # Right: Distance vs left arm progress
    ax2 = axes[1]
    ax2.scatter(l_indices, distances, c=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Left arm trajectory index')
    ax2.set_ylabel('Min penetration (m)')
    ax2.set_title('Distance vs Left Arm Progress')

    plt.tight_layout()
    plt.show()


def animate_trajectory(data: dict, interval: int = 500) -> None:
    """Animate the trajectory showing spheres moving through time."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.widgets import Slider, Button
    except ImportError:
        print("matplotlib not available")
        return

    if not data['sample_pairs']:
        print("No sample data available")
        return

    samples = data['sample_pairs']
    n_samples = len(samples)

    # Compute global bounds for consistent view
    all_spheres = []
    for s in samples:
        sr = s['spheres_r']
        sl = s['spheres_l']
        valid_r = sr[:, 3] > 0
        valid_l = sl[:, 3] > 0
        if valid_r.any():
            all_spheres.append(sr[valid_r, :3])
        if valid_l.any():
            all_spheres.append(sl[valid_l, :3])

    if not all_spheres:
        print("No valid spheres found")
        return

    all_pts = np.vstack(all_spheres)
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.6

    # Create figure with slider
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.2)

    # Slider axis
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))
    slider = Slider(ax_slider, 'Frame', 0, n_samples - 1, valinit=0, valstep=1)

    # Play button
    ax_play = plt.axes((0.85, 0.05, 0.1, 0.03))
    btn_play = Button(ax_play, 'Play')

    playing = [False]
    anim = [None]

    def draw_frame(idx):
        ax.clear()
        sample = samples[int(idx)]
        spheres_r = sample['spheres_r']
        spheres_l = sample['spheres_l']

        valid_r = spheres_r[:, 3] > 0
        valid_l = spheres_l[:, 3] > 0
        sr = spheres_r[valid_r]
        sl = spheres_l[valid_l]

        # Draw spheres
        if len(sr) > 0:
            ax.scatter(sr[:, 0], sr[:, 1], sr[:, 2],
                       s=sr[:, 3] * 5000, c='red', alpha=0.4, label='Right arm')
        if len(sl) > 0:
            ax.scatter(sl[:, 0], sl[:, 1], sl[:, 2],
                       s=sl[:, 3] * 5000, c='blue', alpha=0.4, label='Left arm')

        # Find and draw closest pair
        if len(sr) > 0 and len(sl) > 0:
            min_dist = float('inf')
            min_pair = None
            for sph_r in sr:
                for sph_l in sl:
                    dist = np.linalg.norm(sph_r[:3] - sph_l[:3]) - sph_r[3] - sph_l[3]
                    if dist < min_dist:
                        min_dist = dist
                        min_pair = (sph_r, sph_l)

            if min_pair is not None:
                closest_r, closest_l = min_pair
                color = 'red' if min_dist <= 0 else 'green'
                ax.plot([closest_r[0], closest_l[0]], [closest_r[1], closest_l[1]], [closest_r[2], closest_l[2]],
                        color=color, linewidth=3, label=f'd={min_dist:.3f}m')
                ax.scatter([closest_r[0], closest_l[0]], [closest_r[1], closest_l[1]], [closest_r[2], closest_l[2]],
                           s=150, c=color, marker='x', linewidths=3)

        # Set bounds
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend(loc='upper right')

        status = 'COLLISION' if sample['is_colliding'] else 'clear'
        ax.set_title(f"Frame {int(idx)+1}/{n_samples}: R[{sample['r_idx']}] vs L[{sample['l_idx']}]\n"
                     f"Min distance: {sample['min_penetration']:.4f}m ({status})")

        fig.canvas.draw_idle()

    def on_slider_change(val):
        draw_frame(val)

    def animate_step(frame):
        new_val = (slider.val + 1) % n_samples
        slider.set_val(new_val)
        return []

    def on_play(event):
        if playing[0]:
            playing[0] = False
            btn_play.label.set_text('Play')
            if anim[0] is not None:
                anim[0].event_source.stop()
        else:
            playing[0] = True
            btn_play.label.set_text('Pause')
            anim[0] = FuncAnimation(fig, animate_step, interval=interval, blit=True, cache_frame_data=False)
            plt.draw()

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(on_play)

    draw_frame(0)
    plt.show()


def visualize_all_samples_grid(data: dict, cols: int = 4) -> None:
    """Show all samples in a grid view for quick overview."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if not data['sample_pairs']:
        print("No sample data available")
        return

    samples = data['sample_pairs']
    n = len(samples)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             subplot_kw={'projection': '3d'})
    axes = np.array(axes).flatten()

    for i, sample in enumerate(samples):
        ax = axes[i]
        spheres_r = sample['spheres_r']
        spheres_l = sample['spheres_l']

        valid_r = spheres_r[:, 3] > 0
        valid_l = spheres_l[:, 3] > 0
        sr = spheres_r[valid_r]
        sl = spheres_l[valid_l]

        if len(sr) > 0:
            ax.scatter(sr[:, 0], sr[:, 1], sr[:, 2],
                       s=sr[:, 3] * 2000, c='red', alpha=0.5)
        if len(sl) > 0:
            ax.scatter(sl[:, 0], sl[:, 1], sl[:, 2],
                       s=sl[:, 3] * 2000, c='blue', alpha=0.5)

        status = 'COL' if sample['is_colliding'] else 'ok'
        ax.set_title(f"[{i}] d={sample['min_penetration']:.3f} ({status})", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Hide unused axes
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize collision debug data")
    parser.add_argument("path", type=str, help="Path to collision_debug.pkl file")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize (default: 0)")
    parser.add_argument("--show-links", action="store_true", help="Show link names in visualization")
    parser.add_argument("--histogram", action="store_true", help="Show distance histogram")
    parser.add_argument("--trajectory", action="store_true", help="Show trajectory distance plot")
    parser.add_argument("--animate", action="store_true", help="Animate trajectory with slider/play controls")
    parser.add_argument("--grid", action="store_true", help="Show all samples in a grid")
    parser.add_argument("--all", action="store_true", help="Show all visualizations")
    parser.add_argument("--interval", type=int, default=500, help="Animation interval in ms (default: 500)")
    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"File not found: {args.path}")
        sys.exit(1)

    data = load_debug_data(args.path)
    print_summary(data)

    if args.animate:
        animate_trajectory(data, interval=args.interval)
        return

    if args.grid:
        visualize_all_samples_grid(data)
        return

    if args.all or args.histogram:
        visualize_distance_histogram(data)

    if args.all or args.trajectory:
        visualize_trajectory_distances(data)

    if args.all or (not args.histogram and not args.trajectory):
        visualize_sample_matplotlib(data, args.sample, args.show_links)


if __name__ == "__main__":
    main()

