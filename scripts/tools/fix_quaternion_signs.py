#!/usr/bin/env python3
# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to fix quaternion sign inconsistencies in HDF5 demonstration datasets.

This script processes an HDF5 dataset and ensures all quaternions have consistent
signs by:
1. Ensuring w component is non-negative (quat_unique)
2. Ensuring temporal consistency (consecutive frames have same hemisphere)

Usage:
    python fix_quaternion_signs.py --input <input.hdf5> --output <output.hdf5>
"""

import argparse
import h5py
import numpy as np
from pathlib import Path


def quat_unique(q: np.ndarray) -> np.ndarray:
    """Ensure quaternion has non-negative w component.
    
    Args:
        q: Quaternions of shape (..., 4) in (w, x, y, z) format.
        
    Returns:
        Quaternions with w >= 0.
    """
    sign = np.sign(q[..., 0:1])
    sign = np.where(sign == 0, 1, sign)  # Handle exact zero
    return q * sign


def ensure_temporal_consistency(quats: np.ndarray) -> np.ndarray:
    """Ensure consecutive quaternions are on the same hemisphere.
    
    This prevents sudden 180-degree flips in the quaternion representation
    by ensuring each quaternion has positive dot product with the previous one.
    
    Args:
        quats: Quaternions of shape (T, 4) where T is the number of timesteps.
        
    Returns:
        Quaternions with temporal consistency.
    """
    result = quats.copy()
    
    # First frame: use quat_unique (positive w)
    result[0] = quat_unique(result[0:1])[0]
    
    # For subsequent frames, ensure positive dot product with previous
    for t in range(1, len(result)):
        dot = np.dot(result[t], result[t-1])
        if dot < 0:
            result[t] = -result[t]
    
    return result


def count_discontinuities(quats: np.ndarray) -> int:
    """Count actual quaternion discontinuities (negative dot products).
    
    A discontinuity is when consecutive quaternions have negative dot product,
    meaning they represent the same rotation but with opposite signs.
    This is different from smooth transitions through w=0.
    
    Args:
        quats: Quaternions of shape (T, 4).
        
    Returns:
        Number of discontinuities.
    """
    if len(quats) < 2:
        return 0
    
    dots = np.sum(quats[:-1] * quats[1:], axis=1)
    return np.sum(dots < 0)


def fix_quaternions_in_array(data: np.ndarray, quat_indices: list) -> tuple:
    """Fix quaternions at specified indices in an array.
    
    Args:
        data: Array of shape (T, D) containing quaternions at specified indices.
        quat_indices: List of (start_idx, end_idx) tuples for quaternion locations.
        
    Returns:
        Tuple of (fixed_data, num_flips_fixed).
    """
    result = data.copy()
    total_flips = 0
    
    for start_idx, end_idx in quat_indices:
        quats = result[:, start_idx:end_idx].copy()
        
        # Count original discontinuities (negative dot products)
        original_discontinuities = count_discontinuities(quats)
        
        # Fix quaternions
        fixed_quats = ensure_temporal_consistency(quats)
        result[:, start_idx:end_idx] = fixed_quats
        
        # Count remaining discontinuities (should be 0)
        remaining_discontinuities = count_discontinuities(fixed_quats)
        
        total_flips += original_discontinuities - remaining_discontinuities
    
    return result, total_flips


def process_dataset(input_path: str, output_path: str, dry_run: bool = False):
    """Process HDF5 dataset to fix quaternion signs.
    
    Args:
        input_path: Path to input HDF5 file.
        output_path: Path to output HDF5 file.
        dry_run: If True, only analyze without modifying.
    """
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"Dry run: {dry_run}")
    print()
    
    # Action quaternion indices: [left_pos(3), left_quat(4), right_pos(3), right_quat(4), hand(22)]
    action_quat_indices = [
        (3, 7),    # left quaternion
        (10, 14),  # right quaternion
    ]
    
    total_action_flips_fixed = 0
    total_obs_left_flips_fixed = 0
    total_obs_right_flips_fixed = 0
    total_demos = 0
    
    with h5py.File(input_path, 'r') as f_in:
        if dry_run:
            f_out = None
        else:
            f_out = h5py.File(output_path, 'w')
        
        # Copy top-level attributes
        if f_out is not None:
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value
        
        data_group_in = f_in['data']
        if f_out is not None:
            data_group_out = f_out.create_group('data')
            for attr_name, attr_value in data_group_in.attrs.items():
                data_group_out.attrs[attr_name] = attr_value
        
        demo_names = sorted(data_group_in.keys(), key=lambda x: int(x.split('_')[1]))
        total_demos = len(demo_names)
        
        print(f"Processing {total_demos} demonstrations...")
        
        for i, demo_name in enumerate(demo_names):
            demo_in = data_group_in[demo_name]
            
            if f_out is not None:
                demo_out = data_group_out.create_group(demo_name)
                # Copy attributes
                for attr_name, attr_value in demo_in.attrs.items():
                    demo_out.attrs[attr_name] = attr_value
            
            # Process actions
            actions = demo_in['actions'][:]
            fixed_actions, action_flips = fix_quaternions_in_array(actions, action_quat_indices)
            total_action_flips_fixed += action_flips
            
            # Process observation quaternions
            obs_left_quat = demo_in['obs/left_eef_quat'][:]
            fixed_obs_left, obs_left_flips = fix_quaternions_in_array(
                obs_left_quat.reshape(-1, 4), [(0, 4)]
            )
            fixed_obs_left = fixed_obs_left.reshape(obs_left_quat.shape)
            total_obs_left_flips_fixed += obs_left_flips
            
            obs_right_quat = demo_in['obs/right_eef_quat'][:]
            fixed_obs_right, obs_right_flips = fix_quaternions_in_array(
                obs_right_quat.reshape(-1, 4), [(0, 4)]
            )
            fixed_obs_right = fixed_obs_right.reshape(obs_right_quat.shape)
            total_obs_right_flips_fixed += obs_right_flips
            
            # Also fix obs/actions (which mirrors actions)
            obs_actions = demo_in['obs/actions'][:]
            fixed_obs_actions, _ = fix_quaternions_in_array(obs_actions, action_quat_indices)
            
            # Also fix processed_actions if it has the same format
            processed_actions = demo_in['processed_actions'][:]
            # processed_actions might be joint positions, check if it has quaternions
            # For now, skip this as it's the output of IK, not input quaternions
            
            if f_out is not None:
                # Write fixed data
                demo_out.create_dataset('actions', data=fixed_actions, compression='gzip')
                demo_out.create_dataset('processed_actions', data=processed_actions, compression='gzip')
                
                # Create obs group
                obs_out = demo_out.create_group('obs')
                obs_out.create_dataset('actions', data=fixed_obs_actions, compression='gzip')
                obs_out.create_dataset('left_eef_quat', data=fixed_obs_left, compression='gzip')
                obs_out.create_dataset('right_eef_quat', data=fixed_obs_right, compression='gzip')
                
                # Copy other obs datasets unchanged
                for key in demo_in['obs'].keys():
                    if key not in ['actions', 'left_eef_quat', 'right_eef_quat']:
                        demo_out.copy(demo_in[f'obs/{key}'], obs_out, key)
                
                # Copy other groups unchanged
                for key in demo_in.keys():
                    if key not in ['actions', 'processed_actions', 'obs']:
                        demo_out.copy(demo_in[key], demo_out, key)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{total_demos} demos...")
        
        if f_out is not None:
            f_out.close()
    
    print()
    print("=" * 50)
    print("Summary:")
    print("=" * 50)
    print(f"Total demos processed: {total_demos}")
    print(f"Action quaternion discontinuities fixed: {total_action_flips_fixed}")
    print(f"Obs left quaternion discontinuities fixed: {total_obs_left_flips_fixed}")
    print(f"Obs right quaternion discontinuities fixed: {total_obs_right_flips_fixed}")
    total_fixed = total_action_flips_fixed + total_obs_left_flips_fixed + total_obs_right_flips_fixed
    print(f"Total discontinuities fixed: {total_fixed}")
    
    if not dry_run:
        print(f"\nOutput saved to: {output_path}")


def verify_fix(file_path: str):
    """Verify that the fixed file has no quaternion discontinuities."""
    print(f"\nVerifying: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        data_group = f['data']
        
        total_action_left_disc = 0
        total_action_right_disc = 0
        total_obs_left_disc = 0
        total_obs_right_disc = 0
        
        for demo_name in list(data_group.keys())[:10]:
            demo = data_group[demo_name]
            actions = demo['actions'][:]
            
            left_quat = actions[:, 3:7]
            right_quat = actions[:, 10:14]
            obs_left_quat = demo['obs/left_eef_quat'][:]
            obs_right_quat = demo['obs/right_eef_quat'][:]
            
            # Count discontinuities (negative dot products)
            total_action_left_disc += count_discontinuities(left_quat)
            total_action_right_disc += count_discontinuities(right_quat)
            total_obs_left_disc += count_discontinuities(obs_left_quat)
            total_obs_right_disc += count_discontinuities(obs_right_quat)
        
        print(f"Verification (first 10 demos) - counting actual discontinuities:")
        print(f"  Action left discontinuities: {total_action_left_disc} (should be 0)")
        print(f"  Action right discontinuities: {total_action_right_disc} (should be 0)")
        print(f"  Obs left discontinuities: {total_obs_left_disc} (should be 0)")
        print(f"  Obs right discontinuities: {total_obs_right_disc} (should be 0)")
        
        total = total_action_left_disc + total_action_right_disc + total_obs_left_disc + total_obs_right_disc
        if total == 0:
            print("\n  SUCCESS: All quaternion discontinuities have been fixed!")
        else:
            print(f"\n  WARNING: {total} discontinuities remain!")


def main():
    parser = argparse.ArgumentParser(
        description="Fix quaternion sign inconsistencies in HDF5 demonstration datasets."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input HDF5 file."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output HDF5 file."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only analyze, don't create output file."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the output file after processing."
    )
    
    args = parser.parse_args()
    
    # Validate paths
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path = Path(args.output)
    if output_path.exists() and not args.dry_run:
        response = input(f"Output file exists: {output_path}\nOverwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Process the dataset
    process_dataset(str(input_path), str(output_path), dry_run=args.dry_run)
    
    # Verify if requested
    if args.verify and not args.dry_run:
        verify_fix(str(output_path))


if __name__ == "__main__":
    main()

