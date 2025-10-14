#!/usr/bin/env python3
"""
Convert COCO format annotations to CSV format required by patcher.py

COCO format: bbox = [x, y, width, height]
Patcher format: CSV with columns: images,x_min,y_min,x_max,y_max,labels
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def coco_to_patcher_csv(coco_json_path, output_csv_path):
    """
    Convert COCO annotations to patcher CSV format.

    Args:
        coco_json_path: Path to COCO JSON file
        output_csv_path: Path to output CSV file
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Create image_id to filename mapping
    image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

    # Create category_id to name mapping
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Create category name to integer label mapping (0 to n)
    unique_categories = sorted(set(category_id_to_name.values()))
    category_name_to_label = {name: idx for idx, name in enumerate(unique_categories)}

    # Convert annotations
    rows = []
    for ann in coco_data['annotations']:
        image_name = image_id_to_name[ann['image_id']]
        bbox = ann['bbox']  # [x, y, width, height]
        category_name = category_id_to_name[ann['category_id']]
        label_id = category_name_to_label[category_name]

        # Convert to x_min, y_min, x_max, y_max
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[0] + bbox[2]
        y_max = bbox[1] + bbox[3]

        rows.append({
            'images': image_name,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'labels': label_id,
            'class_names': category_name
        })

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    print(f"Converted {len(rows)} annotations from {len(image_id_to_name)} images")
    print(f"Categories mapping:")
    for name, label_id in category_name_to_label.items():
        print(f"  {label_id}: {name}")
    print(f"Output saved to: {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert COCO format annotations to patcher CSV format'
    )
    parser.add_argument(
        'coco_json',
        type=str,
        help='Path to COCO JSON file'
    )
    parser.add_argument(
        'output_csv',
        type=str,
        help='Path to output CSV file'
    )

    args = parser.parse_args()

    coco_to_patcher_csv(args.coco_json, args.output_csv)


if __name__ == '__main__':
    main()
