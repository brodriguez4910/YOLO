from svw_torch.database import SVWClient
import cv2
import re
from datetime import datetime
import os
import json
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon, box
from tqdm import tqdm
import shutil


# Function to flatten a list of lists (if polygons are in multiple parts)
def flatten_polygon(polygon):
    return [coord for segment in polygon for coord in segment]


# Function to calculate area of polygon (for COCO format)
def calculate_polygon_area(polygon):
    # Flatten polygon if it's a list of lists (multiple contours)
    flat_polygon = (
        flatten_polygon(polygon)
        if any(isinstance(i, list) for i in polygon)
        else polygon
    )

    x_coords = flat_polygon[::2]  # even indices are x
    y_coords = flat_polygon[1::2]  # odd indices are y

    # Area calculation using the Shoelace formula
    return 0.5 * abs(
        sum(
            x_coords[i] * y_coords[i + 1] - x_coords[i + 1] * y_coords[i]
            for i in range(-1, len(x_coords) - 1)
        )
    )


# Function to create COCO-style annotations for instance segmentation
def create_coco_annotations(labels, img_size, drop_classes):
    categories_set = set()
    coco_dataset = {
        "info": {
            "description": "COCO Dataset",
            "version": "1.0",
            "year": 2024,
            "contributor": "",
            "date_created": "",
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    class_to_category_id = {}
    annotations_map = defaultdict(list)
    annotation_id = 1

    for label in labels:
        image_id = str(label["_id"])
        consensus_label = label.get("consensus_label", {})
        instances = consensus_label.get("lemon_instance_segmentation", {}).get(
            "instances", {}
        )

        for category_name, instance_list in instances.items():
            if category_name in drop_classes:
                continue

            # Add the category if it's not already in the dataset
            if category_name not in class_to_category_id:
                category_id = len(coco_dataset["categories"])
                class_to_category_id[category_name] = category_id
                coco_dataset["categories"].append(
                    {"id": category_id, "name": category_name}
                )

            # Process each instance (each containing a list of polygons)
            for instance in instance_list:
                polygons = instance.get("polygons", [])

                for polygon in polygons:
                    area = calculate_polygon_area(polygon)
                    coco_annotation = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_to_category_id[category_name],
                        "segmentation": [polygon],
                        "area": area,
                        "bbox": [],  # Bounding box is not needed for instance segmentation
                        "iscrowd": 0,
                    }

                    annotations_map[image_id].append(coco_annotation)
                    annotation_id += 1

    return coco_dataset, annotations_map


def save_split_images(images, split_name):
    os.makedirs(split_name, exist_ok=True)

    for image in tqdm(images):
        shutil.copy(image["filepath"], f"{split_name}/{str(image['_id'])}.png")


# Main function to create COCO dataset for instance segmentation
def create_coco_dataset(
    database_name, dataset_name, img_size=(2048, 2448), drop_classes=[], min_date=None
):
    client = SVWClient(database=database_name)
    instance_segmentation_labels = client.collection("instance_segmentation_labels")

    # Query instance segmentation labels
    query = {
        "consensus_label.lemon_instance_segmentation": {"$exists": True},
    }
    if min_date:
        query["timestamp"] = {"$gte": min_date}

    labels = list(instance_segmentation_labels.find(query))
    images = [
        {
            "_id": label["_id"],
            "filepath": label["filepath"],
            "width": label["image_width"],
            "height": label["image_height"],
        }
        for label in labels
    ]

    # Create COCO-style annotations
    coco_dataset, annotations_map = create_coco_annotations(
        labels, img_size, drop_classes
    )

    # Add images to COCO dataset
    for image in images:
        coco_image = {
            "id": str(image["_id"]),
            "file_name": image["filepath"],
            "width": image["width"],
            "height": image["height"],
        }
        coco_dataset["images"].append(coco_image)

    # Add annotations to COCO dataset
    for image_id in annotations_map:
        coco_dataset["annotations"].extend(annotations_map[image_id])

    # Save COCO dataset
    with open(f"{dataset_name}_coco.json", "w") as f:
        json.dump(coco_dataset, f, indent=4)

    print(f"COCO dataset created successfully: {dataset_name}_coco.json")

    if "test" in dataset_name:
        save_split_images(images, f"{dataset_name}/images/test")
    else:
        # create train and test splits
        image_ids = [str(image["_id"]) for image in images]
        np.random.shuffle(image_ids)
        train_size = int(0.9 * len(image_ids))
        train_image_ids = image_ids[:train_size]
        val_image_ids = image_ids[train_size:]

        train_images = [
            image for image in images if str(image["_id"]) in train_image_ids
        ]
        val_images = [image for image in images if str(image["_id"]) in val_image_ids]

        save_split_images(train_images, f"{dataset_name}/images/train")
        save_split_images(val_images, f"{dataset_name}/images/val")


# Example usage
min_date = datetime(2023, 5, 6)

create_coco_dataset(
    database_name="pipeline",
    dataset_name="lemon_instance_segmentation",
    img_size=(2048, 2448),
    drop_classes=["outlier", "irrelevant"],
    min_date=None,
)
