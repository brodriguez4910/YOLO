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


def calculate_iou(polygon_coords, bbox_coords):
    # Create Polygon and Bounding Box using Shapely
    polygon = Polygon(polygon_coords)
    bbox = box(*bbox_coords)  # Unpack bounding box coordinates (minx, miny, maxx, maxy)

    # Calculate the intersection
    intersection = polygon.intersection(bbox)

    # Calculate the union
    union = polygon.union(bbox)

    # Calculate the areas
    intersection_area = intersection.area
    union_area = union.area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def save_split_to_txts(images, split_name, annotations_map):
    os.makedirs(split_name, exist_ok=True)

    for image in images:
        image_id = str(image["_id"])
        annotations = annotations_map.get(image_id, [])
        with open(f"{split_name}/{image_id}.txt", "w") as f:
            for annotation in annotations:
                # convert box to x_center, y_center, width, height
                x_center = annotation["bbox"][0] + annotation["bbox"][2] / 2
                y_center = annotation["bbox"][1] + annotation["bbox"][3] / 2
                width = annotation["bbox"][2]
                height = annotation["bbox"][3]
                f.write(
                    f"{annotation['category_id']} {x_center} {y_center} {width} {height}\n"
                )


def save_split_images(images, split_name):
    os.makedirs(split_name, exist_ok=True)

    for image in tqdm(images):
        shutil.copy(image["filepath"], f"{split_name}/{str(image['_id'])}.png")


def extract_polygons(json_data, dataset):
    # Navigate to consensus_label -> instances
    instances = (
        json_data.get("consensus_label", {}).get(f"{dataset}", {}).get("instances", {})
    )

    # Dictionary to store polygons
    extracted_polygons = {}

    # Iterate over each category
    for category, items in instances.items():
        for item in items:
            polygons = item.get("polygons")
            if polygons:
                if category not in extracted_polygons:
                    extracted_polygons[category] = []
                extracted_polygons[category].extend(polygons)

    return extracted_polygons


def extract_date_from_filename(filename):
    """
    Extract the year, month, and day from a filename with the format YYYY-MM-DD-HH-MM-SS.mmmmmm.png.

    Args:
    filename (str): The filename to extract the date from.

    Returns:
    datetime: A datetime object representing the extracted date.
    """
    # Define the regex pattern to extract the date components
    pattern = r"(\d{4})-(\d{2})-(\d{2})"

    # Extract the components using the regex pattern
    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Filename does not match the expected format")

    year, month, day = match.groups()

    # Create a datetime object from the extracted components
    extracted_date = datetime(int(year), int(month), int(day))

    return extracted_date


def is_date_greater_than(filename, compare_date):
    """
    Check if the date extracted from the filename is less than the compare_date.

    Args:
    filename (str): The filename to extract the date from.
    compare_date (datetime): The date to compare against.

    Returns:
    bool: True if the extracted date is less than the compare_date, False otherwise.
    """
    extracted_date = extract_date_from_filename(filename)
    return extracted_date > compare_date


def create_coco_dataset(
    database_name,
    dataset_name,
    segmentation_dataset_name=None,
    iou_threshold=0.9,
    img_size=(2048, 2448),
    drop_classes=[],
    flter: dict = {},
):
    client = SVWClient(database=database_name)

    # Filter ground_truth_labels collection by dataset name
    ground_truth_labels = client.collection("ground_truth_labels")
    images_collection = client.collection("images")
    instance_seg_labels = client.collection("instance_segmentation_labels")

    labels = list(ground_truth_labels.find({"dataset": dataset_name}))
    image_ids = set([label["image_id"] for label in labels])

    segmentations = list(
        instance_seg_labels.find({"image_object_id": {"$in": list(image_ids)}})
    )

    segmentations = {
        segmentation["image_object_id"]: segmentation
        for segmentation in segmentations
        if extract_polygons(segmentation, segmentation_dataset_name)
    }

    # Get image information from the images collection
    images = list(
        images_collection.find(
            {"_id": {"$in": list(image_ids)}, "attributes": {"$nin": ["throwout"]}}
        )
    )
    images = [
        image
        for image in images
        if is_date_greater_than(image["filename"], datetime(2024, 5, 6, 0, 0, 0, 0))
    ]
    image_ids = set([image["_id"] for image in images])
    labels = [label for label in labels if label["image_id"] in image_ids]

    # Create COCO dataset structure
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

    # Create categories set to avoid duplicate entries
    categories_set = set()

    # Map to store annotations grouped by image_id
    annotations_map = defaultdict(list)

    annotation_id = 1

    # group labels by image_id
    import itertools

    labels = sorted(labels, key=lambda x: x["image_id"])
    labels = [
        {"image_id": k, "classes": [v for v in g]}
        for k, g in itertools.groupby(labels, key=lambda x: x["image_id"])
    ]

    # build a dictionary of class names and their corresponding coco category ids
    class_to_category_id = {}
    for label in labels:
        classes = label["classes"]
        for class_label in classes:
            category_name = class_label["class"]
            if (
                category_name not in class_to_category_id
                and category_name not in drop_classes
            ):
                category_id = len(coco_dataset["categories"])
                class_to_category_id[category_name] = category_id
                coco_category = {"id": category_id, "name": category_name}
                coco_dataset["categories"].append(coco_category)

    # remove images without all categories and not in the drop_classes list
    # labels = [
    #     label
    #     for label in labels
    #     if len(
    #         set(
    #             class_to_category_id[class_label["class"]]
    #             for class_label in label["classes"]
    #             if class_label["class"] not in drop_classes
    #         )
    #     )
    #     == len(class_to_category_id)
    # ]

    # filter images based on the labels
    image_ids = set([label["image_id"] for label in labels])
    images = [image for image in images if image["_id"] in image_ids]

    # Add images to the COCO dataset
    for image in images:
        coco_image = {
            "id": str(image["_id"]),
            "file_name": image["filepath"],
            "width": img_size[
                1
            ],  # You may want to include width and height if available
            "height": img_size[
                0
            ],  # You may want to include width and height if available
        }
        coco_dataset["images"].append(coco_image)

    for label in tqdm(labels):
        image_id = str(label["image_id"])
        classes = label["classes"]
        segmentation = segmentations.get(label["image_id"], {})

        # iterate through each class and add the bounding box annotation
        for class_label in classes:
            category_name = class_label["class"]

            if category_name in drop_classes:
                continue

            category_id = class_to_category_id[category_name]

            # Add bounding box annotation
            for box in class_label.get("boxes", []):
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [box["x"], box["y"], box["w"], box["h"]],
                    "area": box["w"] * box["h"],
                    "segmentation": [],
                    "iscrowd": 0,
                }
                # Search for matching polygon for the segmentation
                matched_segmentation = None
                max_iou = 0
                ious = []
                if segmentation:
                    extracted = extract_polygons(
                        segmentation, segmentation_dataset_name
                    )
                    for name, polygons in extracted.items():
                        for polygon in polygons:
                            # Calculate IoU to match polygon with the bounding box
                            iou = calculate_iou(
                                polygon,
                                [
                                    box["x"] * img_size[1],
                                    box["y"] * img_size[0],
                                    (box["x"] + box["w"]) * img_size[1],
                                    (box["y"] + box["h"]) * img_size[0],
                                ],
                            )
                            ious.append(iou)

                        # Update the matched segmentation if the IoU is the highest so far
                        if iou > max_iou and iou >= iou_threshold:
                            max_iou = iou
                            matched_segmentation = (
                                polygon  # Store the polygon with the highest IoU
                            )

                # If a matching segmentation was found, add it to the annotation
                if matched_segmentation:
                    coco_annotation["segmentation"] = matched_segmentation

                annotations_map[image_id].append(coco_annotation)
                annotation_id += 1

    # Add annotations to the COCO dataset
    for image_id in image_ids:
        coco_dataset["annotations"].extend(annotations_map[str(image_id)])

    # # remove images with no annotations
    # images = [
    #     image
    #     for image in images
    #     if str(image["_id"]) in annotations_map and annotations_map[str(image["_id"])]
    # ]

    # display image with segmentations
    # for image in images:
    #     annotations = annotations_map.get(str(image["_id"]), [])
    #     if not annotations:
    #         continue

    #     for annotation in annotations:
    #         if not annotation["segmentation"]:
    #             continue

    #         img = cv2.imread(image["filepath"])
    #         x, y, w, h = annotation["bbox"]
    #         # scale to image size
    #         x, y, w, h = (
    #             x * img_size[1],
    #             y * img_size[0],
    #             w * img_size[1],
    #             h * img_size[0],
    #         )
    #         cv2.rectangle(
    #             img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
    #         )

    #         for polygon in annotation["segmentation"]:
    #             polygon = np.array(polygon, np.int32)
    #             polygon = polygon.reshape((-1, 1, 2))
    #             cv2.polylines(img, [polygon], True, (0, 0, 255), 2)
    #         cv2.imwrite(f"{image['filename']}_annotated.png", img)

    # Save COCO dataset to file
    with open(f"{dataset_name}_coco.json", "w") as f:
        json.dump(coco_dataset, f, indent=4)

    print(f"COCO dataset created successfully: {dataset_name}_coco.json")
    if "test" in dataset_name:
        save_split_images(images, f"{dataset_name}/images/test")
        save_split_to_txts(images, f"{dataset_name}/labels/test", annotations_map)
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

        # save_split_images(train_images, f"{dataset_name}/images/train")
        # save_split_images(val_images, f"{dataset_name}/images/val")

        # save_split_to_txts(
        #     train_images, f"{dataset_name}/labels/train", annotations_map
        # )
        # save_split_to_txts(val_images, f"{dataset_name}/labels/val", annotations_map)

        # Filter images for train and validation sets
        train_images = [
            image for image in coco_dataset["images"] if image["id"] in train_image_ids
        ]
        val_images = [
            image for image in coco_dataset["images"] if image["id"] in val_image_ids
        ]

        # Filter annotations for train and validation sets
        train_annotations = [
            annotation
            for annotation in coco_dataset["annotations"]
            if annotation["image_id"] in train_image_ids
        ]
        val_annotations = [
            annotation
            for annotation in coco_dataset["annotations"]
            if annotation["image_id"] in val_image_ids
        ]

        # Create train and validation COCO datasets
        train_coco_dataset = {
            "info": coco_dataset.get("info", {}),
            "licenses": coco_dataset.get("licenses", []),
            "categories": coco_dataset["categories"],
            "images": train_images,
            "annotations": train_annotations,
        }

        val_coco_dataset = {
            "info": coco_dataset.get("info", {}),
            "licenses": coco_dataset.get("licenses", []),
            "categories": coco_dataset["categories"],
            "images": val_images,
            "annotations": val_annotations,
        }

        # Save the train and validation COCO JSON files
        with open(f"instances_train_coco.json", "w") as train_file:
            json.dump(train_coco_dataset, train_file, indent=4)

        with open(f"instances_val_coco.json", "w") as val_file:
            json.dump(val_coco_dataset, val_file, indent=4)

        print("Train and validation COCO datasets created successfully.")


# filter:
#   filename:
#     $not:
#       $regex: "2024-05-05" convert to datetime
# Example usage
from datetime import datetime

create_coco_dataset(
    database_name="pipeline",
    dataset_name="fm_boxes_blueberries",
    segmentation_dataset_name="fm_segmentation_blueberries",
    iou_threshold=0.5,
    img_size=(2048, 2448),
    drop_classes=["maybe_fm", "hand_sleeve"],
)
