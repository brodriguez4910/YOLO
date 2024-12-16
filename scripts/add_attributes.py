from svw_torch.database import SVWClient
import re
from datetime import datetime
import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons


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


def add_attributes(dataset_name: str, attribute: str):
    client = SVWClient("pipeline")
    # Filter ground_truth_labels collection by dataset name
    ground_truth_labels = client.collection("ground_truth_labels")
    images_collection = client.collection("images")

    labels = list(ground_truth_labels.find({"dataset": dataset_name}))

    image_ids = set([label["image_id"] for label in labels])

    # Get image information from the images collection
    images = list(images_collection.find({"_id": {"$in": list(image_ids)}}))
    images = [
        image
        for image in images
        if is_date_greater_than(image["filename"], datetime(2024, 5, 6, 0, 0, 0, 0))
    ]
    image_ids = set([image["_id"] for image in images])
    labels = [label for label in labels if label["image_id"] in image_ids]

    # shuffle images
    # np.random.shuffle(images)

    for image in tqdm(images):
        # Get the labels for the current image
        image_labels = [label for label in labels if label["image_id"] == image["_id"]]

        if any([len(label["boxes"]) > 0 for label in image_labels]):
            continue

        # Get the image path
        image_path = image["filepath"]

        # load the image with matplotlib and add a check box if the attribute should be added
        # to the document
        image_data = plt.imread(image_path)
        fig, ax = plt.subplots()
        ax.imshow(image_data)
        plt.axis("off")

        # add check box for the attribute
        rax = plt.axes([0.05, 0.4, 0.1, 0.15])
        check = CheckButtons(rax, [attribute], [False])

        def func(label):
            attributes = image.get("attributes", [])
            if attribute in attributes:
                return
            attributes.append(attribute)
            images_collection.update_one(
                {"_id": image["_id"]}, {"$set": {"attributes": attributes}}
            )

        check.on_clicked(func)

        plt.show()
        plt.close()


if __name__ == "__main__":
    add_attributes("test_fm_boxes_blueberries", "negative_blueberries_gold_standard")
