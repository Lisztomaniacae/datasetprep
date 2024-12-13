import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import os
import csv
import matplotlib.pyplot as plt


def overlay_images(base_img, overlay_img, x, y, alpha):
    # Ensure the overlay image is in RGBA mode (with transparency)
    overlay_img = overlay_img.convert("RGBA")

    # Rotate the overlay image by the specified angle (alpha in radians)
    overlay_img = overlay_img.rotate(np.degrees(-alpha), resample=Image.Resampling.BICUBIC, expand=True)

    # Get the dimensions of the rotated overlay image
    overlay_w, overlay_h = overlay_img.size

    # Get the dimensions of the base image
    base_w, base_h = base_img.size

    # Compute the position (center the overlay on (x, y))
    top_left_x = int(base_w // 2 - overlay_w // 2 + x)
    top_left_y = int(base_h // 2 - overlay_h // 2 + y)

    # Create a transparent background (RGBA) canvas
    canvas = Image.new("RGBA", base_img.size, (0, 0, 0, 0))  # Transparent background

    # Paste the base image onto the canvas
    canvas.paste(base_img.convert("RGBA"), (0, 0))

    # Paste the overlay image onto the canvas at the calculated position
    # The mask ensures that the transparency (alpha) is respected
    canvas.paste(overlay_img, (top_left_x, top_left_y), overlay_img)

    # Return the result (optional: you can return as "RGB" to remove transparency)
    return canvas


def load_parameters(csv_file):
    parameters = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x = int(row['x'])
            y = int(row['y'])
            alpha = float(row['alpha'])
    return x, y, alpha


def test(detail):
    test_folder = ("dataset/" + str(detail))
    for folder in sorted(os.listdir(test_folder)):
        #if not folder.startswith("normal"):
        #    continue

        folder_path = os.path.join(test_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        parameters_path = os.path.join(folder_path, "parameters.csv")

        machined_part_path = os.path.join(folder_path, "machined_part.png")
        gripper_path = os.path.join(folder_path, "gripper.png")

        if (not os.path.exists(machined_part_path) or not os.path.exists(gripper_path)
                or not os.path.exists(parameters_path)):
            print(f"Missing files in {folder_path}. Skipping...")
            continue

        # Load images using PIL
        machined_part = Image.open(machined_part_path)
        gripper = Image.open(gripper_path)

        x, y, alpha = load_parameters(parameters_path)

        # Overlay the gripper on the machined part
        overlaid_image = overlay_images(machined_part, gripper, x, y, alpha)

        # Display the result using matplotlib
        plt.title(folder_path + f" {x}, {y}, {alpha}")
        plt.imshow(overlaid_image)
        plt.axis('off')  # Hide the axes
        plt.show()

if(__name__ == "__main__"):
    test(15)
