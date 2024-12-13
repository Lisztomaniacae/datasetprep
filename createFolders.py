import os
from PIL import Image


def main():
    MACHINED_PARTS_FOLDER = "machined_parts"
    GRIPPER_IMAGE_FOLDER = "gripper"
    GRIPPER_IMAGE_PATH = "./gripper/gripper.png"
    DATASET_FOLDER = "dataset"
    if not os.path.exists(MACHINED_PARTS_FOLDER):
        os.makedirs(MACHINED_PARTS_FOLDER)
        print("machined_parts directory was successfully created")
    if not os.path.exists(GRIPPER_IMAGE_FOLDER):
        os.makedirs(GRIPPER_IMAGE_FOLDER)
        print("gripper directory was successfully created")
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
        print("dataset directory was successfully created")
    if not os.path.exists(GRIPPER_IMAGE_PATH):
        image = Image.new("RGB", (1, 1))
        image.save(GRIPPER_IMAGE_PATH, "PNG")


if __name__ == "__main__":
    main()
