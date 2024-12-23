import random
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageEnhance
import os
import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

# Define paths for machined parts and gripper images
MACHINED_PARTS_FOLDER = "machined_parts"
GRIPPER_IMAGE_PATH = "./gripper/gripper.png"
DATASET_FOLDER = "dataset"
resize_constant = 32

# Ensure dataset folder exists
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


class ConvolutionalNeuralNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(2, resize_constant // 8, kernel_size=4, stride=1, padding=2)  # Input has 2 channels
        self.bn1 = nn.BatchNorm2d(resize_constant // 8)  # Batch Norm for first conv layer

        self.conv2 = nn.Conv2d(resize_constant // 8, resize_constant // 4, kernel_size=8, stride=1, padding=4)
        self.bn2 = nn.BatchNorm2d(resize_constant // 4)  # Batch Norm for second conv layer

        self.conv3 = nn.Conv2d(resize_constant // 4, resize_constant // 2, kernel_size=16, stride=1, padding=8)
        self.bn3 = nn.BatchNorm2d(resize_constant // 2)  # Batch Norm for third conv layer

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.3)

        # Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        flattened_size = (resize_constant // 8) ** 2 * resize_constant // 2  # Account for pooling
        self.fc1 = nn.Linear(flattened_size, flattened_size // 2)
        self.fc2 = nn.Linear(flattened_size // 2, flattened_size // 4)
        self.fc3 = nn.Linear(flattened_size // 4, flattened_size // 8)
        self.fc4 = nn.Linear(flattened_size // 8, 3)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Apply Kaiming He initialization to convolutional layers
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')

        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 1)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 1)
        if self.conv3.bias is not None:
            nn.init.constant_(self.conv3.bias, 1)

        # Optionally initialize fully connected layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 1)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 1)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 1)

    def forward(self, x):
        # Convolutional Layers with ReLU, Batch Norm, and Pooling
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 + BN1
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 + BN2
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 + BN3
        x = self.pool(x)

        # Flatten for the fully connected layer
        x = torch.flatten(x, start_dim=1)

        # Apply dropout for more randomness
        x = self.dropout(x)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = torch.load('cnn.pth', map_location=device).to(device)
model.eval()


def remove_black_borders_and_track_bias(image):
    """
    Removes black rows and columns from a grayscale image and tracks the x and y biases.

    Args:
        image (numpy.ndarray): Grayscale image to process.

    Returns:
        numpy.ndarray: Cropped image without black borders.
        float: x_bias
        float: y_bias
    """
    x_bias, y_bias = 0, 0
    rows, cols = image.shape

    # Remove rows from the top
    while rows > 0 and np.all(image[0, :] % 255 == 0):
        image = image[1:, :]  # Remove the top row
        y_bias += 0.5
        rows -= 1

    # Remove rows from the bottom
    while rows > 0 and np.all(image[-1, :] % 255 == 0):
        image = image[:-1, :]  # Remove the bottom row
        y_bias -= 0.5
        rows -= 1

    # Remove columns from the left
    while cols > 0 and np.all(image[:, 0] % 255 == 0):
        image = image[:, 1:]  # Remove the left column
        x_bias += 0.5
        cols -= 1

    # Remove columns from the right
    while cols > 0 and np.all(image[:, -1] % 255 == 0):
        image = image[:, :-1]  # Remove the right column
        x_bias -= 0.5
        cols -= 1

    return image, x_bias, y_bias


def pad_and_resize(img, target_pad_size, target_size=(32, 32)):
    """
    Pads the image to the target size times 8 with black pixels and resizes it to target size.
    Args:
        img (numpy.ndarray): Input grayscale image.
        target_size (tuple): Desired size (width, height).
    Returns:
        numpy.ndarray: Resized image.
        :param img:
        :param target_size:
        :param target_pad_size:
    """

    # First pad the image to maintain aspect ratio
    target_pad_width, target_pad_height = target_pad_size
    current_height, current_width = img.shape

    # Calculate padding
    pad_width = max(target_pad_width - current_width, 0)
    pad_height = max(target_pad_height - current_height, 0)

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # Resize the padded image to target size
    resized_img = cv2.resize(padded_img, target_size, interpolation=cv2.INTER_AREA)
    return resized_img


def parse_images_to_input_tensor(machined_part_path, gripper_path):
    # Load images using PIL
    machined_part = Image.open(machined_part_path).convert("RGBA")

    # Convert to numpy array (RGBA)
    machined_part = np.array(machined_part)

    # Replace transparent pixels with black
    # Pixels where alpha channel == 0
    machined_part[machined_part[:, :, 3] == 0] = [0, 0, 0, 255]

    # Convert RGBA to RGB (drop alpha channel)
    machined_part = cv2.cvtColor(machined_part, cv2.COLOR_RGBA2RGB)

    # Convert to grayscale
    machined_part = cv2.cvtColor(machined_part, cv2.COLOR_RGB2GRAY)
    gripper = cv2.imread(gripper_path, cv2.IMREAD_GRAYSCALE)

    # Crop blackspace of machined part
    machined_part, bias_x, bias_y = remove_black_borders_and_track_bias(machined_part)

    width_part, height_part = machined_part.shape[1], machined_part.shape[0]
    width_gripper, height_gripper = gripper.shape[1], gripper.shape[0]

    norm_x = max(width_part, width_gripper)
    norm_y = max(height_part, height_gripper)

    norm_x = max(norm_x, norm_y)
    norm_y = norm_x

    # Resize images with padding
    machined_part = pad_and_resize(machined_part, (norm_x, norm_y))
    gripper = pad_and_resize(gripper, (norm_x, norm_y))

    combined_images = np.stack((machined_part, gripper), axis=0)  # Shape: (2, resize_constant, resize_constant)
    combined_images = combined_images / 255.0  # Normalize pixel values to [0, 1]

    # Combine inputs
    input_tensor = torch.tensor(combined_images, dtype=torch.float32).unsqueeze(0).to(device)

    return input_tensor, bias_x, bias_y, norm_x, norm_y


def denormalize_predictions(pred, x_bias, y_bias, norm_x, norm_y):
    """
    Denormalizes predictions back to the original scale of the images.
    Args:
        pred (torch.Tensor): Predictions from the model (normalized).
        machined_dim (tuple): Dimensions (width, height) of the machined part image.
        gripper_dim (tuple): Dimensions (width, height) of the gripper image.
    Returns:
        tuple: Denormalized (x, y, alpha).
    """
    x = int(np.floor((pred[0].item() - 0.5) * norm_x + x_bias))  # x in original scale
    y = int(np.floor((pred[1].item() - 0.5) * norm_y + y_bias))  # y in original scale
    alpha = (pred[2].item() - 0.5) * np.pi * 2  # alpha back to original range [-pi, pi]
    return x, y, alpha


def enhance_image(image):
    factor = random.uniform(0.5, 2)
    enhancer = ImageEnhance.Color(image)
    enhanced_image = enhancer.enhance(factor)
    factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(factor)

    return enhanced_image


# Helper function to overlay gripper on machined part
def overlay_images(base_image, gripper_image, x, y, alpha):
    base_image = base_image.copy()
    gripper_rotated = gripper_image.rotate(-math.degrees(alpha), expand=True)
    gx, gy = gripper_rotated.size
    paste_position = (x - gx // 2, y - gy // 2)
    base_image.paste(gripper_rotated, paste_position, gripper_rotated)
    return base_image


def mirror_image(img):
    mirrored_img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
    return mirrored_img


def calculate_mirrored_parameters(x, y, alpha):
    new_x = -x
    new_y = y
    new_alpha = (-alpha + 2 * math.pi) % (2 * math.pi) - math.pi

    return new_x, new_y, new_alpha


def inverse_image(img):
    # Ensure the image is in RGBA mode to handle transparency
    img = img.convert("RGBA")

    # Split into individual channels
    r, g, b, a = img.split()

    # Invert RGB channels, keeping alpha channel unchanged
    r = ImageOps.invert(r)
    g = ImageOps.invert(g)
    b = ImageOps.invert(b)

    # Merge the inverted channels back with the original alpha channel
    inverted_img = Image.merge("RGBA", (r, g, b, a))

    return inverted_img


class DatasetCreatorApp:
    def __init__(self, _root):
        self.root = _root
        self.root.title("Dataset Creator")
        self.root.geometry(f"1000x1200")
        self.index = 0
        self.shift_multiplier = 1  # Multiplier for adjustments (1x by default)

        # Load gripper image
        self.gripper_image = Image.open(GRIPPER_IMAGE_PATH).convert("RGBA")

        # Load machined part images
        self.machined_parts = sorted([
            os.path.join(MACHINED_PARTS_FOLDER, f)
            for f in os.listdir(MACHINED_PARTS_FOLDER)
            if f.lower().endswith(".png")
        ])

        if not self.machined_parts:
            raise FileNotFoundError("No machined part images found in the specified folder!")

        # Declare images parameters
        self.tk_image = None
        self.base_image = None
        self.current_image_path = None

        # Initialize parameters
        self.x = 100
        self.y = 100
        self.alpha = 0

        # Create UI components
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack()

        self.x_slider = ttk.Scale(self.controls_frame, from_=0, to=500, command=self.update_image)
        self.x_slider.grid(row=0, column=1)
        self.x_label = ttk.Label(self.controls_frame, text="")
        self.x_label.grid(row=0, column=0)

        self.y_slider = ttk.Scale(self.controls_frame, from_=0, to=500, command=self.update_image)
        self.y_slider.grid(row=1, column=1)
        self.y_label = ttk.Label(self.controls_frame, text="")
        self.y_label.grid(row=1, column=0)

        self.alpha_slider = ttk.Scale(self.controls_frame, from_=-math.pi, to=math.pi, command=self.update_image)
        self.alpha_slider.grid(row=2, column=1)
        self.alpha_label = ttk.Label(self.controls_frame, text="")
        self.alpha_label.grid(row=2, column=0)

        self.save_button = ttk.Button(self.root, text="Save & Next", command=self.save_data)
        self.save_button.pack()

        self.skip_button = ttk.Button(self.root, text="Skip", command=self.skip_image)
        self.skip_button.pack()

        # Bind keyboard events
        self.root.bind("<Left>", self.decrease_x)
        self.root.bind("<Right>", self.increase_x)
        self.root.bind("<Up>", self.decrease_y)
        self.root.bind("<Down>", self.increase_y)
        self.root.bind("a", self.decrease_alpha)
        self.root.bind("d", self.increase_alpha)
        self.root.bind("A", self.decrease_alpha)
        self.root.bind("D", self.increase_alpha)
        self.root.bind("<Return>", lambda event: self.save_data())
        self.root.bind("<Shift_L>", self.activate_shift)
        self.root.bind("<KeyRelease-Shift_L>", self.deactivate_shift)
        self.root.bind("<space>", lambda event: self.skip_image())

        # Load first image
        self.load_image()

    def load_image(self):
        # Load current machined part image
        self.current_image_path = self.machined_parts[self.index]
        self.base_image = Image.open(self.current_image_path).convert("RGBA")
        self.pad(self.calculate_padding())
        input_tensor, bias_x, bias_y, norm_x, norm_y = parse_images_to_input_tensor(self.current_image_path,
                                                                                    GRIPPER_IMAGE_PATH)
        with torch.no_grad():
            pred = model(input_tensor).squeeze(0)
        x, y, alpha = denormalize_predictions(pred, bias_x, bias_y, norm_x, norm_y)
        x = x + self.base_image.width // 2
        y = y + self.base_image.height // 2
        self.x_slider.set(x)
        self.x_slider.configure(from_=0, to=self.base_image.width)
        self.y_slider.set(y)
        self.y_slider.configure(from_=0, to=self.base_image.height)
        self.alpha_slider.set(alpha)
        self.update_image()

    def skip_image(self):
        """Skips the current image and loads the next one."""
        # Move to the next image
        if (self.index + 1) == len(self.machined_parts):
            quit()
        self.index = (self.index + 1) % len(self.machined_parts)
        self.load_image()

    def update_image(self, *args):
        self.x = int(self.x_slider.get())
        self.y = int(self.y_slider.get())
        self.alpha = float(self.alpha_slider.get())
        self.x_label.config(text=f"X = {self.x - self.base_image.width // 2}")
        self.y_label.config(text=f"Y = {self.y - self.base_image.height // 2}")
        self.alpha_label.config(text=f"Alpha = {self.alpha:.2f}")
        overlay_image = overlay_images(self.base_image, self.gripper_image, self.x, self.y, self.alpha)
        self.display_image(overlay_image)

    def activate_shift(self, *args):
        self.shift_multiplier = 10

    def deactivate_shift(self, *args):
        self.shift_multiplier = 1

    def display_image(self, img):
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image)

    # Key Bind Handlers
    def decrease_x(self, *args):
        if self.x > self.x_slider.cget("from"):
            self.x_slider.set(self.x - 1 * self.shift_multiplier)

    def increase_x(self, *args):
        if self.x < self.x_slider.cget("to"):
            self.x_slider.set(self.x + 1 * self.shift_multiplier)

    def decrease_y(self, *args):
        if self.y > self.y_slider.cget("from"):
            self.y_slider.set(self.y - 1 * self.shift_multiplier)

    def increase_y(self, *args):
        if self.y < self.y_slider.cget("to"):
            self.y_slider.set(self.y + 1 * self.shift_multiplier)

    def decrease_alpha(self, *args):
        if self.alpha > self.alpha_slider.cget("from"):
            self.alpha_slider.set(self.alpha - 0.1 * self.shift_multiplier)

    def increase_alpha(self, *args):
        if self.alpha < self.alpha_slider.cget("to"):
            self.alpha_slider.set(self.alpha + 0.1 * self.shift_multiplier)

    def pad(self, target_pad_size):
        # First pad the image to maintain aspect ratio
        target_pad_width, target_pad_height = target_pad_size
        current_height, current_width = self.base_image.size

        # Calculate padding
        pad_width = max(target_pad_width - current_width, 0)
        pad_height = max(target_pad_height - current_height, 0)

        top = int(pad_height // 2)
        bottom = int(pad_height - top)
        left = int(pad_width // 2)
        right = int(pad_width - left)

        self.base_image = ImageOps.expand(self.base_image, border=(left, top, right, bottom), fill=0)

    def calculate_padding(self):
        max_dimension = max(self.base_image.height, self.base_image.width)
        return (max_dimension * np.sqrt(2) + 1 // 2 * 2), (max_dimension * np.sqrt(2) + 1) // 2 * 2

    def calculate_rotation_of_parameters(self, beta):
        radius = math.sqrt(self.x ** 2 + self.y ** 2)
        angle_to_origin = math.atan2(-self.y, self.x)

        new_angle = angle_to_origin - beta

        rotated_x = radius * math.cos(new_angle)
        rotated_y = -radius * math.sin(new_angle)

        rotated_alpha = (self.alpha + beta + 3 * math.pi) % (2 * math.pi) - math.pi
        return rotated_x, rotated_y, rotated_alpha

    def rotate_image(self, beta):
        beta_degrees = -beta * (180.0 / math.pi)
        return self.base_image.rotate(angle=beta_degrees, resample=Image.Resampling.BICUBIC, expand=False)

    def save_data(self):
        self.x = self.x - self.base_image.width // 2
        self.y = self.y - self.base_image.height // 2

        # Create a numbered sub-folder
        folder_number = len(os.listdir(DATASET_FOLDER))
        save_folder = os.path.join(DATASET_FOLDER, str(folder_number))
        os.makedirs(save_folder)

        # Number of elements and range
        num_elements = 4
        range_start = -np.pi
        range_end = np.pi

        # Create evenly spaced radian values
        radians = np.linspace(range_start, range_end, num_elements)

        # Add random noise to each value within (-pi/12, pi/12)
        noise_range = np.pi / 4
        random_noise = np.random.uniform(-noise_range, noise_range, num_elements)

        radians_with_noise = radians + random_noise
        for beta in radians_with_noise:
            self.base_image = enhance_image(self.base_image)

            # Create a numbered sub-folder
            subfolder_number = len(os.listdir(save_folder)) // 3

            mirrored_subfolder_name = "mirrored" + str(subfolder_number)
            normal_subfolder_name = "normal" + str(subfolder_number)
            inverse_subfolder_name = "inverse" + str(subfolder_number)

            mirrored_subfolder = os.path.join(save_folder, mirrored_subfolder_name)
            normal_subfolder = os.path.join(save_folder, normal_subfolder_name)
            inverse_subfolder = os.path.join(save_folder, inverse_subfolder_name)

            os.makedirs(mirrored_subfolder)
            os.makedirs(normal_subfolder)
            os.makedirs(inverse_subfolder)

            rotated_x, rotated_y, rotated_alpha = self.calculate_rotation_of_parameters(beta)
            rotated_base_image = self.rotate_image(beta)

            base_image_path_normal = os.path.join(normal_subfolder, "machined_part.png")
            gripper_image_path_normal = os.path.join(normal_subfolder, "gripper.png")
            csv_path_normal = os.path.join(normal_subfolder, "parameters.csv")

            rotated_base_image.save(base_image_path_normal)
            self.gripper_image.save(gripper_image_path_normal)

            with open(csv_path_normal, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "alpha"])
                writer.writerow(
                    [int(rotated_x), int(rotated_y),
                     f"{rotated_alpha:.5f}"])

            base_image_path_inverted = os.path.join(inverse_subfolder, "machined_part.png")
            gripper_image_path_inverted = os.path.join(inverse_subfolder, "gripper.png")
            csv_path_inverted = os.path.join(inverse_subfolder, "parameters.csv")

            inverted_base_image = inverse_image(rotated_base_image)

            inverted_base_image.save(base_image_path_inverted)
            self.gripper_image.save(gripper_image_path_inverted)

            with open(csv_path_inverted, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "alpha"])
                writer.writerow(
                    [int(rotated_x), int(rotated_y),
                     f"{rotated_alpha:.5f}"])

            mirrored_x, mirrored_y, mirrored_alpha = calculate_mirrored_parameters(rotated_x, rotated_y, rotated_alpha)
            mirrored_rotated_image = mirror_image(rotated_base_image)

            base_image_path_mirrored = os.path.join(mirrored_subfolder, "machined_part.png")
            gripper_image_path_mirrored = os.path.join(mirrored_subfolder, "gripper.png")
            csv_path_mirrored = os.path.join(mirrored_subfolder, "parameters.csv")

            mirrored_rotated_image.save(base_image_path_mirrored)
            self.gripper_image.save(gripper_image_path_mirrored)

            with open(csv_path_mirrored, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["x", "y", "alpha"])
                writer.writerow(
                    [int(mirrored_x), int(mirrored_y),
                     f"{mirrored_alpha:.5f}"])

        # Move to the next image
        if (self.index + 1) == len(self.machined_parts):
            quit()
        self.index = (self.index + 1) % len(self.machined_parts)

        self.load_image()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetCreatorApp(root)
    root.mainloop()
