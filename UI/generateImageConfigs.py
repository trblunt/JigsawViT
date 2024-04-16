import os
import random
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt


#Define configurations
puzzle_side_length = 14  # 14x14 grid
piece_size = (16, 16)   # Size of each piece

# Function to shuffle and rotate pieces
def shuffle_and_rotate_pieces(pieces):
    shuffled_pieces = random.sample(pieces, len(pieces))
    rotated_pieces = [piece.rotate(random.choice([0, 90, 180, 270])) for piece in shuffled_pieces]
    right_answers = [pieces.index(piece) for piece in shuffled_pieces]
    rotations = [int(piece.rotate(0).info['angle'] / 90) for piece in rotated_pieces]  # 0, 1, 2, 3 for 0, 90, 180, 270 degrees
    return rotated_pieces, right_answers, rotations

# Function to save the dataset
def save_dataset(h5file, image_name, original_image, rotated_pieces, right_answers, rotations):
    # Create a group for this image
    grp = h5file.create_group(image_name)
    # Save the original image
    grp.create_dataset('original', data=np.array(original_image), compression="gzip")
    # Save the shuffled and rotated pieces as a dataset
    shuffled_data = np.array([np.array(piece) for piece in rotated_pieces])
    grp.create_dataset('shuffled', data=shuffled_data, compression="gzip")
    # Save the right answers
    grp.create_dataset('answers', data=np.array(right_answers), compression="gzip")
    # Save the rotations
    grp.create_dataset('rotations', data=np.array(rotations), compression="gzip")

# Function to cut the image into pieces
def cut_into_pieces(image, puzzle_side_length, piece_size):
    pieces = []
    
    for i in range(puzzle_side_length):
        for j in range(puzzle_side_length):
            left = j * piece_size[0]
            upper = i * piece_size[1]
            right = left + piece_size[0]
            lower = upper + piece_size[1]
            
            piece = image.crop((left, upper, right, lower))
            piece.info['angle'] = 0 # set default angle of rotaiton to 0 degrees
            pieces.append(piece)
            
    return pieces

# Function to visualize the original and shuffled images - if needed!
def visualize_images(original_image, shuffled_pieces, puzzle_side_length, piece_size):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].title.set_text('Original Image (resized)')
    axs[0].axis('off')

    # Create a new image for the shuffled pieces
    shuffled_image = Image.new('RGB', original_image.size)

    for idx, piece in enumerate(shuffled_pieces):
        row_idx, col_idx = divmod(idx, puzzle_side_length)
        x = col_idx * piece_size[0]
        y = row_idx * piece_size[1]
        # print(f"Piece value: \n{piece}\n")
        shuffled_image.paste(piece, box=(x, y), mask=None)

    # Display the shuffled image
    axs[1].imshow(shuffled_image)
    axs[1].title.set_text('Shuffled Pieces')
    axs[1].axis('off')

    # Show the plot
    plt.show()

from PIL import Image

def get_visualized_images(original_image, shuffled_pieces, puzzle_side_length, piece_size):
    # Create a new image for the shuffled pieces
    shuffled_image = Image.new('RGB', original_image.size)
    
    for idx, piece in enumerate(shuffled_pieces):
        row_idx, col_idx = divmod(idx, puzzle_side_length)
        x = col_idx * piece_size[0]
        y = row_idx * piece_size[1]
        shuffled_image.paste(piece, box=(x, y), mask=None)
    
    return original_image, shuffled_image

    


def print_package_versions():
    import pkg_resources

    # List of package names you are using in your script
    packages = ['numpy', 'matplotlib', 'h5py', 'pillow', 'tk']  # Add other package names as needed
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    print("\nPackage Versions used in this project:")
    for package in packages:
        # Normalize package name to lowercase to match keys in the installed_packages
        package_name = package.lower()
        if package_name in installed_packages:
            print(f"{package}: {installed_packages[package_name]}")
        else:
            print(f"{package}: Not found")

