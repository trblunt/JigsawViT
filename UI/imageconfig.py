import os
import random
from PIL import Image
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load the image
image = Image.open('/Users/nehabalamurugan/Downloads/IMG_8983 2.JPG')
image = image.resize((224, 224))  # Resize the image to 224x224 if it's not already

# TO DO: Define paths
imagenet_directory = 'path/to/imagenet/images'
output_h5_path = 'path/to/output/dataset.h5'

#Definec configurations
image_paths = [os.path.join(imagenet_directory, filename) for filename in os.listdir(imagenet_directory)]
puzzle_side_length = 14  # 14x14 grid
piece_size = (16, 16)   # Size of each piece

# Function to shuffle and rotate pieces
def shuffle_and_rotate_pieces(pieces):
    shuffled_pieces = random.sample(pieces, len(pieces))
    rotated_pieces = [piece.rotate(random.choice([0, 90, 180, 270])) for piece in shuffled_pieces]
    # The "right answer" is the original index in the pieces list
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

# Main process to generate the dataset
def process_image_dataset(image_paths, output_h5_path):
    with h5py.File(output_h5_path, 'w') as h5file:
        for image_path in image_paths:
            # Load and preprocess the image
            image_name = os.path.basename(image_path).split('.')[0]
            image = Image.open(image_path).resize((224, 224))
            pieces = cut_into_pieces(image, puzzle_side_length, piece_size)
            
            # Shuffle and rotate pieces
            rotated_pieces, right_answers, rotations = shuffle_and_rotate_pieces(pieces)
            
            # Save to the HDF5 dataset
            save_dataset(h5file, image_name, image, rotated_pieces, right_answers, rotations)

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
            pieces.append(piece)
            
    return pieces

# Function to visualize the original and shuffled images - if needed!
def visualize_images(original_image, shuffled_pieces, puzzle_side_length, piece_size):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axs[0].imshow(original_image)
    axs[0].title.set_text('Original Image')
    axs[0].axis('off')

    # Create a new image for the shuffled pieces
    shuffled_image = Image.new('RGB', original_image.size)

    for idx, piece in enumerate(shuffled_pieces):
        row_idx, col_idx = divmod(idx, puzzle_side_length)
        x = col_idx * piece_size[0]
        y = row_idx * piece_size[1]
        shuffled_image.paste(piece, (x, y))

    # Display the shuffled image
    axs[1].imshow(shuffled_image)
    axs[1].title.set_text('Shuffled Pieces')
    axs[1].axis('off')

    # Show the plot
    plt.show()




# Process ImageNet dataset
process_image_dataset(image_paths, output_h5_path)