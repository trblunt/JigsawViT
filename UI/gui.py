import os
import tkinter as tk
from tkinter import filedialog, Label, Toplevel, messagebox, Entry, StringVar
from PIL import Image, ImageTk
from tkinterdnd2 import TkinterDnD, DND_FILES

# helper files
import generateImageConfigs as config

# Variables to hold piece size and puzzle side length
pieceSize = (16, 16)  # default piece size as a tuple
puzzleSideLength = 14  # default puzzle side length

def update_piece_size():
    """Update piece size from the entry box."""
    global pieceSize
    try:
        a, b = map(int, piece_size_var.get().split('x'))
        pieceSize = (a, b)
        status_label.config(text="Piece size updated.")
    except:
        status_label.config(text="Invalid piece size format. Use 'AxB'.")

def update_puzzle_side_length():
    """Update puzzle side length from the entry box."""
    global puzzleSideLength
    try:
        puzzleSideLength = int(puzzle_side_length_var.get())
        status_label.config(text="Puzzle side length updated.")
    except:
        status_label.config(text="Invalid puzzle side length. Enter an integer.")

def load_image(path):
    """Load an image from the given path."""
    try:
        image = Image.open(path)
        if image.width != image.height:
            messagebox.showinfo("Image Resize", "The image is not square. It will be resized to 224x224 pixels.")
        image = image.resize((224, 224))

        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image
        status_label.config(text=f"Loaded: {os.path.basename(path)}")

        performShuffle(image)
        image_label.original_image = image

    except Exception as e:
        status_label.config(text=f"Failed to load image: {str(e)}")

def drop(event):
    """Handle the drop event for image files."""
    file_paths = root.tk.splitlist(event.data)
    for file_path in file_paths:
        file_path = file_path.strip('{}')
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            load_image(file_path)
            break

def select_folder():
    """Handle selection of a directory."""
    folder_path = filedialog.askdirectory()
    if folder_path:
        process_images(folder_path)

def process_images(folder_path):
    """Process all images in the directory."""
    try:
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                full_path = os.path.join(folder_path, file_name)
                img = Image.open(full_path)
                print(f"Processing \"{file_name}\": Image size is {img.size}")
        messagebox.showinfo("Process Complete", "All images processed.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process images: {str(e)}")


def display_result(original_image, shuffled_image):
    # Initialize the window
    window = tk.Tk()
    window.title("Image Puzzle")

    # Convert PIL images to a format that tkinter can use
    original_photo = ImageTk.PhotoImage(original_image)
    shuffled_photo = ImageTk.PhotoImage(shuffled_image)

    # Create labels for images and add them to the window
    original_label = tk.Label(window, image=original_photo)
    original_label.image = original_photo  # keep a reference!
    original_label.pack(side="left")

    shuffled_label = tk.Label(window, image=shuffled_photo)
    shuffled_label.image = shuffled_photo  # keep a reference!
    shuffled_label.pack(side="left")

    # Start the GUI
    window.mainloop()


def showShuffledImage(theImage, theShuffledPieces):
    """Display shuffled image pieces."""
    original_image, shuffled_image = config.get_visualized_images(theImage, theShuffledPieces, puzzleSideLength, pieceSize)
    tk_shuffled_image = ImageTk.PhotoImage(shuffled_image)
    shuffled_image_label.config(image=tk_shuffled_image)
    shuffled_image_label.image = tk_shuffled_image

    # display_result(original_image, shuffled_image)
    shuffled_image_label.shuffled_image = shuffled_image

def performShuffle(theImage):
    """Shuffles the image that has been dragged and dropped."""
    print(f"\npuzzleSideLength value: {puzzleSideLength}, pieceSize value: {pieceSize}")
    thePieces = config.cut_into_pieces(theImage, puzzleSideLength, pieceSize)
    rotated_pieces, _, _ = config.shuffle_and_rotate_pieces(thePieces)

    showShuffledImage(theImage, rotated_pieces)

# Call the function to print package versions
config.print_package_versions()

# Create a drag-and-drop enabled Tkinter window
root = TkinterDnD.Tk()
root.title("Image Loader and Folder Processor")
root.geometry('1280x720') # Increased size to accommodate new elements
root.minsize(600, 600)

# Styling
bg_color = "#f0f0f0"
button_color = "#d9d9d9"
text_color = "#333"

# Label to display the dropped image
image_label = Label(root, text="Drag and drop an image here", relief="solid", bd=1, fg=text_color, bg=bg_color)
image_label.pack(expand=True, fill=tk.BOTH)

# Label for shuffled output image
shuffled_image_label = Label(root, text="Shuffled and rotated puzzle pieces will appear here", relief="solid", bd=1, fg=text_color, bg=bg_color)
shuffled_image_label.pack(expand=True, fill=tk.BOTH)


# Label for piece size entry
piece_size_label = tk.Label(root, text="Enter Piece Size")
piece_size_label.pack(pady=(20, 0))  # Add some padding above the label
# Entry for piece size
piece_size_var = StringVar(value="16x16")
piece_size_entry = Entry(root, textvariable=piece_size_var)
piece_size_entry.pack(pady=5)
piece_size_entry.bind("<Return>", lambda event: update_piece_size())


# Label for puzzle side length entry
puzzle_side_length_label = tk.Label(root, text="Enter Puzzle Side Length:")
puzzle_side_length_label.pack(pady=(20, 0))  # Add some padding above the label
# Entry for puzzle side length
puzzle_side_length_var = StringVar(value="14")
puzzle_side_length_entry = Entry(root, textvariable=puzzle_side_length_var)
puzzle_side_length_entry.pack(pady=5)
puzzle_side_length_entry.bind("<Return>", lambda event: update_puzzle_side_length())

# Label for entry
puzzle_side_length_label = tk.Label(root, text="*Press Enter/Return to save value changes*")
puzzle_side_length_label.pack(pady=(20, 0))  # Add some padding above the label

# Button to select a folder
folder_button = tk.Button(root, text="Select Image Folder (not wired to work yet!!!)", command=select_folder, bg=button_color)
folder_button.pack(pady=10, padx=10, fill=tk.X)

# Status label
status_label = tk.Label(root, text="No image loaded", bg=bg_color, fg=text_color)
status_label.pack(side=tk.BOTTOM, fill=tk.X)


def resize_images():
    # Check if images exist
    if hasattr(image_label, 'original_image') and hasattr(shuffled_image_label, 'shuffled_image'):
        # Get current size of the label or window to decide new image size
        new_width = image_label.winfo_width()
        new_height = image_label.winfo_height()
        
        # Resize the original image
        resized_original = image_label.original_image.resize((new_width, new_height))
        photo_original = ImageTk.PhotoImage(resized_original)
        image_label.config(image=photo_original)
        image_label.image = photo_original  # Keep a reference!
        
        # Resize the shuffled image
        resized_shuffled = shuffled_image_label.shuffled_image.resize((new_width, new_height))
        photo_shuffled = ImageTk.PhotoImage(resized_shuffled)
        shuffled_image_label.config(image=photo_shuffled)
        shuffled_image_label.image = photo_shuffled  # Keep a reference!

# root.bind('<Configure>', lambda e: resize_images()) # bug here, will fix later. 


# Enable dropping files into the window
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

root.mainloop()


