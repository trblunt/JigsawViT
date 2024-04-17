import math
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
IMAGE_RESIZE_VAL = 224

def update_puzzle_side_length():
    """Update puzzle side length from the entry box."""
    global puzzleSideLength
    global pieceSize
    try:
        puzzleSideLength = int(puzzle_side_length_var.get())
        pieceSizeDims = math.ceil(IMAGE_RESIZE_VAL/puzzleSideLength)
        pieceSize = (pieceSizeDims, pieceSizeDims)
        status_label.config(text="Puzzle side length updated.")
    except:
        status_label.config(text="Invalid puzzle side length. Enter an integer.")

def load_image(path):
    """Load an image from the given path."""
    try:
        image = Image.open(path)
        if image.width != image.height:
            messagebox.showinfo("Image Resize", f"The image is not square. It will be resized to {IMAGE_RESIZE_VAL}x{IMAGE_RESIZE_VAL} pixels.")
        image = image.resize((IMAGE_RESIZE_VAL, IMAGE_RESIZE_VAL))

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

def saveShuffledImage():
    """Saves the shuffled image to a file."""
    if hasattr(shuffled_image_label, 'shuffled_image'):
        # Opens a file dialog asking the user where to save the image
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                    filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
        if file_path:
            # Save the shuffled image at the chosen path
            shuffled_image_label.shuffled_image.save(file_path)
            messagebox.showinfo("Save Image", "Shuffled image saved successfully!")
        else:
            messagebox.showinfo("Save Image", "Save operation cancelled.")
    else:
        messagebox.showerror("Save Error", "No shuffled image to save.")

def clear_images():
    """Clears all images and resets the GUI to its initial state."""
    image_label.config(image='', text="Drag and drop an image here")
    image_label.image = None
    shuffled_image_label.config(image='', text="Shuffled and rotated puzzle pieces will appear here")
    shuffled_image_label.image = None
    status_label.config(text="No image loaded")


# Call the function to print package versions
# config.print_package_versions()

# Create a drag-and-drop enabled Tkinter window
root = TkinterDnD.Tk()
root.title("Image Loader and Folder Processor")
root.geometry('1000x800') # Increased size to accommodate new elements
root.minsize(600, 600)

# Dark mode styling
bg_color = "#333333"  # Dark gray
button_color = "#555555"  # Lighter gray
text_color = "#FFFFFF"  # White

# Applying the dark theme to GUI components
root.configure(bg=bg_color)

# Label to display the dropped image
image_label = Label(root, text="Drag and drop an image here", relief="solid", bd=1, fg=text_color, bg=bg_color)
image_label.pack(expand=True, fill=tk.BOTH)

# Label for shuffled output image
shuffled_image_label = Label(root, text="Shuffled and rotated puzzle pieces will appear here", relief="solid", bd=1, fg=text_color, bg=bg_color)
shuffled_image_label.pack(expand=True, fill=tk.BOTH)

# Button to save the shuffled image
save_image_button = tk.Button(root, text="Save Shuffled Image", command=saveShuffledImage, bg=button_color, fg=text_color)
save_image_button.pack(pady=10, padx=10, fill=tk.X)

# Add Clear Button to the GUI
clear_button = tk.Button(root, text="Clear Images", command=clear_images, bg=button_color, fg=text_color)
clear_button.pack(pady=10, padx=10, fill=tk.X)

# Applying dark mode colors to the puzzle side length label and entry
puzzle_side_length_label = tk.Label(root, text="Enter Puzzle Dimension Length:", bg=bg_color, fg=text_color)
puzzle_side_length_label.pack(pady=(20, 0))  # Add some padding above the label

puzzle_side_length_var = StringVar(value="14")
puzzle_side_length_entry = Entry(root, textvariable=puzzle_side_length_var, bg=button_color, fg=text_color, insertbackground=text_color)  # Ensure cursor is visible
puzzle_side_length_entry.pack(pady=5)
puzzle_side_length_entry.bind("<Return>", lambda event: update_puzzle_side_length())

# Label for entry note
entry_note_label = tk.Label(root, text="*Press Enter/Return to save value changes*", bg=bg_color, fg=text_color)
entry_note_label.pack(pady=(20, 0))  # Add some padding above the label


# Button to select a folder
folder_button = tk.Button(root, text="Select Image Folder (not wired to work yet!!!)", command=select_folder, bg=button_color, fg=text_color)
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


# Color schemes for dark and light modes
themes = {
    "dark": {
        "bg_color": "#333333",
        "text_color": "#FFFFFF",
        "button_color": "#555555",
        "entry_bg": "#555555",
        "entry_fg": "#FFFFFF",
        "insert_bg": "#FFFFFF"
    },
    "light": {
        "bg_color": "#FFFFFF",
        "text_color": "#000000",
        "button_color": "#CCCCCC",
        "entry_bg": "#FFFFFF",
        "entry_fg": "#000000",
        "insert_bg": "#000000"
    }
}

current_theme = "light"  # Start with light mode

def apply_theme(theme):
    global current_theme
    current_theme = theme
    theme_colors = themes[theme]
    
    # Apply theme colors to all components
    root.configure(bg=theme_colors['bg_color'])
    
    image_label.config(bg=theme_colors['bg_color'], fg=theme_colors['text_color'])
    shuffled_image_label.config(bg=theme_colors['bg_color'], fg=theme_colors['text_color'])
    save_image_button.config(bg=theme_colors['button_color'], fg=theme_colors['text_color'])
    folder_button.config(bg=theme_colors['button_color'], fg=theme_colors['text_color'])
    clear_button.config(bg=theme_colors['button_color'], fg=theme_colors['text_color'])
    puzzle_side_length_label.config(bg=theme_colors['bg_color'], fg=theme_colors['text_color'])
    puzzle_side_length_entry.config(bg=theme_colors['entry_bg'], fg=theme_colors['entry_fg'], insertbackground=theme_colors['insert_bg'])
    entry_note_label.config(bg=theme_colors['bg_color'], fg=theme_colors['text_color'])
    status_label.config(bg=theme_colors['bg_color'], fg=theme_colors['text_color'])

def toggle_theme():
    if current_theme == "light":
        apply_theme("dark")
    else:
        apply_theme("light")

apply_theme(current_theme)


# Button to toggle theme
theme_toggle_button = tk.Button(root, text="Toggle Light/Dark Mode", command=toggle_theme, bg="#CCCCCC")
theme_toggle_button.pack(pady=10, padx=10, fill=tk.X)


# Enable dropping files into the window
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop)

root.mainloop()


