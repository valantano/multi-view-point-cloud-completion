from PIL import Image, ImageSequence
import os

def make_gif_loop_forever(input_path, output_path=None):
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_looped{ext}"

    with Image.open(input_path) as im:
        frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,  # Set to loop forever
            duration=im.info.get('duration', 80),
            disposal=2
        )
    print(f"Saved looping GIF to: {output_path}")

def get_gif_dimensions(input_path):
    """Get the width and height of a GIF"""
    with Image.open(input_path) as im:
        width, height = im.size
        return width, height

def crop_gif(input_path, output_path, new_width, new_height, x_offset=0, y_offset=0):
    """Crop a GIF to specified dimensions"""
    with Image.open(input_path) as im:
        frames = []
        orig_width, orig_height = im.size
        
        # Calculate center position
        center_x = (orig_width - new_width) // 2
        center_y = (orig_height - new_height) // 2
        
        # Apply offsets (positive moves crop area up/left, negative moves down/right)
        left = center_x - x_offset
        top = center_y - y_offset
        right = left + new_width
        bottom = top + new_height
        
        # Ensure crop box is within image bounds
        left = max(0, min(left, orig_width - new_width))
        top = max(0, min(top, orig_height - new_height))
        right = left + new_width
        bottom = top + new_height
        
        for frame in ImageSequence.Iterator(im):
            # Crop the frame
            cropped_frame = frame.crop((left, top, right, bottom))
            frames.append(cropped_frame)
        
        # Save the cropped GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            loop=im.info.get('loop', 0),
            duration=im.info.get('duration', 80),
            disposal=2
        )
    print(f"Cropped GIF saved to: {output_path}")

def main():
    gif_folder = "/home/valantano/mt/repository/gifs/process"
    file_list = [f for f in os.listdir(gif_folder) if f.endswith('.gif')]
    print(f"Found GIF files: {file_list} in {gif_folder}")
    
    # Display dimensions for all GIFs
    print("\nGIF Dimensions:")
    for file_name in file_list:
        input_path = os.path.join(gif_folder, file_name)
        width, height = get_gif_dimensions(input_path)
        print(f"{file_name}: {width}x{height}")
    
    # Ask user for cropping preferences
    crop_gifs = input("\nDo you want to crop the GIFs? (y/n): ").lower() == 'y'
    
    if crop_gifs:
        new_width = int(input("Enter new width: "))
        new_height = int(input("Enter new height: "))
        x_offset = int(input("Enter x offset (0 for center, + moves left, - moves right): ") or "0")
        y_offset = int(input("Enter y offset (0 for center, + moves up, - moves down): ") or "0")
        
        for file_name in file_list:
            input_path = os.path.join(gif_folder, file_name)
            base_name = os.path.splitext(file_name)[0]
            
            cropped_output = os.path.join(gif_folder, f"{base_name}_cropped.gif")
            crop_gif(input_path, cropped_output, new_width, new_height, x_offset, y_offset)
    
    # Original looping functionality
    loop_gifs = input("\nDo you want to make GIFs loop forever? (y/n): ").lower() == 'y'
    
    if loop_gifs:
        for file_name in file_list:
            input_path = os.path.join(gif_folder, file_name)
            output_path = os.path.join(gif_folder, f"looped_{file_name}")
            make_gif_loop_forever(input_path, output_path)

if __name__ == "__main__":
    main()
    print("GIF processing completed.")