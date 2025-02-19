import os
from PIL import Image
import numpy as np
import shutil
import time
import tempfile
from paddleocr import PaddleOCR
import json
import cv2
from moviepy.editor import VideoFileClip
from datetime import datetime

def pixel_difference(img1, img2, region):
    """
    Compares the pixel values of the same region in two images and calculates the percentage difference.
    This version accepts already opened Image objects.
    """
    try:
        # Open the images
        img1 = Image.open(img1)
        img2 = Image.open(img2)
    except Exception as e:
        # print(e)
        pass
    # Crop the region from both images
    img1_cropped = img1.crop(region)
    img2_cropped = img2.crop(region)
    # img1_cropped.show()
    # img2_cropped.show()
    # Convert images to numpy arrays
    img1_array = np.array(img1_cropped)
    img2_array = np.array(img2_cropped)
    
    # Ensure the images are of the same shape
    if img1_array.shape != img2_array.shape:
        raise ValueError("The regions in both images must have the same dimensions")
    
    # Compare pixel differences
    diff = np.abs(img1_array - img2_array)
    diff_count = np.count_nonzero(diff)  # Count non-zero differences
    total_pixels = np.prod(img1_array.shape)  # Total number of pixels in the region
    
    # Calculate percentage difference
    percentage_diff = (diff_count / total_pixels) * 100
    return percentage_diff

def find_unique_frames(input_dir, output_dir, region, threshold=30):
    """
    Processes a directory of images and saves unique frames to a new directory based on pixel differences.

    Parameters:
    - input_dir: Path to the input directory containing images.
    - output_dir: Path to the output directory where unique images will be saved.
    - region: A tuple (x1, y1, x2, y2) defining the region to compare.
    - threshold: Percentage difference threshold to consider an image "unique".
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted list of image files in the input directory
    image_files = sorted((f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"Number of images to compare: {len(image_files)}")
    
    if not image_files:
        print("No images found in the input directory.")
        return
    
    # Load the first image as the initial "unique" image
    first_image_path = os.path.join(input_dir, image_files[0])
    prev_image = Image.open(first_image_path)
    shutil.copy(first_image_path, os.path.join(output_dir, image_files[0]))  # Save the first image to the output directory
    print(f"Saved unique image: {image_files[0]}")
    
    # Iterate through the rest of the images
    for image_file in image_files[1:]:
        current_image_path = os.path.join(input_dir, image_file)
        current_image = Image.open(current_image_path)
        
        # Compare the current image with the previous unique image
        diff = pixel_difference(prev_image, current_image, region)
        
        if diff > threshold:
            # If the difference is significant, save the image and update the previous unique image
            shutil.copy(current_image_path, os.path.join(output_dir, image_file))
            prev_image = current_image
            print(f"Saved unique image: {image_file}")

def get_text_bounding_box(image_path, ocr, region):
    """
    Runs PaddleOCR on an image and returns the bounding box of the text detected
    which is inside or intersects the given region of interest.

    Parameters:
    - image_path: Path to the image file.
    - region: A tuple (x1, y1, x2, y2) defining the region of interest.

    Returns:
    - A list of tuples containing bounding boxes and the corresponding text that intersect with the region.
    """
    
    result = ocr.ocr(image_path, cls=False)
    print(f"OCR Result for {image_path}:\n{result}")
    x1, y1, x2, y2 = region
    y1 = y1 - 80 # Adjust to include double decker text
    bounding_boxes_with_text = []

    for line in result:
        if line == None:
            return None
        for word_info in line:
            bbox = word_info[0]
            text = word_info[1][0]
            # Check if the bounding box intersects with the region of interest
            if not (bbox[2][0] < x1 or bbox[0][0] > x2 or bbox[2][1] < y1 or bbox[0][1] > y2):
                print(f"{text} with {bbox} intersects with the region of interest, appending.")
                bounding_boxes_with_text.append((bbox, text))
            else:
                print(f"{text} with {bbox} does not intersect with the region of interest, ignoring.")

    return bounding_boxes_with_text

def ocr_directory_to_json(input_dir, region, ocr_region, output_json, ocr, existing_ocr_json=None):
    """
    Runs get_text_bounding_box on a directory of image files and writes the OCR'd text and bounding boxes to a JSON file.
    Skips files that have already been OCR'd if an existing JSON file is provided.

    Parameters:
    - input_dir: Path to the directory containing the image files.
    - region: A tuple (x1, y1, x2, y2) defining the region of interest.
    - output_json: Path to the output JSON file.
    - existing_ocr_json: Path to an existing JSON file with already OCR'd entries (optional).
    """
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(
    files,
    key=lambda x: int(x.split("_")[1].split(".")[0])  # Extract the number between "_" and ".jpg"
    )
    print("Images: ", image_files)
    ocr_results = []

    already_ocrd_frames = set()
    if existing_ocr_json and os.path.exists(existing_ocr_json):
        with open(existing_ocr_json, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            for entry in existing_data:
                already_ocrd_frames.add(entry["start_frame"])
            ocr_results.extend(existing_data)

    for image_file in image_files:
        frame_number = int(image_file.split('_')[1].split('.')[0])
        if frame_number in already_ocrd_frames:
            continue

        image_path = os.path.join(input_dir, image_file)
        bounding_boxes_with_text = get_text_bounding_box(image_path, ocr, ocr_region)
        print(f"frame: {image_file}: "+str(bounding_boxes_with_text))

        if bounding_boxes_with_text:
            merged_text = ""
            x1, y1 = float('inf'), float('inf')
            x2, y2 = float('-inf'), float('-inf')

            for bbox, text in bounding_boxes_with_text:
                merged_text += text
                x1 = min(x1, bbox[0][0])
                y1 = min(y1, bbox[0][1])
                x2 = max(x2, bbox[2][0])
                y2 = max(y2, bbox[2][1])
            print()
            merged_bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

            ocr_results.append({
                "text": merged_text,
                "bounding_box": merged_bbox,
                "start_frame": frame_number
            })
        else:
            file_idx = image_files.index(image_file)
            if file_idx > 0:
                for entry in ocr_results:
                    print(entry)
                    if "start_frame" in entry and entry["start_frame"] == int(image_files[file_idx-1].split('_')[1].split('.')[0]):
                        print(f"Setting end frame for {entry} to {str(int(frame_number))}")
                        print(f"Frame number type is {type(frame_number)}. Frame number: {str(frame_number)}")
                        entry["end_frame"] = int(frame_number) # If the image before a blank image isn't blank, set the previous image's end_frame to be the current image's frame number
                        print(f"Set the end frame to {entry['end_frame']} full entry: {entry}")
                        break
    for entry in ocr_results:
        if "start_frame" in entry and isinstance(entry["start_frame"], list):
            entry["start_frame"] = entry["start_frame"][0]
        if "end_frame" in entry and isinstance(entry["end_frame"], list):
            entry["end_frame"] = entry["end_frame"][0]
    print(f"OCR results: {ocr_results}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=4)

def remove_non_subtitle_frames(directory, region, json_file, ocr, region_reduction=None):
    """
    Reads a directory of files in the format "frame_0000.jpg" and removes groups of consecutive frames
    if no text is detected in the given region of interest. Updates the JSON file with the text, bounding box,
    and frame number of the last subtitle frame before the sequence of non-text frames.

    Parameters:
    - directory: Path to the directory containing the image files.
    - region: A tuple (x1, y1, x2, y2) defining the region of interest for OCR.
    - json_file: Path to the JSON file to update with subtitle information.
    """
    image_files = sorted((f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                         key=lambda x: int(x.split('_')[1].split('.')[0]))
    subtitles = []

    def is_consecutive(file1, file2):
        num1 = int(file1.split('_')[1].split('.')[0])
        num2 = int(file2.split('_')[1].split('.')[0])
        return num2 == num1 + 1

    consecutive_groups = []
    current_group = []

    for i, file in enumerate(image_files):
        if i == 0 or is_consecutive(image_files[i-1], file):
            current_group.append(file)
        else:
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
            current_group = [file]
    
    if len(current_group) > 1:
        consecutive_groups.append(current_group)

    for group in consecutive_groups:
        text_detected = False
        for file in group:
            image_path = os.path.join(directory, file)
            result = ocr.ocr(image_path, cls=False)
            
            if result == [None]:
                continue  # Skip this frame if OCR result is None

            x1, y1, x2, y2 = region
            for line in result:
                for word_info in line:
                    bbox = word_info[0]
                    if not (bbox[2][0] < x1 or bbox[0][0] > x2 or bbox[2][1] < y1 or bbox[0][1] > y2):
                        text_detected = True
                        break
                if text_detected:
                    break
            if text_detected:
                break

        if not text_detected:
            for file in group:
                os.remove(os.path.join(directory, file))
                print(f"Removed non-subtitle frame: {file}")
        else:
            last_subtitle_frame = group[0]
            last_image_path = os.path.join(directory, last_subtitle_frame)
            bounding_boxes_with_text = get_text_bounding_box(last_image_path, ocr, region)
            if bounding_boxes_with_text:
                for bbox, text in bounding_boxes_with_text:
                    if region_reduction:
                        x1, y1 = bbox[0]
                        x2, y2 = bbox[2]
                        width = x2 - x1
                        height = y2 - y1
                        original_width = width / region_reduction
                        original_height = height / region_reduction
                        new_x1 = x1 - (original_width - width) / 2
                        new_y1 = y1 - (original_height - height) / 2
                        new_x2 = x2 + (original_width - width) / 2
                        new_y2 = y2 + (original_height - height) / 2
                        bbox = [[new_x1, new_y1], [new_x2, new_y1], [new_x2, new_y2], [new_x1, new_y2]]

                subtitle_info = {
                    "text": text,
                    "bounding_box": bbox,
                    "end_frame": int(last_subtitle_frame.split('_')[1].split('.')[0])
                }
                subtitles.append(subtitle_info)
            else:
                text_detected = False
                for file in group:
                    os.remove(os.path.join(directory, file))
                    print(f"Removed non-subtitle frame: {file}")
    for entry in subtitles:
        if isinstance(entry["end_frame"], list):
            entry["end_frame"] = entry["end_frame"][0]
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)

def remove_duplicate_subtitle_frames(directory, region, json_file, ocr, fps=30):
    """
    Reads a directory of files in the format "frame_0000.jpg" and removes duplicate subtitle frames.
    Saves the subtitle text, bounding box, timing, and frames to a JSON file.

    Parameters:
    - directory: Path to the directory containing the image files.
    - region: A tuple (x1, y1, x2, y2) defining the region of interest for OCR.
    - fps: Frames per second used to sample the frames.
    """
    image_files = sorted((f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), key=lambda x: int(x.split('_')[1].split('.')[0]))
    subtitles = []
    existing_ocr_frames = {}

    # Load existing OCR data from JSON file
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            for entry in existing_data:
                existing_ocr_frames[entry["end_frame"]] = entry

    def is_consecutive(file1, file2):
        """
        Checks if two files are consecutive based on their frame numbers.
        """
        num1 = int(file1.split('_')[1].split('.')[0])
        num2 = int(file2.split('_')[1].split('.')[0])
        return num2 == num1 + 1

    def extract_text_within_region(result, region):
        """
        Extracts text within a specified region from the OCR result.
        """
        x1, y1, x2, y2 = region
        text = []
        for line in result:
            for word_info in line:
                bbox = word_info[0]
                if not (bbox[2][0] < x1 or bbox[0][0] > x2 or bbox[2][1] < y1 or bbox[0][1] > y2):
                    text.append(word_info[1][0])
        return " ".join(text)

    consecutive_groups = []
    current_group = []

    # Group consecutive frames together
    for i, file in enumerate(image_files):
        if i == 0 or is_consecutive(image_files[i-1], file):
            current_group.append(file)
        else:
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
            current_group = [file]
    
    if len(current_group) > 1:
        consecutive_groups.append(current_group)

    # Process each group of consecutive frames
    for group in consecutive_groups:
        second_image_path = os.path.join(directory, group[1])
        result_second = ocr.ocr(second_image_path, cls=False)
        text_second = extract_text_within_region(result_second, region)

        last_image_path = os.path.join(directory, group[-1])
        result_last = ocr.ocr(last_image_path, cls=False)
        text_last = extract_text_within_region(result_last, region)

        if text_second != text_last:
            second_last_image_path = os.path.join(directory, group[-2])
            result_second_last = ocr.ocr(second_last_image_path, cls=False)
            text_second_last = extract_text_within_region(result_second_last, region)

            if text_second == text_second_last:
                subtitle_info = {
                    "text": text_second,
                    "bounding_box": result_second[0][0][0],  # Bounding box of the detected text
                    "start_frame": group[0],
                }
                if group[-1] in existing_ocr_frames:
                    subtitle_info["end_frame"] = existing_ocr_frames[group[-1]]["end_frame"]
                    del existing_ocr_frames[group[-1]]
                subtitles.append(subtitle_info)
                for file in group[1:-1]:
                    os.remove(os.path.join(directory, file))
                    print(f"Removed duplicate subtitle frame: {file}")
            else:
                print(f"Group with multiple subtitles detected: {group}")
                return group

            # Add the last frame as a separate entry
            subtitle_info_last = {
                "text": text_last,
                "bounding_box": result_last[0][0][0],  # Bounding box of the detected text
                "start_frame": group[-1],
            }
            subtitles.append(subtitle_info_last)

    with open(os.path.join(directory, json_file), "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)

def json_to_srt(json_file, srt_file, fps):
    """
    Converts a JSON file containing subtitle information to an SRT file.

    Parameters:
    - json_file: Path to the JSON file containing subtitle information.
    - srt_file: Path to the output SRT file.
    - fps: Frames per second used to calculate the timings.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    def frame_to_timecode(frame, fps):
        total_seconds = frame / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    with open(srt_file, "w", encoding="utf-8") as f:
        for i, subtitle in enumerate(subtitles, start=1):
            if isinstance(subtitle["start_frame"], list):
                subtitle["start_frame"] = subtitle["start_frame"][0]
            if isinstance(subtitle["end_frame"], list):
                subtitle["end_frame"] = subtitle["end_frame"][0]
            start_time = frame_to_timecode(subtitle["start_frame"], fps)
            end_time = frame_to_timecode(subtitle["end_frame"], fps)
            text = subtitle["text"]
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def calculate_end_frames(subtitle_json, all_frames_folder):
    # The end frame is determined by the start frame of the next entry, regardless of whether the text matches.
    # Note:
    # - The end frame is set to the start frame of the next subtitle, even if it has matching text.
    """
    Calculates the end frame for each subtitle entry in the JSON if it does not exist.
    The end frame is determined by the start frame of the next entry with a different text.
    If no next frame is found, the end frame is set to the last frame in the folder.

    Parameters:
    - subtitle_json: Path to the JSON file containing subtitle entries.
    - all_frames_folder: Path to the folder containing all frames.
    """
    with open(subtitle_json, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    # Get the last frame number from the folder
    all_frames = sorted((f for f in os.listdir(all_frames_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))), key=lambda x: int(x.split('_')[1].split('.')[0]))
    last_frame_number = int(all_frames[-1].split('_')[1].split('.')[0])

    for i in range(len(subtitles) - 1):
        if "end_frame" not in subtitles[i] or not subtitles[i]["end_frame"]:
            subtitles[i]["end_frame"] = subtitles[i + 1]["start_frame"]

    # Adjust end_frame to be one frame less than the next start_frame
    for i in range(len(subtitles) - 1):
        if subtitles[i]["end_frame"] == subtitles[i + 1]["start_frame"]:
            subtitles[i]["end_frame"] -= 1

    # For the last subtitle entry, if it does not have an end_frame, set it to the last frame number
    if "end_frame" not in subtitles[-1] or not subtitles[-1]["end_frame"]:
        subtitles[-1]["end_frame"] = last_frame_number

    with open(subtitle_json, "w", encoding="utf-8") as f:
        json.dump(subtitles, f, ensure_ascii=False, indent=4)

def resize_region(input_path, output_path, region, scale):
    """
    Resize a specified region of the image to a lower amount and fill the new blank space with white.

    :param input_path: Path to the input image.
    :param output_path: Path to save the modified image.
    :param region: A tuple (x1, y1, x2, y2) defining the region to resize.
    :param scale: Scale factor to resize the region (e.g., 0.75 to shrink to 75%).
    """
    # Open the image
    img = Image.open(input_path)
    width, height = img.size

    # Extract the region to resize
    x1, y1, x2, y2 = region
    region_width = x2 - x1
    region_height = y2 - y1

    # Resize the region
    region_img = img.crop(region)
    new_size = (int(region_width * scale), int(region_height * scale))
    resized_region = region_img.resize(new_size, Image.LANCZOS)

    # Create a new image with white background
    new_img = Image.new("RGB", (width, height), "white")

    # Calculate the position to paste the resized region
    new_x1 = x1 + (region_width - new_size[0]) // 2
    new_y1 = y1 + (region_height - new_size[1]) // 2

    # Paste the resized region onto the new image
    new_img.paste(resized_region, (new_x1, new_y1))

    # Save the new image
    new_img.save(output_path)
    print(f"Image saved to {output_path}")

def merge_duplicate_subtitles(subtitle_json):
    """
    Merges duplicate subtitles that are next to one another in the JSON file.

    Parameters:
    - subtitle_json: Path to the JSON file containing subtitle entries.
    """
    with open(subtitle_json, "r", encoding="utf-8") as f:
        subtitles = json.load(f)

    merged_subtitles = []
    i = 0

    while i < len(subtitles):
        current_subtitle = subtitles[i]
        j = i + 1

        while j < len(subtitles) and subtitles[j]["text"] == current_subtitle["text"]:
            print(f"Merging subtitles: {current_subtitle} and {subtitles[j]}")
            if isinstance(subtitles[j]["end_frame"], list):
                subtitles[j]["end_frame"] = subtitles[j]["end_frame"][0]
            current_subtitle["end_frame"] = max(current_subtitle["end_frame"], subtitles[j]["end_frame"])
            current_subtitle["start_frame"] = min(current_subtitle["start_frame"], subtitles[j]["start_frame"])
            current_subtitle["bounding_box"] = [
                [
                    min(current_subtitle["bounding_box"][0][0], subtitles[j]["bounding_box"][0][0]),
                    min(current_subtitle["bounding_box"][0][1], subtitles[j]["bounding_box"][0][1])
                ],
                [
                    max(current_subtitle["bounding_box"][1][0], subtitles[j]["bounding_box"][1][0]),
                    min(current_subtitle["bounding_box"][1][1], subtitles[j]["bounding_box"][1][1])
                ],
                [
                    max(current_subtitle["bounding_box"][2][0], subtitles[j]["bounding_box"][2][0]),
                    max(current_subtitle["bounding_box"][2][1], subtitles[j]["bounding_box"][2][1])
                ],
                [
                    min(current_subtitle["bounding_box"][3][0], subtitles[j]["bounding_box"][3][0]),
                    max(current_subtitle["bounding_box"][3][1], subtitles[j]["bounding_box"][3][1])
                ]
            ]
            j += 1

        merged_subtitles.append(current_subtitle)
        i = j
        print(f"Merged subtitle: {current_subtitle}")

    with open(subtitle_json, "w", encoding="utf-8") as f:
        json.dump(merged_subtitles, f, ensure_ascii=False, indent=4)

def process_srt(file_path):
    """
    Processes an SRT (SubRip Subtitle) file to merge consecutive duplicate subtitles.
    
    This function performs the following operations:
    1. Parses the SRT file into subtitle entries.
    2. Merges consecutive entries with identical text after removing punctuation and numbers.
    3. Adjusts subtitle indices to maintain sequential order.
    4. Generates the cleaned SRT content.

    Args:
        file_path (str): Path to the SRT file to be processed.

    Returns:
        str: The cleaned SRT content with merged duplicate entries.
    """
    def parse_srt(content):
        blocks = content.split('\n\n')
        entries = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.split('\n')
            if len(lines) < 3:
                continue  # Invalid block, skip
            index_line = lines[0].strip()
            time_line = lines[1].strip()
            text_lines = [line.strip() for line in lines[2:]]
            text = ' '.join(text_lines)
            if '-->' not in time_line:
                continue  # Invalid time line, skip
            start, end = map(str.strip, time_line.split('-->'))
            try:
                index = int(index_line)
            except ValueError:
                index = 0  # Default to 0 if invalid
            entries.append({
                'index': index,
                'start': start,
                'end': end,
                'text': text
            })
        return entries

    def clean_text(text):
        punctuation = '.,。?？！!、・'
        return text.translate(str.maketrans('', '', punctuation)).strip()

    def remove_numbers(text):
        return ''.join(ch for ch in text if not ch.isdigit()).strip()

    def merge_entries(entries):
        processed = []
        i = 0
        while i < len(entries):
            current = entries[i]
            j = i + 1
            while j < len(entries):
                next_entry = entries[j]
                # Clean text by removing punctuation then digits
                current_comp = remove_numbers(clean_text(current['text']))
                next_comp = remove_numbers(clean_text(next_entry['text']))
                # Only merge if both are non-empty and equal after removing numbers
                if current_comp and next_comp and current_comp == next_comp:
                    current = {
                        'index': current['index'],
                        'start': current['start'],
                        'end': next_entry['end'],
                        'text': current['text']
                    }
                    j += 1
                else:
                    break
            processed.append(current)
            i = j
        return processed

    def generate_srt(entries):
        if not entries:
            return ""
        # Adjust indices to be sequential starting from the first entry's index
        starting_index = entries[0]['index']
        for i, entry in enumerate(entries):
            entry['index'] = starting_index + i
        blocks = []
        for entry in entries:
            block = f"{entry['index']}\n{entry['start']} --> {entry['end']}\n{entry['text']}"
            blocks.append(block)
        return '\n\n'.join(blocks)

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    entries = parse_srt(content)
    merged_entries = merge_entries(entries)
    return generate_srt(merged_entries)

def write_srt(content, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def convert_video(input_file, output_file):
    """
    Convert a video file to 720p (1280x720) resolution at 30fps using the h264 codec.

    Parameters:
        input_file (str): Path to the input video file.
        output_file (str): Path where the converted video will be saved.
    """
    # Load the original video
    clip = VideoFileClip(input_file)
    
    # Resize the clip to 720p, preserve aspect ratio
    clip_resized = clip.resize(height=720)
    
    # Write the video file with 30fps and using the h264 codec (libx264)
    clip_resized.write_videofile(
        output_file,
        fps=30,
        codec="libx264"
    )
    
    # Close clips to free resources
    clip.close()
    clip_resized.close()
    return output_file

# Extract frames from mp4
def extract_all_frames(video_path, output_dir):
    """
    Extracts all frames from an MP4 video file to a specified directory and prints progress.
    Frames are saved as JPEG images named "frame_00000.jpg", "frame_00001.jpg", etc.

    Args:
        video_path (str): Path to the input MP4 video file.
        output_dir (str): Path to the directory to save extracted frames.
                           The directory will be created if it doesn't exist.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Warning: Could not reliably determine total frames. Progress will be frame count only.")
        total_frames = None  # Indicate total frames is unknown

    frame_count = 0
    saved_frame_count = 0
    start_time = time.time()
    progress_interval = 1000  # Print progress every 1000 frames

    print("Extracting frames...") # Initial progress message

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_name = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
        cv2.imwrite(frame_name, frame)
        saved_frame_count += 1
        frame_count += 1

        if frame_count % progress_interval == 0:
            elapsed_time = time.time() - start_time
            frames_per_sec = frame_count / elapsed_time if elapsed_time > 0 else 0

            if total_frames:
                progress_percent = (frame_count / total_frames) * 100
                progress_str = f"Processed frame: {frame_count}/{total_frames} ({progress_percent:.2f}%) - FPS: {frames_per_sec:.2f}"
            else:
                progress_str = f"Processed frame: {frame_count} - FPS: {frames_per_sec:.2f}"

            print(f"\r{progress_str}", end="") # \r to overwrite the previous line

    # Release video capture
    cap.release()
    end_time = time.time()
    total_elapsed_time = end_time - start_time

    if total_frames:
        final_message = f"\nExtracted {saved_frame_count} frames to {output_dir} in {total_elapsed_time:.2f} seconds."
    else:
        final_message = f"\nExtracted {saved_frame_count} frames to {output_dir} in {total_elapsed_time:.2f} seconds (Total frames unknown)."

    print(final_message)

def extract_subs_main(video_path, output_dir, all_frames_folder, unique_frames_folder, output_json, output_srt,
                      ocr_region=(481, 585, 740, 630), unique_frame_threshold=78, region_to_compare=(481, 615, 740, 630),
                      resize_scale=0.78, fps=30):
    """
    Extracts subtitles from a video file and processes them into a structured format.
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where the output files will be saved.
        all_frames_folder (str): Directory to save all extracted frames.
        unique_frames_folder (str): Directory to save unique frames.
        output_json (str): Path to the output JSON file.
        output_srt (str): Path to the output SRT file.
        ocr_region (tuple, optional): Region of bounding box intersection where OCRd text is considered to be subtitles.
            Defaults to (481, 585, 740, 630).
        unique_frame_threshold (int, optional): Threshold for determining unique frames (percentage pixel difference
            between frames in the region of interest). Defaults to 78.
        region_to_compare (tuple, optional): Region to compare for unique frames. Defaults to (481, 615, 740, 630).
        resize_scale (float, optional): Scale factor for resizing the subtitle region. Defaults to 0.78.
        fps (int, optional): Frames per second of the video. Defaults to 30.
    Returns:
        str: Processed subtitle string in SRT format.
    Raises:
        ValueError: If the video file cannot be opened or read.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created dir because {output_dir} did not exist")
    else:
        print(f"Didn't create ouput_dir because it already exists: {output_dir}")

    # Set the current working directory to the output directory
    os.chdir(output_dir)

    json_file = output_json # The name of the output json
    # region_to_compare = (481, 615, 740, 630)

    ocr = PaddleOCR(use_angle_cls=False, lang='japan', use_gpu=True)
    
    extract_all_frames(video_path, all_frames_folder)
    find_unique_frames(all_frames_folder, unique_frames_folder, region_to_compare, threshold=unique_frame_threshold)
    
    remove_non_subtitle_frames(unique_frames_folder, region_to_compare, json_file, ocr)
    remove_duplicate_subtitle_frames(unique_frames_folder, region_to_compare, json_file, ocr)
    
    # Calculate the region for the bottom 40% of a 720p image
    # Get the image height and width from the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Error reading frame from video")

    image_height, image_width, _ = frame.shape
    cap.release()
    bottom_percent = int(image_height * 0.4)
    region = (0, image_height - bottom_percent, image_width, image_height)

    # Resize the subtitle region to make OCR work better
    for image in os.listdir(unique_frames_folder):
        if image.lower().endswith(('.png', '.jpg', '.jpeg')):
            resize_region(os.path.join(unique_frames_folder, image), os.path.join(unique_frames_folder, image),
                          region, scale=resize_scale)
    
    ocr_directory_to_json(unique_frames_folder, region_to_compare, ocr_region, json_file, ocr)
    calculate_end_frames(json_file, all_frames_folder)
    merge_duplicate_subtitles(json_file)
    json_to_srt(json_file, output_srt, fps)
    processed_output_srt = f"processed_{output_srt}"
    processed_output_string = process_srt(output_srt)
    write_srt(processed_output_string, processed_output_srt)
    print(f"Finished processing. Output: {processed_output_srt}")
    return processed_output_string

if __name__ == "__main__":
    input_video_file = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\output3.mp4"
    output_folder = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\output_test"
    all_frames_folder = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\output3_test_frames_all"
    unique_frames_folder = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\output3_test_frames_unique"
    extract_subs_main(input_video_file, output_folder, all_frames_folder, unique_frames_folder, "ocr_results4.json", "subtitles4.srt")
    # Tests
    # print(get_text_bounding_box("C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\test_vid5_all_frames\\frame_00000.jpg", 
    #                       ocr=PaddleOCR(use_angle_cls=False, lang='japan', use_gpu=True), region=(481, 585, 740, 630)))
    # unique_frames_folder = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\testvid5testframes"
    # cap = cv2.VideoCapture(input_video_file)
    # if not cap.isOpened():
    #     raise ValueError("Error opening video file")

    # ret, frame = cap.read()
    # if not ret:
    #     raise ValueError("Error reading frame from video")

    # image_height, image_width, _ = frame.shape
    # cap.release()
    # bottom_percent = int(image_height * 0.4)
    # region = (0, image_height - bottom_percent, image_width, image_height)

    # # Resize the subtitle region to make OCR work better
    # for image in os.listdir(unique_frames_folder):
    #     if image.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         resize_region(os.path.join(unique_frames_folder, image), os.path.join(unique_frames_folder, image), region, 0.78)
    
    # ocr_directory_to_json(unique_frames_folder,(481, 585, 740, 630),
    #                       (481, 585, 740, 630), "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\testoutputdelete.json",
    #                       ocr=PaddleOCR(use_angle_cls=False, lang='japan', use_gpu=True))
    
    