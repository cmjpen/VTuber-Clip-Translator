import re

def format_srt(srt_text):
    """
    Formats the given SRT (SubRip Subtitle) text by ensuring there is an empty line
    between each subtitle block.
    Args:
        srt_text (str): The SRT content as a string.
    Returns:
        str: The formatted SRT content with empty lines between subtitle blocks.
    """
    # Split the SRT content into lines
    lines = srt_text.strip().split('\n')
    
    # Result list to collect formatted lines
    formatted_lines = []
    
    for i in range(len(lines)):
        formatted_lines.append(lines[i])
        
        # Check if the current line is subtitle text (not index or timestamp)
        # If the next line is an index (digit) or end of the file, insert an empty line
        if i + 1 < len(lines):
            if lines[i + 1].isdigit():
                formatted_lines.append('')
    
    return '\n'.join(formatted_lines)

def process_srt(file_path):
    """
    Processes an SRT (SubRip Subtitle) file to merge consecutive duplicate subtitles.
    
    This function performs the following operations:
    1. Parses the SRT file into subtitle entries.
    2. Merges consecutive entries with identical text after removing punctuation, numbers, and specific characters.
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

    def remove_specific_chars(text):
        specific_chars = 'やゃつっ'
        return text.translate(str.maketrans('', '', specific_chars)).strip()

    def merge_entries(entries):
        processed = []
        i = 0
        while i < len(entries):
            current = entries[i]
            j = i + 1
            while j < len(entries):
                next_entry = entries[j]
                # Clean text by removing punctuation, specific characters, then digits
                current_comp = remove_numbers(remove_specific_chars(clean_text(current['text'])))
                next_comp = remove_numbers(remove_specific_chars(clean_text(next_entry['text'])))
                # Only merge if both are non-empty and equal after removing numbers and specific characters
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

def split_srt_batches(srt: str, batch_size: int = 30) -> list[dict]:
    """
    Splits an SRT string into batches of batch_size groups.
    
    For each batch (except the first), the function computes a context string
    consisting of all text lines before the batch from the original SRT,
    omitting the index and timing lines.
    
    Parameters:
      srt (str): The full SRT content as a single string.
      batch_size (int): The number of subtitle blocks (each block is 3 lines) 
                        to include in each srt_string.
    
    Returns:
      list[dict]: A list of dictionaries. Each dictionary contains:
         - 'srt_string': A batch of blocks (as a string) from the SRT.
         - 'context_before': The text-only context (as a string) preceding that batch.
    """
    # Remove all blank lines
    cleaned_srt = "\n".join(line for line in srt.splitlines() if line.strip() != "")
    # Split the cleaned SRT into lines
    lines = cleaned_srt.splitlines()
    print("Total lines:", len(lines))
    
    # Group the SRT lines into blocks of 3 (index, timing, text)
    sub_line_groups = [lines[i:i+3] for i in range(0, len(lines), 3)]
    
    # Warn if the number of lines is not a multiple of 3
    if len(lines) % 3 != 0:
        print("Warning: The number of lines is not a multiple of 3. "
              "Check that the SRT is formatted as groups of 3 lines (index, timing, text).")
    
    results = []
    total_groups = len(sub_line_groups)
    # Split the groups into batches
    batches = [sub_line_groups[i:i+batch_size] for i in range(0, total_groups, batch_size)]
    
    # Keep track of text lines from all groups processed so far (for context)
    previous_texts = []
    
    for batch in batches:
        # Build the srt_string for the current batch:
        # For each block, join its three lines with newlines.
        srt_string = "\n".join("\n".join(block) for block in batch)
        
        # For the context, join all text lines from earlier blocks (if any)
        context_before = "\n".join(previous_texts) if previous_texts else ""
        
        results.append({
            'srt_string': srt_string,
            'context_before': context_before
        })
        
        # Update previous_texts with the text line (assumed to be the third line) from each block
        for block in batch:
            if len(block) >= 3:
                previous_texts.append(block[2])
    
    return results

def combine_srt_strings(srt_list: list[str]) -> str:
    """
    Combines a list of SRT strings in order by appending one after the other,
    making sure there is at least one new line between each addition.
    
    Parameters:
      srt_list (list[str]): A list of SRT strings in the desired order.
    
    Returns:
      str: A single SRT string resulting from the combination.
    """
    # Clean each SRT string by stripping leading/trailing whitespace
    cleaned_srts = [srt.strip() for srt in srt_list if srt.strip()]
    # Join the cleaned SRT strings with two newlines between each.
    # (This guarantees a blank line between each SRT addition.)
    combined_srt = "\n\n".join(cleaned_srts)
    return combined_srt

def read_srt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def write_srt(content, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    preview = content[:8] + '...' if len(content) >= 8 else content
    print(f"Contents ({preview}) written to {file_path}")

def extract_srt_from_text(text):
    """
    Extracts valid SRT content from a given string.

    Args:
        text (str): The input string containing SRT content mixed with other text.

    Returns:
        str: Cleaned SRT content.
    """
    srt_pattern = re.compile(
        r'(\d+\s*\n'                        # Match the subtitle index number
        r'\d{2}:\d{2}:\d{2},\d{3} --> '      # Match the start timestamp
        r'\d{2}:\d{2}:\d{2},\d{3}\s*\n'       # Match the end timestamp
        r'(?:.+\n?)+?(?=\n\d+\s*\n|$))',      # Match subtitle text until the next index or end-of-string
        re.MULTILINE
    )

    # Find all SRT blocks and join them
    srt_content = '\n'.join(match.group(0).strip() for match in srt_pattern.finditer(text))
    
    return srt_content