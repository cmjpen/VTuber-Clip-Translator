import re
from datetime import datetime, timedelta
import os
import google.generativeai as genai
from google.generativeai.types import content_types
from collections.abc import Iterable
import time
from moviepy.editor import AudioFileClip
import soundfile as sf
import numpy as np
import json
from openai import OpenAI
from typing import Sequence

def convert_mp4_to_wav(input_file: str, output_file: str):
    """
    Converts an MP4 file to a WAV file.

    Args:
        input_file (str): Path to the input MP4 file.
        output_file (str): Path to save the output WAV file.

    Example:
        convert_mp4_to_wav("input.mp4", "output.wav")
    """

    # Load the audio from the MP4 file
    audio_clip = AudioFileClip(input_file)
    # Write the audio to a WAV file
    audio_clip.write_audiofile(output_file, codec="pcm_s16le")
    # Close the clip to release resources
    audio_clip.close()

def split_audio_file(input_file, max_chunk_size_mb=25):
    """
    Split an audio file into chunks without interrupting dialogues,
    ensuring each chunk is under 25MB, and return a dictionary of chunk paths and timestamps.

    Args:
        input_file (str): Path to the input audio file
        max_chunk_size_mb (int): Maximum chunk size in megabytes

    Returns:
        dict: A dictionary where keys are chunk filenames and values are (start_time, end_time) tuples
    """
    data, sample_rate = sf.read(input_file)
    
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    bytes_per_sample = data.dtype.itemsize
    max_samples = int((max_chunk_size_mb * 1024 * 1024) / bytes_per_sample)
    
    silence_threshold = 0.1 * np.max(np.abs(data))
    chunk_splits = [0]
    current_start = 0
    
    for i in range(len(data)):
        if i - current_start >= max_samples:
            silence_search_end = min(i + int(sample_rate * 5), len(data))
            silence_segment = data[i:silence_search_end]
            silence_indices = np.where(np.abs(silence_segment) < silence_threshold)[0]
            
            if len(silence_indices) > 0:
                split_point = i + silence_indices[0]
                chunk_splits.append(split_point)
                current_start = split_point
    
    chunk_splits.append(len(data))
    
    chunk_dict = {}
    
    for j in range(len(chunk_splits) - 1):
        start = chunk_splits[j]
        end = chunk_splits[j+1]
        
        chunk = data[start:end]
        output_filename = f"chunk_{j+1}.wav"
        sf.write(output_filename, chunk, sample_rate)
        print("Saved "+output_filename)
        
        start_time = start / sample_rate
        end_time = end / sample_rate
        
        chunk_dict[output_filename] = (start_time, end_time)
    
    return chunk_dict

def analyze_srt_gaps(srt_filepath, gap_threshold_seconds=2):
    """
    Analyzes an SRT file for gaps in dialogue (subtitles) and returns gap lengths
    and subtitle entries.

    A gap in dialogue is defined as a period where there are no subtitles
    for longer than the specified gap_threshold_seconds.

    Args:
        srt_filepath (str): The path to the SRT file.
        gap_threshold_seconds (int, float): The minimum duration in seconds
            for a period without subtitles to be considered a gap.
            Defaults to 2 seconds.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries, where each dictionary represents a gap
                    and contains:
                      'start_timestamp': The timestamp (in SRT time format HH:MM:SS,mmm)
                                         indicating the start of the gap.
                      'duration_seconds': The duration of the gap in seconds (float).
            - list: A list of dictionaries, where each dictionary is a subtitle entry
                    and contains:
                      'start': datetime object, start time of subtitle
                      'end': datetime object, end time of subtitle
                      'text': str, the subtitle text
              Returns an empty list for gaps if no gaps are found, or an error dictionary.
    """

    gap_info = []
    previous_end_time = None
    subtitle_entries = []
    srt_content_str = "" # Initialize to store SRT content for later use

    try:
        with open(srt_filepath, 'r', encoding='utf-8') as srt_file:
            srt_content_str = srt_file.read() # Read SRT content into string
    except FileNotFoundError:
        return {"error": "FileNotFoundError", "message": f"SRT file not found at path: {srt_filepath}"}, []
    except Exception as e:
        return {"error": "FileError", "message": f"Error reading SRT file: {e}"}, []


    # Regex to parse SRT entry (number, timecode, text - capturing groups for timecode and text)
    srt_entry_regex = re.compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?=\n\d+|\Z)", re.DOTALL)

    matches = srt_entry_regex.findall(srt_content_str)

    if not matches:
        return {"warning": "NoSubtitlesFound", "message": "No valid subtitle entries found in the SRT file."}, []

    for match in matches:
        start_time_str, end_time_str, subtitle_text = match[1], match[2], match[3] # Extract subtitle text
        try:
            start_time = datetime.strptime(start_time_str, '%H:%M:%S,%f')
            end_time = datetime.strptime(end_time_str, '%H:%M:%S,%f')
        except ValueError:
            return {"error": "FormatError", "message": f"Invalid time format in SRT entry: {start_time_str} --> {end_time_str}"}, []


        subtitle_entries.append({
            'start': start_time,
            'end': end_time,
            'text': subtitle_text.strip() # Store subtitle text, removing leading/trailing whitespace
        })


    if not subtitle_entries:
        return {"warning": "NoSubtitlesProcessed", "message": "No subtitle entries could be processed."}, []


    for i in range(len(subtitle_entries)):
        current_entry = subtitle_entries[i]
        current_start_time = current_entry['start']
        current_end_time = current_entry['end']

        if i > 0: # Compare with the previous entry if it exists
            previous_entry = subtitle_entries[i-1]
            previous_end = previous_entry['end']

            time_diff = current_start_time - previous_end

            if time_diff > timedelta(seconds=gap_threshold_seconds):
                gap_start_timestamp_dt = previous_end # Start of the gap is the end of the last subtitle
                gap_start_timestamp_str = gap_start_timestamp_dt.strftime('%H:%M:%S,%f')[:-3] # Format back to SRT timecode
                gap_duration_seconds = time_diff.total_seconds() # Calculate gap duration in seconds

                gap_info.append({
                    'start_timestamp': gap_start_timestamp_str,
                    'end_timestamp': current_start_time.strftime('%H:%M:%S,%f')[:-3],
                    'duration_seconds': gap_duration_seconds
                })

    print("Gap Info:", gap_info)
    print("Subtitle Entries:", subtitle_entries)
    print("SRT Content String:", srt_content_str)
    return gap_info, subtitle_entries, srt_content_str # Return gap info, subtitle entries, and srt_content

def get_text_before_gap(gap_data, subtitle_entries):
    """
    Extracts the subtitle text preceding each gap.

    Args:
        gap_data (list): List of gap dictionaries from analyze_srt_gaps.
        subtitle_entries (list): List of subtitle entry dictionaries from analyze_srt_gaps.

    Returns:
        list: A list of strings, where each string is the concatenated subtitle
              text (lines separated by '\n') that appears before the corresponding gap.
              Returns an empty list if there are no gaps or subtitle entries.
    """
    text_segments_before_gap = []

    if not gap_data or not subtitle_entries:
        return []

    for gap in gap_data:
        gap_start_timestamp_str = gap['start_timestamp']
        gap_start_datetime = datetime.strptime(gap_start_timestamp_str, '%H:%M:%S,%f')

        preceding_text_lines = []
        for sub_entry in subtitle_entries:
            if sub_entry['end'] <= gap_start_datetime: # Subtitle ends before or at the gap start
                preceding_text_lines.append(sub_entry['text'])

        text_segments_before_gap.append("\n".join(preceding_text_lines)) # Join lines with newline

    return text_segments_before_gap

def upload_file_to_gemini(filename):
    try:
        print(genai.list_files())
        audio_file = genai.get_file(name=os.path.basename(filename))
        print(f"Audio file already exists: {audio_file.uri}")
    except:
        print("No audio file found. Uploading audio file...")
        print(f"Uploading file...")
        audio_file = genai.upload_file(path=filename)
        print(f"Completed upload: {audio_file.uri}")
        print(audio_file.name)

        # Poll API for audio processing status
        while audio_file.state.name == "PROCESSING":
            print('Waiting for audio to be processed.')
            time.sleep(5)
            audio_file = genai.get_file(audio_file.name)

        if audio_file.state.name == "FAILED":
            raise ValueError(audio_file.state.name)
        print(f'Audio processing complete: ' + audio_file.uri)
    return audio_file

def gemini_transcribe_audio(prompt, audio_file, model="gemini-2.0-flash"):
    model = genai.GenerativeModel(f'models/{model}')
    response = model.generate_content([prompt, audio_file])
    return response.text

def openai_transcribe_audio(audio_file_path, client):
    audio_file= open(audio_file_path, "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="verbose_json",
    timestamp_granularities=["segment"],
    )
    return transcription.segments

def get_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data['api_key']

def cut_audio_file(audio_file, output_file, start_time, end_time):
    audio_clip = AudioFileClip(audio_file)
    subclip = audio_clip.subclip(start_time, end_time)
    subclip.write_audiofile(output_file, codec="pcm_s16le")
    audio_clip.close()
    subclip.close()
    return output_file

def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds,milliseconds
    hours, minutes, seconds_milliseconds = timestamp.split(':')
    seconds, milliseconds = seconds_milliseconds.split(',')

    # Convert all parts to seconds
    total_seconds = (int(hours) * 3600) + (int(minutes) * 60) + int(seconds) + (int(milliseconds) / 1000)
    return total_seconds

def convert_segment_to_dict(segment):
    """
    TranscriptionSegment オブジェクトを {'start', 'end', 'text'} 形式の辞書に変換する関数。

    Args:
        segment: TranscriptionSegment オブジェクト

    Returns:
        dict: {'start', 'end', 'text'} キーを持つ辞書。
              変換に失敗した場合は None を返す。
    """
    try:
        segment_dict = {
            'start': segment.start,
            'end': segment.end,
            'text': segment.text.strip() # テキストの前後の空白を削除
        }
        return segment_dict
    except AttributeError as e:
        print(f"Error: TranscriptionSegment オブジェクトに 'start', 'end', 'text' 属性が存在しません: {e}")
        return None
    except Exception as e:
        print(f"Error: TranscriptionSegment オブジェクトの変換中に予期せぬエラーが発生しました: {e}")
        return None

def process_gaps_main(srt_file_path, video_file_path, audio_file_path, openai_client, gap_threshold=2):
    """
    Processes gaps in the subtitle file and fills them using Gemini transcription service.
    Args:
        srt_file_path (str): Path to the subtitle (.srt) file.
        video_file_path (str): Path to the video file.
        audio_file_path (str): Path to the audio file.
        gap_threshold (int, optional): The minimum duration (in seconds) of gaps to be considered significant. Defaults to 2.
    Returns:
        list: A list of dictionaries containing gap information and their transcriptions.
    The function performs the following steps:
    1. Analyzes the subtitle file to find gaps longer than the specified threshold.
    2. Prints information about the detected gaps.
    3. Converts the video file to an audio file.
    4. For each gap, extracts the corresponding audio segment and uploads it to Gemini for transcription.
    5. Prints and stores the transcriptions for each gap.
    6. Removes the temporary audio subclips and the main audio file after processing.
    7. Returns the gap data with transcriptions.
    """
    result = analyze_srt_gaps(srt_file_path, gap_threshold) # Get subtitle_entries and srt_content back
    if len(result) == 3:
        gap_data, subtitle_entries, srt_content = result
    else:
        print("No gaps")
        gap_data = None

    if gap_data:
        print(f"Gaps in dialogue (longer than {gap_threshold} seconds) found at:")
        text_before_gaps = get_text_before_gap(gap_data, subtitle_entries) # Call new function

        for i, gap in enumerate(gap_data):
            print(f"\nGap {i+1}:")
            print(f"  Timestamp: {gap['start_timestamp']} to {gap['end_timestamp']}, Duration: {gap['duration_seconds']:.2f} seconds")
            # print(f"  Text before gap:\n---\n{text_before_gaps[i]}\n---") # Print text before gap
            gap["text_before_gap"] = text_before_gaps[i]
    else:
        print(f"No significant gaps in dialogue (longer than {gap_threshold} seconds) found.")
        return None

    convert_mp4_to_wav(video_file_path, audio_file_path)
    
    for i, gap in enumerate(gap_data):
        subclip_name = f"gap_{i+1}"
        start_time = timestamp_to_seconds(gap['start_timestamp'])
        end_time = timestamp_to_seconds(gap['end_timestamp'])
        sub_clip = cut_audio_file(audio_file_path, f"{subclip_name}.wav", start_time, end_time)
        
        whisper_transcription = openai_transcribe_audio(sub_clip, openai_client)
        print(whisper_transcription)
        transcription = []
        for line in whisper_transcription:
            transcription.append(convert_segment_to_dict(line))

        # convert the whisper transcription to json
        print(f"Transcription for gap {i+1} ({start_time} -> {end_time}):\n{transcription}")
        # transcription = extract_json_from_text(transcription) # Not needed for Whisper
        print(f"Extracted transcription for gap {i+1}:\n{transcription}")
        gap["transcription_data"] = transcription
    
    # Remove the subclips after processing
    # for i in range(len(gap_data)):
    #     subclip_name = f"gap_{i+1}.wav"
    #     if os.path.exists(subclip_name):
    #         os.remove(subclip_name)
    #         time.sleep(0.1)
    #         print(f"Removed {subclip_name}")

    # Remove the main audio file after processing
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
        print(f"Removed audio file {audio_file_path}")

    print("Finished processing all gaps.")
    gap_data = [correct_timestamps(gap) for gap in gap_data]
    print(gap_data)
    for index, transcription in enumerate(gap_data):
        srt_string = gap_data_to_srt(transcription)
        transcription["srt_string"] = srt_string
        file_name = f"transcription_{index + 1}.srt"
        # write_srt(srt_string, file_name)
    return gap_data

def gap_data_to_srt(subtitle_data):
    """
    指定された形式の字幕データリストをSRT形式の文字列に変換する関数

    Args:
        subtitle_data (list): 辞書型の字幕データリスト。各辞書は 'start', 'end', 'text' キーを持つ。
            例: [{'start': '00:00:00,298', 'end': '00:00:03,648', 'text': '字幕テキスト'}]

    Returns:
        str: SRT形式に変換された字幕文字列。
    """
    srt_string = ""
    for index, subtitle in enumerate(subtitle_data["transcription_data"]):
        # SRT形式では字幕エントリは1から始まる連番で始まる
        entry_number = index + 1
        start_time = next(value for key, value in subtitle.items() if "abs_start" in key)
        end_time = next(value for key, value in subtitle.items() if "abs_end" in key)
        text = next(value for key, value in subtitle.items() if "text" in key)

        # SRT形式のタイムコードを作成
        timecode = f"{start_time} --> {end_time}"

        # SRTエントリを作成し、文字列に追加
        srt_entry = f"{entry_number}\n{timecode}\n{text}\n\n"
        srt_string += srt_entry

    return srt_string

def write_srt(content, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def correct_timestamps(gap_data):
    start_timestamp = gap_data["start_timestamp"]
    abs_start_datetime = datetime.strptime(start_timestamp, '%H:%M:%S,%f')
    transcription = gap_data["transcription_data"]
    def parse_time(time_str):
        if isinstance(time_str, (int, float)):
            return datetime.strptime('00:00:00,000', '%H:%M:%S,%f') + timedelta(seconds=time_str)
        for fmt in ('%M:%S,%f', '%H:%M:%S,%f', '%M:%S:%f', '%H:%M:%S:%f'):
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                pass
        raise ValueError(f"Time format for '{time_str}' is not recognized")
    for line in transcription:
        start = line["start"]
        end = line["end"]
        start_timedelta = parse_time(start) - datetime.strptime('00:00:00,000', '%H:%M:%S,%f')
        end_timedelta = parse_time(end) - datetime.strptime('00:00:00,000', '%H:%M:%S,%f')
        upd_start_td = abs_start_datetime + start_timedelta
        upd_end_td = abs_start_datetime + end_timedelta
        # cvt back to string
        abs_start = upd_start_td.strftime('%H:%M:%S,%f')[:-3]
        abs_end = upd_end_td.strftime('%H:%M:%S,%f')[:-3]
        line["abs_start"] = abs_start
        line["abs_end"] = abs_end
    return gap_data


# def merge_srts(base_srt: str, srts_to_merge: list[str]) -> str:
#     """
#     Merges multiple SRT files into a base SRT file.

#     Parameters:
#     base_srt (str): The file path of the base SRT file.
#     srts_to_merge (list of str): A list of file paths of the SRT files to merge into the base SRT file.

#     Returns:
#     str: The content of the merged SRT file.
#     """

#     pass
def parse_time(time_str: str) -> timedelta:
    """
    Parse a time string which can be in either '%H:%M:%S,%f' or '%M:%S,%f' format
    into a timedelta object.
    """
    try:
        dt = datetime.strptime(time_str, '%H:%M:%S,%f')
    except ValueError:
        dt = datetime.strptime(time_str, '%M:%S,%f')
    return timedelta(hours=dt.hour, minutes=dt.minute,
                     seconds=dt.second, microseconds=dt.microsecond)

def format_time(delta: timedelta) -> str:
    """
    Format a timedelta into an SRT time string in HH:MM:SS,mmm format.
    """
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # Note: delta.microseconds is the remainder within the current second.
    milliseconds = int(delta.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def parse_srt(srt_text: str) -> list:
    """
    Parse an SRT-formatted string into a list of subtitle entries.
    Each entry is a dictionary with keys: 'start', 'end', and 'text'.
    """
    entries = []
    # Split the SRT file into blocks separated by blank lines.
    blocks = srt_text.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue  # Not a valid subtitle block.
        
        # The first line is the numeric index (we won’t really need it later)
        try:
            idx = int(lines[0].strip())
        except ValueError:
            idx = None
        
        # The second line is the time range.
        time_line = lines[1].strip()
        if ' --> ' not in time_line:
            continue  # Skip malformed block.
        
        start_str, end_str = time_line.split(' --> ')
        start_delta = parse_time(start_str.strip())
        end_delta = parse_time(end_str.strip())
        
        # The rest of the lines form the subtitle text.
        text = "\n".join(lines[2:])
        
        entries.append({
            'index': idx,
            'start': start_delta,
            'end': end_delta,
            'text': text
        })
    return entries

def merge_srts(base_srt: str, srts_to_merge: list[str]) -> str:
    """
    Merge a base SRT and a list of additional SRTs.
    
    Parameters:
      - base_srt: a string containing the base SRT.
      - srts_to_merge: a list of strings, each an SRT to be merged into the base.
    
    Returns:
      - A single merged SRT string with all subtitle entries sorted by start time.
    """
    # Parse the base SRT into entries.
    all_entries = parse_srt(base_srt)
    
    # Parse each additional SRT and add its entries.
    for srt in srts_to_merge:
        all_entries.extend(parse_srt(srt))
    
    # Sort the combined entries by start time.
    all_entries.sort(key=lambda entry: entry['start'])
    
    # Reassemble the sorted entries into SRT format.
    merged_lines = []
    for new_index, entry in enumerate(all_entries, start=1):
        start_formatted = format_time(entry['start'])
        end_formatted = format_time(entry['end'])
        
        merged_lines.append(str(new_index))
        merged_lines.append(f"{start_formatted} --> {end_formatted}")
        merged_lines.append(entry['text'])
        merged_lines.append("")  # Blank line after each block.
    
    return "\n".join(merged_lines)

def merge_main(base_srt_string, gap_data):
    srts_to_add = []
    for gap in gap_data:
        srts_to_add.append(gap["srt_string"])
    
    # with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
    #     base_srt = srt_file.read() # Read SRT content into string
    merged = merge_srts(base_srt_string, srts_to_add)
    print(merged)
    # write_srt(merged, output_srt_filename)
    return merged

if __name__ == '__main__':
    srt_file_path = "processed_corrected_subtitles_gpt4o_2.srt"  
    video_file_path = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\noel_test.mp4"
    audio_file_path = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex\\clip_audio.wav"
    gap_threshold = 1.5 
    gap_data = process_gaps_main(srt_file_path, video_file_path, audio_file_path, gap_threshold=gap_threshold)
    # gap_data = [correct_timestamps(gap) for gap in gap_data]
    # print(gap_data)
    # for index, transcription in enumerate(gap_data):
    #     srt_string = gap_data_to_srt(transcription)
    #     transcription["srt_string"] = srt_string
    #     file_name = f"transcription_{index + 1}.srt"
    #     write_srt(srt_string, file_name)
    print(gap_data)
    
    # Fill in the gaps
    with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
        base_srt = srt_file.read() # Read SRT content into string
    merged = merge_main(base_srt, gap_data)
    write_srt(merged, "merged_subtitles.srt")
    
    
    