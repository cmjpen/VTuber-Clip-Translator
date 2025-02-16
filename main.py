import os
import re
import json
import time
import datetime

from openai import OpenAI
import google.generativeai as genai

from vtuber_clip_translator.scrape_webpage import (
    scrape_page, extract_and_convert, decode_all_encoded_strings, write_to_file, get_video_info
)
from vtuber_clip_translator.whisper_gap_filler import (
    process_gaps_main, merge_main, convert_mp4_to_wav
)
from vtuber_clip_translator.subtitle_extractor import extract_subs_main
from vtuber_clip_translator.download_video import download_youtube_video
from vtuber_clip_translator.utils.srt_processing import (
    extract_srt_from_text, format_srt, process_srt,
    split_srt_batches, combine_srt_strings, write_srt, read_srt_file 
)

def extract_json_from_text(text):
    """
    Extracts the first JSON object or array found in the given text.

    Args:
        text (str): The input string containing JSON data.

    Returns:
        dict or list: Parsed JSON object or array as a Python dictionary or list if found and valid.
        None: If no valid JSON object or array is found.
    """
    match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Invalid JSON format.")
            return None
    else:
        print("No JSON found in the text.")
        return None

def ask_gemini(prompt, model_id="gemini-2.0-flash-exp"):
    """Asks Gemini a prompt and returns the response."""

    model = genai.GenerativeModel(model_name=model_id)  # No tools needed for simple prompts
    response = model.generate_content(prompt, request_options={"timeout": 60}) # Adjust timeout as needed.

    # Print or return the text of the response.
    if response.candidates:
        return response.candidates[0].content.parts[0].text
    else:
        return "No response from Gemini."

def ask_openai(prompt, client):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return chat_completion.choices[0].message.content

def download_clip(clip_url, video_output_path,):
    video_info_dict = download_youtube_video(clip_url, video_output_path)
    clip_id = re.search(r"watch\?v=([^&]+)", clip_url).group(1)
    return video_info_dict, clip_id, video_output_path

def get_vtuber_name(video_info_dict, gem_model_id):
    vid_inf_prompt = f"""Get the following information from the provided youtube video title and description and return it in JSON format:\n
            ['vtuber_name','original_stream_url'] \n
            Video title and description:\n{video_info_dict['title']}\n\n{video_info_dict['description']}"""
    video_info = ask_gemini(vid_inf_prompt, model_id=gem_model_id)
    video_info = extract_json_from_text(video_info)
    vtuber_name = video_info['vtuber_name']
    og_stream_url = video_info['original_stream_url']
    print("VTuber name:", vtuber_name, "\nOriginal Stream URL:", og_stream_url)
    return vtuber_name, og_stream_url

def scrape_wiki(vtuber_name, gem_model_id):
    wikipedia_url = f"http://ja.wikipedia.org/wiki/{vtuber_name}"
    page_content = scrape_page(wikipedia_url)
    markdown_content = extract_and_convert(page_content)
    markdown_content = decode_all_encoded_strings(markdown_content)

    extraction_prompt = f"""VTuber{vtuber_name}のWikiページをMarkdown形式でスクレイピングしました。
                次の情報を取得してみてください：['愛称', 'ファンの愛称','性別']。結果はJSON形式で返してください。\n
                {markdown_content}
            """
    # Extract info from the wiki page with gemini
    nicknames_dict = extract_json_from_text(ask_gemini(extraction_prompt, gem_model_id))
    return nicknames_dict

def correct_transcription(video_summary, vtuber_name, nicknames_dict, sub_extractor_output, srt_filename, gem_model_id):
    nickname = nicknames_dict['愛称']
    prompt2 = f"""Take a look at the following Japanese srt. There may be some lines which have errors due to OCR. If there's a line that
          seems like it could be an OCR error, correct the error so that it makes sense. Return the corrected srt.
          Context: This is from a VTuber clip from {vtuber_name}'s channel. 
          Their nickname is {nickname}. Here's a short summary of the video:\n {video_summary}
          :\n{sub_extractor_output}"""
    
    corrected_srt_file = f"corrected_{srt_filename}"
    processed_srt_file = f"processed_corrected_{srt_filename}.srt"

    # Get Gemini to fix OCR errors. Extract the srt from the response, format it so blank lines exists so that process_srt can work
    corrected_srt_string = format_srt(extract_srt_from_text(ask_gemini(prompt2, gem_model_id))) 
    print("Corrected SRT extracted:\n", corrected_srt_string)
    write_srt(corrected_srt_string, corrected_srt_file)
    
    # Remove duplicate lines with process_srt again (Now that lines are corrected, more duplicates can be detected)
    processed_subs_string = process_srt(corrected_srt_file)
    print("Processed SRT:\n", processed_subs_string)
    write_srt(processed_subs_string, processed_srt_file)

    return processed_subs_string, processed_srt_file

def fill_gaps(processed_srt_file, processed_subs_string, video_file_path, audio_file_path, openai_client, gap_threshold=2.0):
    gap_data = process_gaps_main(processed_srt_file, video_file_path, audio_file_path,
                                 openai_client, gap_threshold)
    if gap_data:
        print("Gaps present.")
        merged_srt = merge_main(processed_subs_string, gap_data)
        write_srt(merged_srt, f"merged_subtitles_gpt4o_whisper_{video_name}.srt")
    else:
        print("Gaps not present.")
        merged_srt = processed_subs_string
    
    merged_srt = extract_srt_from_text(merged_srt)
    return merged_srt

def translate_srt(nicknames_dict, vtuber_name, video_summary, srt_content, openai_client):
    nickname = nicknames_dict['愛称']
    fan_nickname = nicknames_dict['ファンの愛称']
    gender = nicknames_dict['性別']
    tl_prompt = f"""Translate the following Japanese srt from Japanese to English. 
            Make sure to use the existing subtitle timings for the translated text.
            Context: This is from a VTuber clip from {vtuber_name}'s channel. Their gender is {gender}.
            Their nickname is {nickname}, which they may use when referring to themselves. 
            If this is the case, translate it to the appropriate first person pronoun in English.
            The nickname for {vtuber_name}'s fans is {fan_nickname}, which can be translated as "you guys" or whatever is appropriate for the situation.
            Here's a brief summary of the video in Japanese:\n{video_summary}
            :\n{srt_content}"""

    print("Translation prompt:\n", tl_prompt)

    # Split up the translation request into batches to make sure all lines are translated
    batches = split_srt_batches(srt_content, batch_size=20)
    print("Length of split tl batches:", len(batches))
    translated_batches = []
    for batch in batches:
        line_group = batch["srt_string"]
        if batch["context_before"] == "":
            tl_prompt = f"""Translate the following Japanese srt from Japanese to English. 
                Make sure to use the existing subtitle timings for the translated text.
                Context: This is from a VTuber clip from {vtuber_name}'s channel. Their gender is {gender}.
                Their nickname is {nickname}, which they may use when referring to themselves. 
                If this is the case, translate it to the appropriate first person pronoun in English.
                The nickname for {vtuber_name}'s fans is {fan_nickname}, which can be translated as "you guys" or whatever is appropriate for the situation.
                Here's a brief summary of the video in Japanese:\n{video_summary}
                SRT to translate:\n{line_group}"""
        else:
            prev_context = batch["context_before"]
            tl_prompt = f"""Translate the following Japanese srt from Japanese to English. 
                Make sure to use the existing subtitle timings for the translated text.
                Context: This is from a VTuber clip from {vtuber_name}'s channel. Their gender is {gender}.
                Their nickname is {nickname}, which they may use when referring to themselves. 
                If this is the case, translate it to the appropriate first person pronoun in English.
                The nickname for {vtuber_name}'s fans is {fan_nickname}, which can be translated as "you guys" or whatever is appropriate for the situation.
                Here's a brief summary of the video in Japanese:\n{video_summary}
                Here is the dialogue which happened before the SRT for context: \n{prev_context}
                \n\nSRT to translate:
                :\n{line_group}"""
        # Request translation from OpenAI
        translated_srt = ask_openai(tl_prompt, openai_client)
        translated_srt = extract_srt_from_text(translated_srt)
        translated_batches.append(translated_srt)
    
    # Recombine batch translated srts
    translated_srt = combine_srt_strings(translated_batches)
    translated_srt = format_srt(re.sub(r'```\s*', '', translated_srt))
    print("Translated SRT:\n", translated_srt)
    return translated_srt

def create_summary(video_info_dict, nicknames_dict, vtuber_name, sub_extractor_output, gem_model_id):
    nickname = nicknames_dict['愛称']
    fan_nickname = nicknames_dict['ファンの愛称']
    gender = nicknames_dict['性別']
    # Get a written summary of the video
    prompt3 = f"""以下の日本語のSRTを見て、それが何についてのものか、物語の主なポイント、特定の地名、会社名、登場人物、ゲーム名、サイト名などを含めて120語以内の日本語で教えてください。
        コンテキスト：これは{vtuber_name}のチャンネルからのVTuber切り抜きです。切り抜きのタイトルは「{video_info_dict['title']}」です。
        愛称が{nickname}で、一人称として使う場合もあるかもしれないです。性別が{gender}でファンの愛称は{fan_nickname}です。
          :\n{sub_extractor_output}"""
    
    video_summary = ask_gemini(prompt3, gem_model_id)
    return video_summary

def main(openai_client, gem_model_id, clip_url, video_output_path, output_folder,
         all_frames_folder, unique_frames_folder, output_json, video_name, audio_file_path):
    """
    Main function to process a Japanese VTuber clip, extract hardcoded subtitles, perform wiki and video scraping
    to retrieve contextual information, perform whisper audio transcription on any gaps in the SRT, merge into one transcription SRT,
    then translate the transcription SRT with contextual information via GPT-4o.
    
    Args:
        openai_client (OpenAI): OpenAI client instance.
        gem_model_id (str): Model ID for Gemini.
        clip_url (str): URL of the video clip to be processed.
        video_output_path (str): Path to save the downloaded video.
        output_folder (str): Folder to save the output files.
        all_frames_folder (str): Folder to save all frames extracted from the video.
        unique_frames_folder (str): Folder to save unique frames extracted from the video.
        output_json (str): Path to save the output JSON file.
        video_name (str): Name of the video (without file extension).
        audio_file_path (str): Path to the audio file.
    
    Returns:
        str: Path to the final translated SRT file.
    """
    #
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # CLIP DOWNLOAD
    video_info_dict, clip_id, video_output_path = download_clip(clip_url, video_output_path)

    # Get vtuber_name from yt-dlp
    vtuber_name, og_stream_url = get_vtuber_name(video_info_dict, gem_model_id)

    # Get "愛称" (nickname), "ファンの愛称" (fan nickname), '性別' (gender) from wiki scraping. Extract info from the wiki page with Gemini
    nicknames_dict = scrape_wiki(vtuber_name, gem_model_id)
    print("Nicknames Dict:\n", json.dumps(nicknames_dict, indent=2, ensure_ascii=False))

    # SUBTITLE EXTRACTION
    sub_extractor_output = extract_subs_main(video_output_path, output_folder, all_frames_folder, unique_frames_folder,
                                             output_json, output_srt, unique_frame_threshold=77, resize_scale=.78)
    # sub_extractor_output = read_srt_file(os.path.join(output_folder, output_srt))

    # TRANSCRIPTION
    video_summary = create_summary(video_info_dict, nicknames_dict, vtuber_name, sub_extractor_output, gem_model_id=gem_model_id)
    print("Video Summary:\n", video_summary)
    
    # Correct OCR errors, passing the summary so that names do not get accidentally corrected
    srt_filename = video_name+".srt"
    processed_subs_string, processed_srt_file = correct_transcription(video_summary, vtuber_name, nicknames_dict,
                                                                      sub_extractor_output, srt_filename, gem_model_id)

    # GAP FILLING
    srt_content = fill_gaps(processed_srt_file, processed_subs_string, video_output_path,
                           audio_file_path, openai_client)
    
    # TRANSLATION
    translated_srt = translate_srt(nicknames_dict, vtuber_name, video_summary, srt_content, openai_client)

    # Final output
    output_srt_filename = f"translated_subtitles_gpt4o_{clip_id}_{current_time}.srt"
    write_srt(translated_srt, output_srt_filename)
    return output_srt_filename

if __name__ == "__main__":
    start_time = time.time()
    
    delete_files = True
    clip_url = "https://www.youtube.com/watch?v=AlmAKuTB0cc"
    video_name = "test_vid8"
    output_folder = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex"
    video_output_path = os.path.join(output_folder,video_name+".mp4")
    output_json = video_name+".json"
    output_srt = video_name+"_extracted_subs.srt"
    all_frames_folder = os.path.join(output_folder,video_name+"_all_frames")
    unique_frames_folder = os.path.join(output_folder,video_name+"_unique_frames")
    audio_file_name = video_name + ".wav"
    
    GEMINI_API_KEY = os.environ.get('gemini_key')
    OPENAI_API_KEY = os.environ.get('open_ai_key')

    # Initialize OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
    )
    
    gem_model_id = "gemini-2.0-flash-exp"
    genai.configure(api_key=GEMINI_API_KEY)

    audio_file_path = os.path.join(output_folder, audio_file_name)
    translated_srt_file = main(client, gem_model_id, clip_url, video_output_path, output_folder,
                                all_frames_folder, unique_frames_folder, output_json, video_name, audio_file_path)
    if delete_files:
        os.remove(video_output_path)
        os.rmdir(all_frames_folder)
        os.rmdir(unique_frames_folder)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds")
