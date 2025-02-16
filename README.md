# VTuber Clip Translator

This project automates the process of downloading VTuber clips from YouTube, extracting hardcoded subtitles, filling in gaps in the subtitles using audio transcription, and translating the subtitles from Japanese to English. Transcription and translation accuracy is significantly improved due to extracting hardcoded subtitles - as these are manually transcribed and usually error-free. Therefore, this project will work best with Japanese VTuber clips that have hardcoded subtitles and minimal other text on screen. For example: youtu.be/G3_M-h7u3iU

## Limitations and Known Issues

*   This project is currently configured to only support translations for one speaker. Translating multiple speakers would require identifying speakers by subtitle color (during subtitle extraction) or speaker diarization (during Whisper gap filling). This information would then need to be added to the transcription before each line, and prompts adjusted accordingly.
*   OCR with PaddleOCR is not perfect and can sometimes produce inaccurate transcriptions of hardcoded subtitles. While Google Gemini is used to correct obvious OCR errors, some errors may still remain.
*   Clips with significant text on screen in the subtitle area will likely produce poor or unusable results. For example: https://www.youtube.com/watch?v=wGGfH2Iiydw&t=64s. This could be improved by detecting the subtitle region (perhaps by running PaddleOCR on a few random frames before full extraction) and cropping the image accordingly.
*   Currently, a region of interest must be specified for unique frame detection.
*   Gap filling with Whisper can introduce erroneous transcriptions if the VTuber is screaming, coughing, etc. A "transcription correction" step using Gemini to identify and remove poor audio transcriptions could address this, but is not yet implemented.

## Features

1.  **Download YouTube Videos:** Uses `yt-dlp` to download YouTube videos.
2.  **Extract Subtitles:** Extracts hardcoded subtitles from video frames using OCR (Optical Character Recognition) with `PaddleOCR`.
3.  **Fill Gaps in Subtitles:** Identifies gaps in the subtitles and fills them using audio transcription (OpenAI Whisper).
4.  **Translate Subtitles:** Translates the extracted and filled subtitles from Japanese to English using OpenAI's GPT-4o model.
5.  **Scrape Webpages:** Scrapes the Wikipedia page of the featured VTuber for additional context to improve translation accuracy.

## More Detailed Overview

The process is broken down into several stages:

**1. Downloading:**

*   The clip is downloaded using `yt-dlp`, which returns the title, description, and video file.
*   The title and description are passed to Gemini Flash 2.0 Exp to identify the featured VTuber, original stream URL, and return this information in JSON format.
*   The VTuber's Wikipedia page is scraped to obtain nicknames, fan nicknames, and gender to improve transcription and translation accuracy.

**2. Subtitle Extraction:**

*   Unique frames are identified based on a percentage pixel difference threshold within a specified region of interest (a small box where subtitles usually appear). This region should be manually configured depending on the clip or channel.
*   The `unique_frame_threshold` parameter controls the minimum number of unique frames before skipping subtitle lines (tested to be between 73 and 80).
*   Unique frames are cropped to remove the top 60% (non-subtitle area) to improve OCR speed.
*   The `resize_scale` parameter controls frame resizing for better OCR capture (around 0.78 for most videos, but clip-dependent).
*   Blank and duplicate subtitle frames are removed using `remove_non_subtitle_frames` and `remove_duplicate_subtitle_frames` to increase OCR speed and reduce errors.
*   OCR results are stored in a JSON file with subtitle line timings.
*   Duplicate subtitles are merged.
*   The JSON is converted to an SRT file.
*   A video summary (identifying places, companies, websites, and people) is generated using Gemini and prepended to the transcription and translation prompts.
*   The transcription is corrected for OCR errors using Gemini.

**3. Gap Filling:**

*   Gaps in the SRT file exceeding a threshold are identified.
*   OpenAI Whisper transcribes audio within these gaps.

**4. Translation:**

*   A translation prompt is constructed, including the VTuber's name, gender, nicknames, fan nicknames, and the Japanese video summary.
*   The SRT is split into batches of 20 lines for translation.
*   Previous lines (without timings) are prepended to each batch for context.
*   The translated SRT is extracted from the Gemini response and written to a file.

## Installation

1.  Clone the repository:

    ```sh
    git clone [https://github.com/yourusername/vtuber-clip-translator.git](https://github.com/yourusername/vtuber-clip-translator.git)
    cd vtuber-clip-translator
    ```

2.  Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3.  Download `ffmpeg` and add it to your system path. Instructions for this can be found here: https://phoenixnap.com/kb/ffmpeg-windows

## Usage

1.  Set up your API keys for OpenAI and Gemini by setting the environment variables `open_ai_key` and `gemini_key`.

2.  Run the main script:

    ```sh
    python main.py
    ```

## Project Structure

```
├── main.py                 # Main script to run the translation process
├── README.md               # Project documentation
├── requirements.txt        # List of dependencies
├── vtuber_clip_translator/ # Source code
│   ├── download_video.py   # Module for downloading YouTube videos
│   ├── scrape_webpage.py   # Module for extracting subtitles using OCR
│   ├── whisper_gap_filler.py # Module for filling subtitle gaps using Whisper
│   ├── subtitle_extractor.py # Module for translating subtitles
│   ├── __init__.py         # Makes the directory a package
│   └── utils/              # Utility functions
│       ├── srt_processing.py # Module for processing SRT files
│       └── __init__.py     # Makes the utils directory a package
```
