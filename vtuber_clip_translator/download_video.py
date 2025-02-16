import os
import yt_dlp
import subprocess
import sys
from moviepy.editor import VideoFileClip

def convert_video_to_720p_30fps_h264(input_file, output_file):
    """
    Convert a video file to 720p (1280x720) resolution at 30fps using the h264 codec.

    Parameters:
        input_file (str): Path to the input video file.
        output_file (str): Path where the converted video will be saved.
    """
    # Load the original video
    clip = VideoFileClip(input_file)
    
    # Resize the clip to exactly 720p (1280x720)
    # This will force the video to the specified dimensions.
    # If you want to preserve the aspect ratio, you might instead specify just the height,
    # e.g., clip.resize(height=720)
    clip_resized = clip.resize(newsize=(1280, 720))
    
    # Write the video file with 30fps and using the h264 codec (libx264)
    clip_resized.write_videofile(
        output_file,
        fps=30,
        codec="libx264"
    )
    
    # Close clips to free resources
    clip.close()
    clip_resized.close()

def update_yt_dlp():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"], check=True)

def download_youtube_video(url, output_path):
    """
    Downloads a YouTube video using yt-dlp and saves it to the specified output path.
    Also returns the title and description of the video.

    Args:
        url (str): The URL of the YouTube video to download.
        output_path (str): The file path where the downloaded video will be saved.
        ffmpeg_path (str): The file path to the ffmpeg executable.

    Returns:
        dict: A dictionary containing the video title and description.
    """
    update_yt_dlp()
    ydl_opts = {
        'format': 'bestvideo[height<=720][fps<=30]+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': output_path,
        # 'ffmpeg_location': ffmpeg_path,  # specify the path to ffmpeg
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get('title', 'Unknown Title')
        video_description = info_dict.get('description', 'No Description')
    
    return {
        'title': video_title,
        'description': video_description
    }

# Example usage:
if __name__ == '__main__':
    ffmpeg_path = "ffmpeg.exe"
    video_url = "https://www.youtube.com/watch?v=_-9HzbeYo-M"  # Replace with your URL
    download_folder = "C:\\Users\\wbscr\\Desktop\\More Desktop\\sub_ex"  # Replace with your directory path
    filename = "output3.mp4"
    download_path = os.path.join(download_folder, filename)
    video_info = download_youtube_video(video_url, download_path, ffmpeg_path)
    print(video_info)
    # cvtd_file = convert_video_to_720p_30fps_h264(download_path, os.path.join(download_folder, "cvtd_"+filename))
