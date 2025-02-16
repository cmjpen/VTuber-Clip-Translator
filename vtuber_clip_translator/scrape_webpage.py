from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, NoAlertPresentException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
from bs4 import BeautifulSoup
import html2text
import urllib.parse
import markdownify
import re
import time
import yt_dlp
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException

def accept_cookies(driver):
    try:
        # Wait for cookie popup to load
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//*[contains(translate(text(), 'ACCEPT', 'accept'), 'accept') or contains(text(), '承諾')]"))
        )
        
        # Click using common selectors (prioritized order)
        selectors = [
            'button#cookie-accept',
            'button#accept-cookies',
            'button[aria-label*="cookie"]',
            'button:contains("Accept")',  # Requires jQuery syntax (see note below)
            '//button[contains(., "Accept")]',  # XPath
            '//div[contains(text(), "承諾")]',  # XPath for Japanese text
            '//div[@data-tracking-opt-in-accept="true"]'  # XPath for specific attribute
        ]
        
        for selector in selectors:
            try:
                if "//" in selector:
                    btn = driver.find_element(By.XPATH, selector)
                else:
                    btn = driver.find_element(By.CSS_SELECTOR, selector)
                btn.click()
                print("Clicked cookie accept button")
                return True
            except:
                continue
                
    except Exception as e:
        print(f"No cookie popup found: {str(e)}")
        return False

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # Run in headless mode

    # Get the base path from ChromeDriverManager
    base_path = ChromeDriverManager().install()
    
    # Manually construct the path to the chromedriver executable
    chromedriver_path = os.path.join(os.path.dirname(base_path), 'chromedriver.exe')
    
    # Ensure the constructed path is correct
    if not os.path.exists(chromedriver_path):
        raise FileNotFoundError(f"chromedriver executable not found at {chromedriver_path}")

    # Create the WebDriver service
    service = Service(chromedriver_path)
    
    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_window_size(1920, 1080)
    # Prepare to get the filename of file downloaded later
    params = {'behavior': 'allow', 'downloadPath': os.getcwd()}
    driver.execute_cdp_cmd('Page.setDownloadBehavior', params)

    return driver

def slow_scroll_to_bottom(driver, scroll_step=1080, delay=0.25, max_attempts=50):
    last_height = driver.execute_script("return document.body.scrollHeight")
    attempts = 0
    initial_infobox_count = len(driver.find_elements(By.CSS_SELECTOR, ".portable-infobox")) # Hypothetical class

    while attempts < max_attempts:
        driver.execute_script(f"window.scrollBy(0, {scroll_step});")
        time.sleep(delay)

        driver.execute_script("return document.documentElement.outerHTML;")

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            attempts += 1
            if attempts >= max_attempts:
                break
        else:
            attempts = 0
        last_height = new_height
        attempts += 1
    time.sleep(5)

def scrape_page(url):
    driver = setup_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        accept_cookies(driver)
        # time.sleep(.25)
        page_content = driver.page_source
    finally:
        driver.quit()
    return page_content

def scrape_youtube_video_page(url):
    driver = setup_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        accept_cookies(driver)
        # time.sleep(.25)
        page_content = driver.page_source
    finally:
        driver.quit()
    return page_content

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    return ' '.join(text.split())

def html_to_markdown(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = False  # Keep hyperlinks
    h.ignore_images = False  # Keep images
    h.ignore_emphasis = True  # Ignore emphasis like bold and italics
    h.body_width = 0  # Do not wrap text at a specific width
    return h.handle(html_content).strip()

def extract_and_convert(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Target the main content area.  This is crucial and will likely
    # need adjustment depending on the wiki's HTML structure.
    # Inspect the HTML source of the wiki page to find the div,
    # section, or other element that contains the main text.
    main_content = soup.find('div', {'id': 'mw-content-text'})  # Example: adjust this!

    if main_content is None:
        main_content = soup.find('div', {'class': 'mw-body'})
    if main_content is None:
        main_content = soup.find('div', {'id': 'bodyContent'})


    if not main_content:  # Handle the case where the target element isn't found
        print("Warning: Main content element not found.  Returning all text.")
        text = soup.get_text()
        return ' '.join(text.split())

    # Convert the *selected* content to Markdown
    md_content = markdownify.markdownify(str(main_content))

    return md_content.strip()


def decode_all_encoded_strings(markdown_text):
    # Regex to find URL-encoded substrings (percent-encoded patterns)
    pattern = re.compile(r'(?:%[0-9A-Fa-f]{2})+')

    # Function to decode each matched URL-encoded string
    def decode_match(match):
        encoded_str = match.group(0)
        return urllib.parse.unquote(encoded_str)

    # Replace all encoded substrings with their decoded version
    return pattern.sub(decode_match, markdown_text)

def write_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

def get_video_info(url):
    # Configure yt-dlp to avoid downloading the video
    ydl_opts = {
        'quiet': True,        # Suppress CLI output
        'skip_download': True # Do not download the video
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            # Extract metadata
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'No title available'),
                'description': info.get('description', 'No description available')
            }
        except Exception as e:
            print(f"Error: {e}")
            return None

if __name__ == "__main__":
    # # Scrape the Wikipedia page for Shirogane Noel
    # wiki_html = scrape_page(pediapage)
    # markdown_content = extract_and_convert(wiki_html)
    # markdown_content = decode_all_encoded_strings(markdown_content)
    # write_to_file(markdown_content, 'shirogane_noel_wiki.md')

    # Get the youtube video title and description
    clip_url = "https://www.youtube.com/watch?v=G3_M-h7u3iU&t=310s&pp=ygUf5YiH44KK5oqc44GNc2FpIOODjuOCqOODq-Wkp-WIhg%3D%3D"
    print(get_video_info(clip_url))
    # Scrape the Virtual YouTuber Fandom page