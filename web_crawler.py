import argparse
import hashlib
import shutil
import time
import traceback
import warnings
import os

import pdfplumber as pdfplumber
import requests
import yaml
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser

from nipype.scripts.utils import add_args_options
from tqdm import tqdm
import io
import pandas as pd
from playwright.sync_api import sync_playwright

WHITE = "\033[97m"
PURPLE = "\033[35m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


def main():
    config = load_config("config.yaml")
    data_topics = config['data_topics']

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    parser.add_argument("--max_depth", type=int, default=2, help="Maximum depth for the web crawler.")
    parser.add_argument("--max_pages", type=int, default=10, help="Maximum number of pages to crawl.")
    parser.add_argument("--respect_robots_txt", action="store_true", help="Respect robots.txt file.")
    args = parser.parse_args()

    if args.debug:

        print(f"{PURPLE}Running 🕷️  Web Crawler{RESET}")
        print(f"{WHITE}Topic: {args.topic}")
        print(f"Max Depth: {args.max_depth}")
        print(f"Max Pages: {args.max_pages}")
        print(f"Respect Robots.txt: {args.respect_robots_txt}{RESET}")

    crawler = WebCrawler(topic=args.topic, max_depth=args.max_depth, max_pages=args.max_pages, reset=args.reset,
                         respect_robots_txt=args.respect_robots_txt, debug=args.debug)
    crawler.crawl()


class WebCrawler:
    def __init__(self, topic, max_depth: int = 2, max_pages: int = 100, reset: bool = False,
                 respect_robots_txt: bool = True, debug: bool = False):

        config = load_config("config.yaml")
        data_topics = config['data_topics']
        default_topic = config['default_topic']

        selected_topic = topic if topic else default_topic
        topic_config = data_topics[selected_topic]
        topic_dir = topic_config['topic_dir']

        allowed_domains = topic_config['allowed_domains']
        start_urls = topic_config['start_urls']
        non_content_phrases = topic_config['non_content_phrases']

        self.topic_dir = topic_dir
        self.data_dir = f"{topic_dir}/documents"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.url_mapping_file = f"{topic_dir}/url_mapping.yml"
        self.hashed_content_file = f"{topic_dir}/hashed_content.txt"
        self.context_file = f"{topic_dir}/context_data.yaml"

        self.debug = debug
        if debug:
            print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")
        if reset:
            self.reset()
        self.start_urls = start_urls
        self.allowed_domains = allowed_domains
        self.non_content_phrases = non_content_phrases
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.urls_to_visit = [(url, 0) for url in start_urls]
        self.pages_crawled = 0
        self.delay = 1
        self.retry_delay = 60
        self.content_hashes = self.load_hashes()

        self.robot_parsers = {}
        self.respect_robots_txt = respect_robots_txt

    def load_hashes(self):
        if os.path.exists(self.hashed_content_file):
            with open(self.hashed_content_file, 'r') as file:
                return set(line.strip() for line in file)
        return set()

    def save_hashes(self):
        with open(self.hashed_content_file, 'w') as file:
            for content_hash in self.content_hashes:
                file.write(content_hash + '\n')

    def crawl(self):

        term_width = shutil.get_terminal_size((80, 20)).columns - 10
        bar_format = f"{WHITE}🕷️ Crawling  {{l_bar}}{PURPLE}{{bar}}{WHITE}{{r_bar}}{RESET}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            with tqdm(total=self.max_pages, bar_format=bar_format, ncols=term_width, unit="page") as pbar:
                while self.urls_to_visit and self.pages_crawled < self.max_pages:
                    url, depth = self.urls_to_visit.pop(0)
                    if url in self.visited_urls or depth > self.max_depth:
                        continue
                    if not self.is_allowed_by_robots(url) and self.respect_robots_txt:
                        continue
                    if self.debug:
                        print(f"\n{WHITE}🔗  Crawling: {url}{RESET}")

                    self.visited_urls.add(url)
                    try:
                        page = browser.new_page()
                        page.goto(url, wait_until="networkidle", timeout=60000)
                        self.process_page_playwright(page, url, depth)
                        pbar.update(1)
                        time.sleep(self.delay)
                        page.close()
                        self.save_hashes()
                    except Exception as e:
                        if self.debug:
                            warnings.warn(f"Failed to process {url} with Playwright: {e}")
                            traceback.print_exc()
            browser.close()

        self.save_hashes()

    def get_robot_parser(self, url):
        domain = urlparse(url).netloc
        if domain not in self.robot_parsers:
            robots_url = f"https://{domain}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            self.robot_parsers[domain] = rp
        return self.robot_parsers[domain]

    def is_allowed_by_robots(self, url):
        rp = self.get_robot_parser(url)
        return rp.can_fetch("*", url)

    def fetch_page_bs4(self, url):
        """Fetch a page using requests."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response
        except requests.RequestException as e:
            if self.debug:
                warnings.warn(f"Failed to retrieve {url} via requests: {e}", UserWarning)
        return None

    def process_page(self, response, url, depth):
        """Process the response: check for surge protection, handle PDFs or HTML, and extract links."""
        if not self.process_response(response, url, depth):
            return

        soup = BeautifulSoup(response.text, 'html.parser')
        self.parse_links(soup, url, depth)
        self.pages_crawled += 1

    def process_page_playwright(self, page, url, depth):
        """Process a page fetched and rendered by Playwright.

        This method uses Playwright’s own DOM querying to check for surge protection,
        compute content hashes, scrape content, and extract links.
        """
        # Use the inner text of the body to check for surge protection.
        page_text = page.inner_text("body")
        if "surge protection" in page_text.lower():
            print(f"{WHITE}❗ Surge protection triggered. Waiting for {self.retry_delay} seconds.{RESET}")
            time.sleep(self.retry_delay)
            self.urls_to_visit.append((url, depth))
            return

        # Compute content hash based on page text.
        content_hash = hashlib.md5(page_text.encode('utf-8')).hexdigest()
        if content_hash not in self.content_hashes:
            self.content_hashes.add(content_hash)
            if url.lower().endswith('.pdf'):
                # For PDFs, fallback to requests-based processing.
                response = self.fetch_page_bs4(url)
                if response:
                    self.process_pdf(url, response.content)
            else:
                self.scrape_text_from_playwright(page, url)

        # Extract links using Playwright's DOM evaluation.
        links = page.eval_on_selector_all("a", "elements => elements.map(el => el.href)")
        for link in links:
            if not re.match(r'^https?://', link):
                link = urljoin(url, link)
            if not any(allowed in link for allowed in self.allowed_domains):
                continue
            if link not in self.visited_urls:
                self.urls_to_visit.append((link, depth + 1))
        self.pages_crawled += 1

    def scrape_text_from_playwright(self, page, url):
        """
        Scrape the page content using Playwright's DOM querying.
        This method mirrors the logic from the bs4-based scraping,
        but it directly queries the page.
        """
        domain = urlparse(url).netloc
        try:
            filter_lines = self.non_content_phrases[domain]
        except KeyError:
            filter_lines = []

        # Try to locate the main article body.
        article_elem = page.query_selector("div[itemprop='articleBody']")
        if article_elem:
            text = article_elem.inner_text()
        else:
            text = page.inner_text("body")
        # Filter the text by splitting into lines and removing unwanted content.
        lines = text.splitlines()
        if self.debug:
            print("Number of lines:", len(lines))
            print("Lines to filter:", len(filter_lines))
        filtered_lines = [line for line in lines if len(line.split()) > 2 and
                          not any(phrase in line for phrase in filter_lines)]
        if self.debug:
            print("Number of filtered lines:", len(filtered_lines))
        filtered_text = "\n".join(filtered_lines)

        # if self.debug:
            # print("Filtered text:", filtered_text)

        if filtered_text.strip():
            filename = self.url_to_filename(url, "txt")
            self.update_url_mapping(filename, url)
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(filtered_text)

    def create_fake_response(self, content):
        """Create a fake response object to mimic requests.Response."""
        class FakeResponse:
            pass

        fake = FakeResponse()
        fake.text = content
        fake.content = content.encode('utf-8')
        fake.status_code = 200
        return fake

    def process_response(self, response, url, depth):
        if "surge protection" in response.text.lower():
            print(f"{WHITE}❗ Surge protection triggered. Waiting for {self.retry_delay} seconds.{RESET}")
            time.sleep(self.retry_delay)
            self.urls_to_visit.append((url, depth))  # Re-add URL to retry later
            return False
        content_hash = hashlib.md5(str(response.content).encode('utf-8')).hexdigest()
        if content_hash not in self.content_hashes:
            self.content_hashes.add(content_hash)
            if url.lower().endswith('.pdf'):
                self.process_pdf(url, response.content)
            else:
                self.process_html(url, response)

    def process_pdf(self, url, content):
        self.scrape_text_from_pdf(url, content)

    def scrape_text_from_pdf(self, url, pdf_content):
        filename = self.url_to_filename(url, "txt")
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text = str()
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() + "\n"
                text = f"{text}\n{page_text}"
            with open(f"{self.data_dir}/{filename}", 'w', encoding='utf-8') as file:
                file.write(page_text)
            self.update_url_mapping(filename, url)

    def process_html(self, url, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        self.scrape_text_from_html(url, soup)

    def scrape_text_from_html(self, url, soup):
        # Extract the article body content
        main_content = soup.find('div', itemprop='articleBody')

        # Check if the article body exists
        if not main_content:
            # Remove menus and other non-content elements
            for element in soup(['footer', 'nav', 'aside', 'form', 'noscript']):
                element.decompose()

            # Extract the main content (attempt to filter out non-content sections)
            main_content = soup.find_all(['article', 'main', 'section']) or soup

        # Get text from the main content
        text = " ".join([content.get_text(separator=' ') for content in main_content])

        # Filter out lines that do not contain actual content and non-content phrases
        content_lines = [line for line in text.splitlines()
                         if len(line.split()) > 2 and
                         not any(phrase in line for phrase in self.non_content_phrases)]

        # Join the filtered lines
        filtered_text = '\n'.join(content_lines)

        # Check if the filtered text is empty
        if filtered_text.strip():
            filename = self.url_to_filename(url, "txt")
            self.update_url_mapping(filename, url)
            html_file_path = f"{self.data_dir}/{filename}"

            os.makedirs(self.data_dir, exist_ok=True)
            with open(html_file_path, 'w', encoding='utf-8') as file:
                file.write(filtered_text)

    @staticmethod
    def url_to_filename(url: str, file_format="", no_type=False, name_extension=""):
        # Replace illegal characters with underscores
        filename = re.sub(r'[<>:"./\\|?*]', '_', url[8:])
        if no_type:
            return f"{filename}{name_extension}"
        return f"{filename}{name_extension}.{file_format}"

    def update_url_mapping(self, filename: str, url: str):
        if not os.path.exists(self.url_mapping_file):
            url_mapping = {'documents': {}}
        else:
            with open(self.url_mapping_file, 'r') as file:
                url_mapping = yaml.safe_load(file) or {'documents': {}}

        url_mapping['documents'][filename] = url

        with open(self.url_mapping_file, 'w') as file:
            yaml.safe_dump(url_mapping, file)

    def parse_links(self, soup, base_url, depth):
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not re.match(r'^https?://', href):
                href = urljoin(base_url, href)
            if not any(allowed_domain in href for allowed_domain in self.allowed_domains):
               continue
            if href not in self.visited_urls:
                self.urls_to_visit.append((href, depth + 1))
        return links

    def update_context_data(self, filename, url, context):
        if not os.path.exists(self.context_file):
            context_data = {'files': {}}
        else:
            with open(self.context_file, 'r') as file:
                context_data = yaml.safe_load(file) or {'files': {}}

        context_data['files'][filename] = {'url': url, 'context': context}

        with open(self.context_file, 'w') as file:
            yaml.safe_dump(context_data, file)

    def reset(self):
        print(f"{WHITE}✨  Clearing Datastorage{RESET}")
        if os.path.isfile(self.url_mapping_file):
            os.remove(self.url_mapping_file)

        if os.path.isfile(self.hashed_content_file):
            os.remove(self.hashed_content_file)

        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir)


if __name__ == "__main__":
    main()
