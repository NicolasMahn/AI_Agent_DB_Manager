import argparse
import hashlib
import shutil
import time
import warnings
import os

import fitz
import pdfplumber as pdfplumber
import requests
import yaml
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse, urljoin
from tqdm import tqdm
from PIL import Image
import io
import pandas as pd

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
    default_topic = config['default_topic']

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    parser.add_argument("--max_depth", type=int, default=2, help="Maximum depth for the web crawler.")
    parser.add_argument("--max_pages", type=int, default=10, help="Maximum number of pages to crawl.")
    args = parser.parse_args()

    selected_topic = args.topic if args.topic else default_topic
    topic_config = data_topics[selected_topic]
    topic_dir = topic_config['topic_dir']

    allowed_domains = topic_config['allowed_domains']
    start_urls = topic_config['start_urls']
    non_content_phrases = topic_config['non_content_phrases']
    black_listed_imgs = topic_config['black_listed_imgs']
    crawler = WebCrawler(start_urls=start_urls, allowed_domains=allowed_domains, topic_dir=topic_dir,
                         non_content_phrases=non_content_phrases, black_listed_imgs=black_listed_imgs,
                         max_depth=args.max_depth, max_pages=args.max_pages, reset=args.reset, debug=args.debug)
    crawler.crawl()


class WebCrawler:
    def __init__(self, start_urls: list[str], allowed_domains: list[str], topic_dir: str,
                 non_content_phrases: list[str], black_listed_imgs: list[str],
                 max_depth: int = 2, max_pages: int = 100, reset: bool = False, debug: bool = False):
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
        self.black_listed_imgs = black_listed_imgs
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.urls_to_visit = [(url, 0) for url in start_urls]
        self.pages_crawled = 0
        self.delay = 1
        self.retry_delay = 60
        self.content_hashes = self.load_hashes()

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
        # Custom bar format with color codes
        bar_format = f"{WHITE}🕷️ Crawling  {{l_bar}}{PURPLE}{{bar}}{WHITE}{{r_bar}}{RESET}"
        with tqdm(total=self.max_pages, bar_format=bar_format, ncols=shutil.get_terminal_size((80, 20)).columns - 10,
                  unit="page") as pbar:
            while self.urls_to_visit and self.pages_crawled < self.max_pages:
                url, depth = self.urls_to_visit.pop(0)
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                self.visited_urls.add(url)
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        self.process_response(response, url, depth)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        self.parse_links(soup, url, depth)
                        self.pages_crawled += 1
                        pbar.update(1)
                    time.sleep(self.delay)  # Delay between requests
                    self.save_hashes()
                except requests.RequestException as e:
                    if self.debug:
                        warnings.warn(f"Failed to retrieve {url}: {e}", UserWarning)
            self.save_hashes()

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
        self.scrape_images_from_pdf(url, content)
        self.scrape_tables_from_pdf(url, content)

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

    def scrape_images_from_pdf(self, url, pdf_content):
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        for i in range(len(pdf_document)):
            page = pdf_document.load_page(i)
            page_text = page.get_text("text")
            for j, img in enumerate(pdf_document.get_page_images(i)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                img_bytes = base_image["image"]
                if len(img_bytes) > 20480:
                    img_format = self.get_image_format(img_bytes)
                    img_filename = self.url_to_filename(url, file_format=f"{img_format.lower()}",
                                                        name_extension=f"_page_{i + 1}_image_{j + 1}")
                    self.process_images(img_bytes, url, img_filename, context=page_text)

    def scrape_tables_from_pdf(self, url, pdf_content):
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                page_text = page.extract_text()
                for j, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        table_filename = self.url_to_filename(url, no_type=True,
                                                              name_extension=f"_page_{i + 1}_table_{j + 1}.csv")
                        table_path = f"{self.data_dir}/{table_filename}"
                        df.to_csv(table_path, index=False)
                        self.update_url_mapping(table_filename, url)
                        self.update_context_data(table_filename, url, page_text)

    def process_html(self, url, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        self.scrape_tables_from_html(url, soup)
        self.scrape_text_from_html(url, soup)
        self.scrape_images_from_html(url, soup)

    def scrape_text_from_html(self, url, soup):

        code = soup.find('textarea', id="read-only-cursor-text-area", attrs={
            'data-testid': "read-only-cursor-text-area",
            'aria-label': "file content",
            'aria-readonly': "true",
            'inputmode': "none",
            'tabindex': "0",
            'aria-multiline': "true",
            'aria-haspopup': "false",
            'data-gramm': "false",
            'data-gramm_editor': "false",
            'data-enable-grammarly': "false",
            'spellcheck': "false",
            'autocorrect': "off",
            'autocapitalize': "off",
            'autocomplete': "off",
            'data-ms-editor': "false",
            'class': "react-blob-textarea react-blob-print-hide"
        })

        if code:
            filename_tag = soup.find('div', {'data-testid': 'breadcrumbs-filename'})
            filename = filename_tag.find('h1').get_text() if filename_tag else 'unknown_filename'

            # Extract branch
            branch_tag = soup.find('svg', {'class': 'octicon-git-branch'})
            branch = branch_tag.find_next_sibling('span').get_text() if branch_tag else 'unknown_branch'

            filtered_text = f"Filename: {filename}\nBranch: {branch}\n\n```\n{code.get_text(separator=' ')}\n```\n"

        else:
            # Extract the article body content
            main_content = soup.find('div', itemprop='articleBody')

            # Check if the article body exists
            if not main_content:
                # Remove menus and other non-content elements
                for element in soup(['footer', 'nav', 'aside', 'form', 'noscript', 'table']):
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

    def scrape_tables_from_html(self, url, soup):
        os.makedirs(self.data_dir, exist_ok=True)
        tables = soup.find_all('table')
        for i, table in enumerate(tables):
            try:
                tables_list = pd.read_html(io.StringIO(str(table)))
                if tables_list:
                    df = tables_list[0]
                    table_filename = self.url_to_filename(url, file_format="csv")
                    table_path = f"{self.data_dir}/{table_filename}"
                    df.to_csv(table_path, index=False)
                    surrounding_text = ' '.join(table.find_parent().stripped_strings)
                    self.update_url_mapping(table_filename, url)
                    self.update_context_data(table_filename, url, surrounding_text)
                else:
                    raise ValueError("No tables found")
            except ValueError as e:
                if self.debug:
                    warnings.warn(f"Failed to parse table at {url}: {e}", UserWarning)

    def scrape_images_from_html(self, base_url, soup):
        for img in soup.find_all('img'):
            context = f"alt: {img.get('alt', '')}, " \
                      f"caption: {img.find_next('figcaption').text if img.find_next('figcaption') else ''}, " \
                      f"surrounding_text: {' '.join(img.find_parent().stripped_strings)}, " \
                      f"base_url: {base_url}"
            img_url = img.get('src')
            if not img_url:
                continue
            img_url = urljoin(base_url, img_url)
            try:
                img_response = requests.get(img_url)
                if img_response.status_code == 200 and len(img_response.content) > 20480:
                    if self.is_descriptive_image(img_url, context):
                        self.process_images(img_response.content, img_url, context=context)
            except requests.RequestException as e:
                if self.debug:
                    warnings.warn(f"Failed to retrieve image {img_url}: {e}", UserWarning)

    def is_descriptive_image(self, img_url, context):
        lower_url = img_url.lower()
        if any(keyword in lower_url for keyword in ['logo', 'icon', 'favicon', 'sprite', 'banner', 'button']):
            return False
        if 'logo' in context.lower() or 'icon' in context.lower() or 'button' in context.lower():
            return False
        if img_url in self.black_listed_imgs:
            return False
        return True

    def process_images(self, img_data, img_url, img_filename=None, context=""):
        if "logo" not in img_url.lower() and "icon" not in img_url.lower():
            img_hash = hashlib.md5(str(img_data).encode('utf-8')).hexdigest()
            if img_hash not in self.content_hashes:
                self.content_hashes.add(img_hash)
                if img_filename is None:
                    img_format = self.get_image_format(img_data)
                    img_filename = self.url_to_filename(img_url, img_format.lower())

                with open(f"{self.data_dir}/{img_filename}", 'wb') as img_file:
                    img_file.write(img_data)
                    self.update_url_mapping(img_filename, img_url)
                    self.update_context_data(img_filename, img_url, context)

    @staticmethod
    def get_image_format(image_bytes):
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return img.format
        except IOError:
            return "svg"

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
            href_domain = urlparse(href).netloc
            if not any(allowed_domain in href_domain for allowed_domain in self.allowed_domains):
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
