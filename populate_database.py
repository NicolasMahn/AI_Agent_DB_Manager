import argparse
import json
import os
import shutil

from embedding_function import openai_ef
from scrt import CHROMADB_HOST, CHROMADB_PORT
from util import load_config

import yaml
from tqdm import tqdm

import chromadb

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"


def populate_db():


    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", help="Select the data topic.")
    args = parser.parse_args()

    config = load_config()
    data_topics = config['data_topics']
    default_topic = config['default_topic']
    topic = args.topic

    topic = topic if topic else default_topic

    if topic == "all":
        topic = list(data_topics.keys())[0]
        print(f"{WHITE}📄  Processing: {topic}{RESET}")
        populater = DatabaseManager(topic, args.reset, args.debug)
        for topic in data_topics.keys():
            if not topic == list(data_topics.keys())[0]:
                print(f"{WHITE}📄  Processing: {topic}{RESET}")
                populater.load_collection(topic)
            populater.save_data()
    else:
        print(f"{WHITE}📄  Processing: {topic}{RESET}")
        populater = DatabaseManager(topic, args.reset, args.debug)
        populater.save_data()


class DatabaseManager:
    def __init__(self, topic: str = None, reset: bool = False, debug: bool = False, separate_in_chunks: bool = True,
                 chunk_separation_warning: bool = True, chunk_size: int = 8192):

        self.data = None
        self.predefined_ids = None
        self.url_mapping = None
        self.context_data = None
        self.data_file = None
        self.predefined_ids_file = None
        self.context_file = None
        self.url_mapping_file = None
        self.topic_dir = None
        self.collection = None
        self.topic_config = None
        self.topic = None

        self.chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

        self.config = load_config()
        self.data_topics = self.config['data_topics']
        self.default_topic = self.config['default_topic']

        self.separate_in_chunks = separate_in_chunks
        self.chunk_separation_warning = chunk_separation_warning
        self.chunk_size = chunk_size
        self.debug = debug

        self.load_collection(topic, reset)

    def load_collection(self, topic: str, reset: bool=False):
        self.topic = topic
        self.topic_config = self.data_topics[self.topic]
        self.collection = self.chroma_client.get_or_create_collection(name=self.topic, embedding_function=openai_ef)

        self.topic_dir =self.topic_config['topic_dir']
        self.url_mapping_file = f"{self.topic_dir}/url_mapping.yml"
        self.context_file = f"{self.topic_dir}/context_data.yaml"
        self.predefined_ids_file = f"{self.topic_dir}/ids.yaml"
        self.data_file = f"{self.topic_dir}/data.json"

        self.context_data = self.open_context_data()

        if self.debug:
            print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")
            print("Topic:", self.topic)
        if reset:
            print(f"{WHITE}✨  Clearing Database{RESET}")
            self.clear_collection()
        self.url_mapping = self.load_url_mapping()

        self.predefined_ids = self.get_predefined_ids()

    def clear_collection(self):
        self.chroma_client.delete_collection(name=self.topic)
        self.collection = self.chroma_client.get_or_create_collection(name=self.topic, embedding_function=openai_ef)

    def save_data(self):
        self.data = self.load_data_from_file()

        # Custom bar format with color codes
        bar_format = f"{WHITE}⌛  Adding Text    {{l_bar}}{BLUE}{{bar}}{WHITE}{{r_bar}}{RESET}"

        ncols = shutil.get_terminal_size((80, 20)).columns - 10

        with tqdm(total=len(self.data), bar_format=bar_format, unit="document", ncols=ncols) as pbar:
            for data_item in self.data:
                doc_chunks = self.process_data(data_item)
                for chunk in doc_chunks:
                    self.add_to_chroma(chunk)
                pbar.update(1)

    def load_data_from_file(self):
        with open(self.data_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    def load_url_mapping(self):
        try:
            with open(self.url_mapping_file, 'r') as file:
                return yaml.safe_load(file).get('documents', {})
        except FileNotFoundError:
            return {}

    def unique(self, metadata: dict):
        # Calculate Page IDs.
        metadata = self.calculate_chunk_id(metadata)

        # Add or Update the documents.
        existing_items = self.collection.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids.yaml"])

        return metadata["id"] not in existing_ids

    def calculate_chunk_id(self, metadata: dict):
        source = metadata.get("doc_name", None)
        url = metadata.get("url", None)
        chunk_number = metadata.get("chunk_number", None)
        if source in self.predefined_ids:
            chunk_id = self.predefined_ids[source]
        elif url in self.predefined_ids:
            chunk_id = self.predefined_ids[url]
        else:
            chunk_id = f"{chunk_number}|{url}|{source}"
        metadata["id"] = chunk_id
        return metadata

    def add_to_chroma(self, doc):
        self.collection.upsert(documents=[doc["content"]], ids=[doc["id"]], metadatas=[doc["metadata"]])

    def get_context_from_filename(self, filename):
        basename = os.path.basename(filename)

        if basename in self.context_data['files']:
            return self.context_data['files'][basename].get('context', None)
        return None

    def filter_non_image_documents_for_url(self, specific_url):
        file_names = []
        for file_name, url in self.url_mapping.items():
            if specific_url in url and not url.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg')):
                file_names.append(file_name)
        return file_names

    def open_context_data(self):
        try:
            with open(self.context_file, 'r') as file:
                context_data = yaml.safe_load(file) or {'files': {}}
        except FileNotFoundError:
            context_data = None
        return context_data

    def process_data(self, data_item: dict):
        content = data_item.pop("content")

        if self.separate_in_chunks:
            content_chunks = self.split_text_into_chunks(content)
        else:
            content_chunks = [content]
        documents = []
        for i, chunk in enumerate(content_chunks):
            metadata = data_item
            metadata["id"] = f"{metadata['pdf_name']}_{metadata['title']}_{i}"
            if metadata is None:
                print(f"{PINK}⚠️  Error: Metadata empty (line 247){RESET}")
            elif self.debug:
                print()
                print(f"{GREEN}📄  Processing: {self.topic} {RESET}")
                print("Metadata:", metadata)

            doc = {"content": chunk, "id": metadata["id"], "metadata": metadata }
            documents.append(doc)
        return documents

    def split_text_into_chunks(self, text):
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        if len(chunks) > 1 and self.debug:
            print(f"\n{PINK}⚠️  Warning: A text was split into {len(chunks)} chunks.{RESET}")
        for chunk in chunks:
            yield chunk

    def get_predefined_ids(self):
        try:
            with open(self.predefined_ids_file, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}


if __name__ == "__main__":
    populate_db()