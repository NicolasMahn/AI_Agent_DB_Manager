import argparse
import os

from scrt import CHROMADB_HOST, CHROMADB_PORT
import chromadb
from embedding_function import openai_ef

import llm_api_wrapper

import util


PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""

WHITE = "\033[97m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[32m"
RESET = "\033[0m"


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    config = util.load_config()
    data_topics = config['data_topics']
    default_topic = config['default_topic']
    parser.add_argument("--query_text", type=str, help="The query text.")
    parser.add_argument("--debug", action="store_true", help="Additional print statements")
    parser.add_argument("--topic", choices=data_topics.keys(), help="Select the data topic.")
    args = parser.parse_args()
    if args.debug:
        print(f"{ORANGE}⭕  DEBUG Mode Active{RESET}")

    if not args.query_text:
        query_text = "What is the capital of France?"
        print(f"{WHITE}🔍  Using default Test query: {query_text}{RESET}")
    else:
        query_text = args.query_text

    if not args.topic:
        topic = default_topic
        if default_topic == "all":
            topic = list(data_topics.keys())[0]
        print(f"{WHITE}📄  Using default Test topic: {topic}{RESET}")
    else:
        topic = args.topic

    response_text, _, _ = query_rag(query_text, topic, debug=args.debug)

    print(f"{WHITE}{response_text}{RESET}")
    print()


def query_rag(query_text: str, topic: str, unique_role: str=None, unique_prompt_template: str=None,
              debug: bool = False, n_results: int = 2):
    # Prepare the DB.
    chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

    # Create or get the test collection
    collection = chroma_client.get_or_create_collection(name=topic, embedding_function=openai_ef)

    # Search the DB.
    results = collection.query(
    query_texts=[query_text],  # Chroma will embed this for you
    n_results=n_results  # how many results to return
    )

    ids = results['ids'][0]
    page_contents = results['documents'][0]
    metadatas = results['metadatas'][0]
    context_texts = []
    for i in range(len(ids)):
        pdf_name = metadatas[i].get("pdf_name", None)
        title = metadatas[i].get("title", None)
        page_content = page_contents[i]
        # if type == "image":
        context_texts.append(f"[source: {pdf_name}, {title}]\n{page_content}")

    context_text = "\n\n---\n\n".join(context_texts)
    # sources = [doc.metadata.get("id", None) for doc, _score in results]
    if debug:
        print("Prompt:\n", query_text)
        # print("Retrieved Summarize:\n", results)
        print("Context:\n", context_text)
        print("Metadata:\n", metadatas)
        print("\n")

    prompt_template = PROMPT_TEMPLATE if not unique_prompt_template else unique_prompt_template

    prompt = prompt_template.format(context=context_text, question=query_text)

    role = "Provide accurate and concise answers based solely on the given context." if not unique_role else unique_role
    response_text = llm_api_wrapper.basic_prompt(prompt, role=role, temperature= 0.2, model="default")

    return response_text, context_text, metadatas


def load_raw_document_content(doc_name: str, data_dir: str):
    file_path = os.path.join(data_dir, doc_name)
    if file_path.endswith('.txt') or file_path.endswith('.csv'):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    return "Content not available"


if __name__ == "__main__":
    main()
