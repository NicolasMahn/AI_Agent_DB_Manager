import os
import json
import PyPDF2
import re

from pygments.lexer import default

from llm_api_wrapper import basic_prompt
from util import extract_code_from_markdown

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"


def save_data(output_dir, pdf_name, results):
    """
    Saves the extracted sections as text files and updates the JSON configuration file.

    Args:
        output_dir (str): Base directory for saving output.
        pdf_name (str): Original PDF file name.
        results (list of dict): List of results with 'heading', 'content', and 'page'.
    """
    config_path = os.path.join(output_dir, "data.json")
    config = []

    # Process each result
    for result in results:
        heading = result["title"]
        content = result.get("content", None)
        page = result.get("page", None)


        # Add entry to the configuration
        config_entry = {
            "pdf_name": pdf_name,
            "page": page,
            "title": heading
        }
        if content:
            config_entry["content"] = content
        config.append(config_entry)

    # Save the updated configuration file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def get_saved_config(output_dir):
    config_path = os.path.join(output_dir, "data.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    return None


def multiline_input():
    """
    Reads multiline input until the user types 'END'.
    Returns:
        str: The complete multiline input.
    """
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def is_doubled_section(s):
    """Check if the entire section is doubled (all characters appear in pairs)."""
    if len(s) % 2 != 0:
        return False
    for i in range(0, len(s), 2):
        if s[i] != s[i + 1]:
            return False
    return True


def remove_doubled_characters_section(s):
    """Remove doubled characters from a section where all characters are doubled."""
    # Split the text into sections based on double line breaks
    sections = s.split(" ")
    cleaned_sections = []
    for section in sections:
        if is_doubled_section(section.replace(" ", "").replace("\n", "")):
            # Remove doubles in this section
            cleaned_section = ''.join([section[i] for i in range(0, len(section), 2)])
            cleaned_sections.append(cleaned_section)
        else:
            # Keep section as is
            cleaned_sections.append(section)
    cleaned_text = " ".join(cleaned_sections)
    # Remove double spaces
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    return cleaned_text


def get_text_from_pdf(pdf_path, start_page, end_page=None):
    """
    Extracts text from a range of pages in a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.
        start_page (int): The first page to extract text from (1-based index).
        end_page (int): The last page to extract text from (1-based index).

    Returns:
        str: The extracted text.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)

            if end_page is None:
                end_page = num_pages

            if start_page < 1 or end_page > num_pages:
                print(f"Invalid page range. The PDF has {num_pages} pages.")
                return text

            for page_number in range(start_page - 1, end_page):
                text += reader.pages[page_number].extract_text()

    except Exception as e:
        print(f"Error reading the PDF: {e}")

    return text


def get_content_list_from_pdf(content_txt):
    """
    Extracts the content list from the extracted text of the PDF.

    Args:
        content_txt (str): The extracted text containing the content list.

    Returns:
        list of dict: A list of dictionaries with 'title' and 'page'.
    """
    prompt = f"Extract the content list from the text: \n```\n{content_txt}\n```"
    role = """Your job is to extract the content list from the text.
Please use the following format:
```
[
    {"title": "Überüberschrift", "page": 10, "hierarchy": 1},
    {"title": "Unterüberschrift", "page": 10, "hierarchy": 2}
]
```
Incase you are unsure about the page numbers or hierarchy, you may remove these two keys from the dictionary."""
    response_txt = basic_prompt(prompt, role, temperature=0.1, model="gpt-4o-mini")
    content_list_str = extract_code_from_markdown(response_txt)[0]
    content_list = json.loads(content_list_str)

    return content_list


def remove_default(content_txt, output_dir):
    default_remove_lines = None
    if os.path.exists(os.path.join(output_dir, "default_remove.json")):
        with open(os.path.join(output_dir, "default_remove.json"), "r", encoding="utf-8") as f:
            default_remove_lines = json.load(f)
    if default_remove_lines:
        for line in default_remove_lines:
            content_txt = re.sub(line, '', content_txt, flags=re.MULTILINE)
            # removing unnecessary break lines
            content_txt = content_txt.replace("\n\n", "\n")

    return content_txt


def main():
    # Input paths
    pdf_path = input(f"{BLUE}Enter the path to the PDF file: {RESET}").strip()
    if pdf_path == "":
        pdf_path = "data/union/km_btw_2025_wahlprogramm_langfassung_ansicht.pdf"

    output_dir = input(f"{BLUE}Enter the output directory: {RESET}").strip()
    if output_dir == "":
        output_dir = "data/union"

    back_flag = False

    content_list = get_saved_config(output_dir)
    if not content_list:

        # Specify start page of content
        start_page_content = int(input(f"{BLUE}If the document has no content pages, enter 0. "
                                       "Otherwise, enter the first page of the content (Inhaltsverzeichnis) "
                                       f"(1-based index): {RESET}").strip())
        if start_page_content == 0:
            print(f"{BLUE}Please input all headings, ideally with the page number in the same line. "
                  f"Don't worry about formatting. Type {WHITE}END{BLUE} when your done.\n{RESET}")
            content_txt = multiline_input()
        else:
            end_page_content = int(input(f"{BLUE}Enter the last page of the content (Inhaltsverzeichnis) "
                                         f"(1-based index): {RESET}").strip())
            content_txt = get_text_from_pdf(pdf_path, start_page_content, end_page_content)
        print(f"{BLUE}The text is being processed...{RESET}\n")

        content_list = get_content_list_from_pdf(content_txt)
        page_offset = int(input(f"{BLUE}Is there a unaccounted title page(s)? {RESET}").strip().lower())
        if page_offset > 0:
            for i in range(len(content_list)):
                content_list[i]["page"] += page_offset

        save_data(output_dir, os.path.basename(pdf_path), content_list)

    i = 0
    while True:
        if i >= len(content_list):
            break
        current_content = content_list[i]
        if current_content.get("content", None) and not back_flag:
            i += 1
            continue
        if back_flag:
            back_flag = False
        next_content = content_list[i+1] if i+1 < len(content_list) else None
        content_txt = get_text_from_pdf(pdf_path, current_content["page"],
                                        next_content["page"] if next_content else None)

        content_txt = remove_default(content_txt, output_dir)
        content_txt = remove_doubled_characters_section(content_txt)

        search_by_cc_title = current_content.get("search_by_title", current_content["title"])
        content_txt = content_txt[content_txt.find(search_by_cc_title):]

        if next_content:
            search_by_nc_title = next_content.get("search_by_title", next_content["title"])
            content_txt = content_txt[:content_txt.find(search_by_nc_title)]



        print(f"{BLUE}The text behind the title {WHITE}'{current_content['title']}'{BLUE} "
              f"(starting from page {WHITE}{current_content['page']}{BLUE}) is:"
              f"{WHITE}\n\n{content_txt}\n\n{BLUE}")
        print(f"The next title is {WHITE}'{next_content['title']}'{BLUE}. \n" if next_content else "")

        next_step = input(f"Is this section accurate? \nDo you want to EDIT this section "
                          f"(or edit the (next) TITLE), REMOVE this section, BACK"
                          f" or CONTINUE? {PINK}(t/nt/e/r/b/C){BLUE}: {RESET}").strip().lower()
        if next_step == "b":
            back_flag = True
            i -= 1
            continue
        if next_step == "t":
            new_title = input(f"{BLUE}Please input the new title: {RESET}")
            current_content["search_by_title"] = new_title
            continue
        elif next_step == "nt":
            new_title = input(f"{BLUE}Please input the new (next) title: {RESET}")
            next_content["search_by_title"] = new_title
            continue
        if next_step == "e":
            original_content = content_txt
            while True:
                remove_line = input(f"{BLUE}Input the characters you would like to remove, \n"
                                    f"{PINK}STOP{BLUE}, if everything is now satisfactory, "
                                    f"{PINK}PRINT{BLUE}, to reprint the section, or "
                                    f"{PINK}UNDO{BLUE}, to undo all edits: {RESET}")
                if remove_line.upper() == "STOP" or remove_line == "":
                    break
                elif remove_line.upper() == "PRINT":
                    print(f"{WHITE}{content_txt}{RESET}")
                elif remove_line.upper() == "UNDO":
                    content_txt = original_content
                else:
                    content_txt = content_txt.replace(remove_line, "")
                    # removing unnecessary break lines
                    content_txt = content_txt.replace("\n\n", "\n")

        elif next_step == "r":
            content_list.remove(current_content)
            continue

        current_content["content"] = content_txt
        save_data(output_dir, os.path.basename(pdf_path), content_list)

        i += 1
        print("\n\n")

    print(f"{PINK}All sections have been extracted and saved.{RESET}")


if __name__ == "__main__":
    main()
