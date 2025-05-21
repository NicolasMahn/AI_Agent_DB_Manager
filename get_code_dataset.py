import os
import shutil
import signal

os.environ['HF_HOME'] = "D:\.cache"

from tqdm import tqdm

from datasets import load_dataset, IterableDataset


def save_python_code(python):
    os.makedirs("data/python_code/documents", exist_ok=True)
    with open(f"data/python_code/documents/{python['hexsha']}.txt", "w") as f:
        f.write(python["content"])

def save_progress(index):
    os.makedirs("data/python_code", exist_ok=True)
    with open("data/python_code/progress.txt", "w") as f:
        f.write(str(index))

def load_progress():
    os.makedirs("data/python_code", exist_ok=True)
    if os.path.exists("data/python_code/progress.txt"):
        with open("data/python_code/progress.txt", "r") as f:
            return int(f.read().strip())
    return 0

def handle_termination(signum, frame):
    global current_index
    save_progress(current_index)
    print(f"\nProgress saved at index {current_index}. Exiting...")
    exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_termination)
signal.signal(signal.SIGTERM, handle_termination)

# Specify the cache directory


# Configure retries
# session = requests.Session()
# retries = Retry(total=10, backoff_factor=1, status_forcelist=[502, 503, 504])
# session.mount('https://', HTTPAdapter(max_retries=retries))

# Load the dataset in streaming mode
ds: IterableDataset = load_dataset("bigcode/the-stack", split="train")

WHITE = "\033[97m"
PURPLE = "\033[35m"
ORANGE = "\033[38;5;208m"
GREEN = "\033[92m"
RESET = "\033[0m"

debug = True

# Filter the dataset to include only Python code
term_width = shutil.get_terminal_size((80, 20)).columns - 10
bar_format = f"{WHITE}📊 Collecting Data{{l_bar}}{GREEN}{{bar}}{WHITE}{{r_bar}}{RESET}"

# Initialize the tqdm progress bar
total_elements = ds.dataset_size
pbar = tqdm(total=total_elements, bar_format=bar_format)

# Define a wrapper function that updates the progress bar
def filter_with_progress(x):
    pbar.update(1)
    return x['lang'].lower() == 'python'

# Filter the dataset to include only Python code
filtered_ds = ds.filter(filter_with_progress)

# Iterate over the filtered dataset
for i, python in enumerate(filtered_ds):
    current_index = i
    if debug:
        print(f"\n--------------------------------------------------------------------------------------------------\n")
        print(f"{GREEN}Saving this Python code: {RESET}\n")
        for k, v in python.items():
            print(f"{GREEN}{k}:{RESET} {v}")
        print(f"\n--------------------------------------------------------------------------------------------------\n")
    save_python_code(python)
