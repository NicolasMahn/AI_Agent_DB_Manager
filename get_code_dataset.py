from datasets import load_dataset

# Specify the cache directory
cache_dir = "./dataset_cache"

# Load the dataset in streaming mode
ds = load_dataset("bigcode/the-stack", streaming=True, cache_dir=cache_dir)

WHITE = "\033[97m"
PURPLE = "\033[35m"
ORANGE = "\033[38;5;208m"
RESET = "\033[0m"


# Filter the dataset to include only Python code
filtered_dataset = []
for i, python in enumerate(ds['train']):
    if python['lang'] == 'python':
        print(python)
        filtered_dataset.append(python)
        break
    else:
        print("Not Python: \n")
        for k, v in python.items():
            print(f"{PURPLE}{k}:{RESET} {v}")
        break
