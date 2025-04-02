from datasets import load_dataset

dataset = load_dataset(path="wikitext", name="wikitext-103-v1", split="train")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")