#This ingests datsets from "the stack" on huggingface. don't let it run through the whole thing unless you've got lots of time and patience
#5-10 parquets is more than enough. Just hit ctrl+c twice to end it. It caches the files in ~/.cache/huggingface. 
#You will need a huggingface account and will need to generate a token to use this
from datasets import load_dataset

ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train")

print(ds)
print(ds[0])

