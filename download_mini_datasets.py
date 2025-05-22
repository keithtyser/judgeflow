import os
import pandas as pd
from datasets import load_dataset
import tiktoken

# Ensure datasets directory exists
os.makedirs('datasets', exist_ok=True)

# Helper to count tokens using tiktoken (GPT-3.5/4 encoding)
def count_tokens(texts, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return sum(len(enc.encode(str(t))) for t in texts)

# 1. TruthfulQA (validation split)
tqa = load_dataset("truthful_qa", "generation", split="validation").select(range(20))
tqa_df = pd.DataFrame({
    'question': tqa['question'],
    'best_answer': tqa['best_answer'],
    'correct_answers': tqa['correct_answers'],
    'incorrect_answers': tqa['incorrect_answers'],
})
tqa_df.to_parquet("datasets/truthfulqa.parquet")

# 2. Jigsaw (toxic comment classification)
# NOTE: Download the Jigsaw dataset manually from Kaggle and extract to 'jigsaw_data' directory.
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
jigsaw = load_dataset("jigsaw_toxicity_pred", data_dir="jigsaw_data", split="train", trust_remote_code=True).select(range(20))
jigsaw_df = pd.DataFrame({
    'text': jigsaw['comment_text'],
    'toxicity': jigsaw['toxic'],
})
jigsaw_df.to_parquet("datasets/jigsaw.parquet")

# 3. MMLU (use 'abstract_algebra' as a sample subject)
mmlu = load_dataset("lukaemon/mmlu", "abstract_algebra", split="test", trust_remote_code=True).select(range(20))
mmlu_df = pd.DataFrame({
    'question': mmlu['input'],
    'choices': list(zip(mmlu['A'], mmlu['B'], mmlu['C'], mmlu['D'])),
    'answer': mmlu['target'],
})
mmlu_df.to_parquet("datasets/mmlu.parquet")

# Token count (all text fields)
tqa_tokens = count_tokens(tqa_df['question'].tolist() + tqa_df['best_answer'].tolist())
jigsaw_tokens = count_tokens(jigsaw_df['text'].tolist())
mmlu_tokens = count_tokens(mmlu_df['question'].tolist())
total_tokens = tqa_tokens + jigsaw_tokens + mmlu_tokens

print(f"Total tokens: {total_tokens} (should be < 8000)")
if total_tokens >= 8000:
    print("Warning: Token count exceeds 8,000!")
else:
    print("Token count is within limit.") 