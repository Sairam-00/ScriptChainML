import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from PositionEncoding import QAWithPositionalEncoding,LearnablePositionalEncoding

def extract_answer_with_spans(model, file_content, question):
    context_embedding, inputs = model(file_content, question)
    pooled_embedding = context_embedding.mean(dim=1)
    similarity_scores = torch.cosine_similarity(context_embedding, pooled_embedding.unsqueeze(1), dim=-1)

    max_score, max_index = similarity_scores.max(dim=1)
    start_index = max_index.item()
    window_size = 15
    end_index = min(start_index + window_size, inputs['input_ids'].size(1))

    input_ids = inputs['input_ids'][0].tolist()
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids)
    answer_tokens = tokens[start_index:end_index]
    extracted_answer = model.tokenizer.convert_tokens_to_string(answer_tokens)

    return extracted_answer, max_score.item()

if __name__ == "__main__":
    file_path = "./story.txt"
    with open(file_path, 'r') as file:
        file_content = file.read()

    question = "What is the moral of the story?"

    max_seq_len = 1028
    embedding_dim = 768
    qa_model = QAWithPositionalEncoding(max_seq_len, embedding_dim)

    extracted_answer, confidence_score = extract_answer_with_spans(qa_model, file_content, question)
    print(f"Extracted Answer: {extracted_answer}")
    print(f"Confidence Score: {confidence_score:.2f}")
