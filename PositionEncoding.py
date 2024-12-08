class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, input_tensor):
        seq_len = input_tensor.size(1)
        batch_size = input_tensor.size(0)
        positions = torch.arange(0, seq_len, device=input_tensor.device).unsqueeze(0).repeat(batch_size, 1)
        position_encodings = self.positional_embedding(positions)
        return input_tensor + position_encodings


class QAWithPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim, transformer_model='bert-base-uncased'):
        super(QAWithPositionalEncoding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        self.transformer = AutoModel.from_pretrained(transformer_model)
        self.positional_encoder = LearnablePositionalEncoding(max_seq_len, embedding_dim)

    def forward(self, context, question):
        inputs = self.tokenizer(context, question, return_tensors='pt', truncation=True, padding=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        embeddings = self.transformer(input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings_with_positional = self.positional_encoder(embeddings)
        return embeddings_with_positional, inputs
