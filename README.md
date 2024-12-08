# ScriptChainML

## Attempt to implement information retrieval model using self-attention layers with positional encoding. The model uses BERT-based models and uses cosine similarity to identify and extract answers from a given context.


# Feasibility of representing a sequence by stacking self-attention layers with positional encoding.
## Stacking multiple self-attention layers with positional encoding in deep architecture to represent a sequence might not always create issues, it depends on various factors such as sequence length, dataset size, and resource availability. The use of multiple self-attention layers, which are used to process sequences, have high computational complexity, so stacking such layers increases computational cost significantly which can lag model training and increase memory usage. 

## Overfitting can be another major issue with many layers and a large number of parameters in a deep architecture if the dataset is not large enough. The model learns to memorize the training data rather than generalizing to unseen data, leading to poor performance on test data. Also using multiple positional encodings in a model can introduce several potential issues. With the use of multiple positional encodings, there is a risk of positional conflicts and redundancy, multiple encodings representing the same positional information can confuse the model and if combined encodings are not aligned with the task it will lead to very slow learning. Therefore, Using multiple positional encodings can cause redundancy, confusion, and instability in model training, which can lead to model inefficiency so stacking self-attention layers with positional encoding must be done carefully to avoid unnecessary complexity.
