# Suggested WebPhish Updates

Make these changes to `src/step_01_training.py`.

## 1. Change the hyperparameters

```python
web_phish_params = {
    "batch_size": 128,
    "embed_dim": 32,
    "conv_channels": 64,
    "learning_rate": 0.0027368243178219833,
    "weight_decay": 0.0001,
    "dropout": 0.3,
    "pos_weight": 1.9,
    "epochs": 15,
    "early_stopping_patience": 3,
    "max_doc_length": 1898,
}
```

Keep `MIN_WORD_COUNT = 10`.

## 2. Replace the current model

Replace the single convolution, pooling, and large flattened layer with:

```python
class WebPhishModel(nn.Module):
    def __init__(self, token_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(
            token_vocab_size,
            32,
            padding_idx=0,
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(32, 64, kernel_size=3),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.Conv1d(32, 64, kernel_size=8),
        ])
        self.fc1 = nn.Linear(64 * 3, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        pooled = [
            torch.amax(F.relu(conv(embedded)), dim=2)
            for conv in self.convs
        ]
        features = torch.cat(pooled, dim=1)
        features = self.dropout(F.relu(self.fc1(features)))
        features = self.dropout(F.relu(self.fc2(features)))
        return self.output(features)
```

## 3. Change the optimizer and loss

```python
optimizer = torch.optim.AdamW(
    nn_model.parameters(),
    lr=web_phish_params["learning_rate"],
    weight_decay=web_phish_params["weight_decay"],
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([1.9], device=gpu_device)
)
```

## 4. Change the training procedure

1. Create a stratified validation split from the training data using
   `random_state=43`.
2. Train for at most 15 epochs.
3. Save the epoch with the highest validation F1.
4. Stop after three epochs without an improvement.
5. Reinitialize the model and train on **all** training rows for seven epochs.
6. Continue using a classification threshold of `0.5`.

For reproducible train/test data, also add `random_state=42` to the stratified
split in `src/step_00_train_test_split.py`.

## 5. Optional ensemble

For the best measured result, train the final model four times with seeds
`101`, `202`, `303`, and `404`. Average their sigmoid probabilities, then apply
the `0.5` threshold.

This CNN ensemble produced:

- Accuracy: `0.9677`
- Precision: `0.9614`
- Recall: `0.9373`
- F1: `0.9492`
- FPR: `0.0179`
- FNR: `0.0627`

The single-model architecture is the main recommended update. The four-model
ensemble is optional when the additional inference cost is acceptable.
