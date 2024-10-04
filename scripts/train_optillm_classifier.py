import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from safetensors.torch import save_model

# Check for MPS (Apple Silicon) support
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Constants
APPROACHES = ["none", "mcts", "bon", "moa", "rto", "z3", "self_consistency", "pvg", "rstar", "cot_reflection", "plansearch", "leap", "re2"]
MAX_LENGTH = 512  # Maximum sequence length for RoBERTa

class OptILMDataset(Dataset):
    def __init__(self, prompts, efforts, approaches, ranks, tokenizer):
        self.prompts = prompts
        self.efforts = efforts
        self.approaches = approaches
        self.ranks = ranks
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        effort = self.efforts[idx]
        approach = self.approaches[idx]
        rank = self.ranks[idx]

        encoding = self.tokenizer.encode_plus(
            prompt,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'effort': torch.tensor(effort, dtype=torch.float),
            'approach': torch.tensor(APPROACHES.index(approach), dtype=torch.long),
            'rank': torch.tensor(rank, dtype=torch.float)
        }

def normalize_tokens(tokens):
    min_tokens = min(tokens)
    max_tokens = max(tokens)
    if min_tokens == max_tokens:
        return [1.0] * len(tokens)
    return [(t - min_tokens) / (max_tokens - min_tokens) for t in tokens]

def load_and_preprocess_data(tokenizer):
    dataset = load_dataset('json', data_files='optillm_dataset_1.jsonl')
    
    data_items = []

    for item in dataset['train']:
        prompt = item['prompt']
        results = item['results']
        
        valid_results = [r for r in results if 'rank' in r and r['rank'] is not None]
        if not valid_results:
            continue

        tokens = [r['tokens'] for r in valid_results]
        efforts = normalize_tokens(tokens)
        
        for result, effort in zip(valid_results, efforts):
            data_items.append({
                'prompt': prompt,
                'effort': effort,
                'approach': result['approach'],
                'rank': result['rank']
            })

    # Print some statistics
    print(f"Total data points: {len(data_items)}")
    print(f"Unique prompts: {len(set(item['prompt'] for item in data_items))}")
    approach_counts = {approach: sum(1 for item in data_items if item['approach'] == approach) for approach in APPROACHES}
    print("Approach distribution:")
    for approach, count in approach_counts.items():
        print(f"  {approach}: {count}")

    # Split the data
    train_data, val_data = train_test_split(data_items, test_size=0.2, random_state=42)

    train_dataset = OptILMDataset(
        [item['prompt'] for item in train_data],
        [item['effort'] for item in train_data],
        [item['approach'] for item in train_data],
        [item['rank'] for item in train_data],
        tokenizer
    )
    val_dataset = OptILMDataset(
        [item['prompt'] for item in val_data],
        [item['effort'] for item in val_data],
        [item['approach'] for item in val_data],
        [item['rank'] for item in val_data],
        tokenizer
    )

    return train_dataset, val_dataset

class OptILMModel(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(0.1)
        hidden_size = base_model.config.hidden_size
        self.classifier = torch.nn.Linear(hidden_size * 4 + hidden_size, num_labels)
        self.effort_proj = torch.nn.Linear(1, hidden_size)

    def forward(self, input_ids, attention_mask, effort):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Use the last 4 layers
        last_four_layers = outputs.hidden_states[-4:]
        concatenated_layers = torch.cat([layer[:, 0] for layer in last_four_layers], dim=-1)
        
        if effort.dim() == 1:
            effort = effort.unsqueeze(1)
        
        effort_proj = self.effort_proj(effort)
        
        combined_features = torch.cat((concatenated_layers, effort_proj), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        return logits

def custom_loss(outputs, approaches, ranks, efforts):
    ce_loss = F.cross_entropy(outputs, approaches)
    
    # Get the predicted approach
    _, predicted = torch.max(outputs, 1)
    
    # Calculate rank loss only for correct predictions
    correct_predictions = (predicted == approaches)
    if correct_predictions.sum() > 0:
        predicted_ranks = ranks[correct_predictions]
        actual_ranks = ranks[correct_predictions]
        rank_loss = F.mse_loss(predicted_ranks, actual_ranks)
    else:
        rank_loss = torch.tensor(0.0).to(outputs.device)
    
    # Combine losses based on effort
    combined_loss = efforts.mean() * rank_loss + (1 - efforts.mean()) * ce_loss
    
    return combined_loss

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            efforts = batch['effort'].to(device)
            approaches = batch['approach'].to(device)
            ranks = batch['rank'].to(device)

            try:
                outputs = model(input_ids, attention_mask=attention_mask, effort=efforts)
                loss = custom_loss(outputs, approaches, ranks, efforts)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                total_accuracy += calculate_accuracy(outputs, approaches)
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                print(f"Shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, efforts: {efforts.shape}, approaches: {approaches.shape}, ranks: {ranks.shape}")
                continue

        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_accuracy = total_accuracy / len(train_dataloader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        total_val_accuracy = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                efforts = batch['effort'].to(device)
                approaches = batch['approach'].to(device)
                ranks = batch['rank'].to(device)

                try:
                    outputs = model(input_ids, attention_mask=attention_mask, effort=efforts)
                    val_loss = custom_loss(outputs, approaches, ranks, efforts)
                    total_val_loss += val_loss.item()
                    total_val_accuracy += calculate_accuracy(outputs, approaches)
                except RuntimeError as e:
                    print(f"Error in validation batch: {e}")
                    print(f"Shapes - input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, efforts: {efforts.shape}, approaches: {approaches.shape}, ranks: {ranks.shape}")
                    continue

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_val_accuracy / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model
            save_model(model, "best_model.safetensors")

def inference(model, tokenizer, prompt, effort):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        effort_tensor = torch.tensor([effort], dtype=torch.float).to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, effort=effort_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_approach_index = torch.argmax(probabilities, dim=1).item()
        
    return APPROACHES[predicted_approach_index], probabilities[0][predicted_approach_index].item()

def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModel.from_pretrained(args.model_name)
    model = OptILMModel(base_model, num_labels=len(APPROACHES))
    model.to(device)

    # Load and preprocess data
    train_dataset, val_dataset = load_and_preprocess_data(tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    # Train the model
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, args.num_epochs)

    # Save the final model
    save_model(model, "final_model.safetensors")

    if args.push_to_hub:
        model.push_to_hub(args.hub_model_id)
        tokenizer.push_to_hub(args.hub_model_id)

    # Example inference
    test_prompt = "Maximize x + y subject to: x + 2y <= 10, x >= 0, y >= 0"
    test_effort = 0.5
    predicted_approach, confidence = inference(model, tokenizer, test_prompt, test_effort)
    print(f"Test Prompt: {test_prompt}")
    print(f"Effort: {test_effort}")
    print(f"Predicted Approach: {predicted_approach}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OptILM model")
    parser.add_argument("--model_name", type=str, default="roberta-base", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="Model ID for Hugging Face Hub")
    
    args = parser.parse_args()
    main(args)