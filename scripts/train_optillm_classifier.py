import argparse
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
from safetensors.torch import save_model
from collections import Counter

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
MAX_LENGTH = 512

class OptILMDataset(Dataset):
    def __init__(self, prompts, approaches, tokenizer):
        self.prompts = prompts
        self.approaches = approaches
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        approach = self.approaches[idx]

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
            'labels': torch.tensor(APPROACHES.index(approach), dtype=torch.long)
        }

def load_and_preprocess_data(tokenizer):
    dataset = load_dataset('json', data_files='optillm_dataset_1.jsonl')
    
    data_items = []

    for item in dataset['train']:
        prompt = item['prompt']
        results = item['results']
        
        valid_results = [r for r in results if 'approach' in r]
        if not valid_results:
            continue

        for result in valid_results:
            data_items.append({
                'prompt': prompt,
                'approach': result['approach']
            })

    # Print some statistics
    print(f"Total data points: {len(data_items)}")
    print(f"Unique prompts: {len(set(item['prompt'] for item in data_items))}")
    approach_counts = Counter(item['approach'] for item in data_items)
    print("Approach distribution:")
    for approach, count in approach_counts.items():
        print(f"  {approach}: {count}")

    # Calculate class weights for balanced sampling
    class_weights = {approach: len(data_items) / count for approach, count in approach_counts.items()}
    sample_weights = [class_weights[item['approach']] for item in data_items]

    # Split the data
    train_data, val_data = train_test_split(data_items, test_size=0.2, random_state=42)

    train_dataset = OptILMDataset(
        [item['prompt'] for item in train_data],
        [item['approach'] for item in train_data],
        tokenizer
    )
    val_dataset = OptILMDataset(
        [item['prompt'] for item in val_data],
        [item['approach'] for item in val_data],
        tokenizer
    )

    # Create a weighted sampler for the training data
    train_sampler = WeightedRandomSampler(
        weights=[class_weights[item['approach']] for item in train_data],
        num_samples=len(train_data),
        replacement=True
    )

    return train_dataset, val_dataset, train_sampler

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    return (predictions == labels).float().mean()

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, num_epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_accuracy = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(logits, labels)

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
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                logits = outputs.logits

                total_val_loss += val_loss.item()
                total_val_accuracy += calculate_accuracy(logits, labels)

        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_val_accuracy / len(val_dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the best model
            save_model(model, "best_model.safetensors")

def inference(model, tokenizer, prompt):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_approach_index = torch.argmax(probabilities, dim=1).item()
        
    return APPROACHES[predicted_approach_index], probabilities[0][predicted_approach_index].item()

def main(args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(APPROACHES))
    model.to(device)

    # Load and preprocess data
    train_dataset, val_dataset, train_sampler = load_and_preprocess_data(tokenizer)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
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
    predicted_approach, confidence = inference(model, tokenizer, test_prompt)
    print(f"Test Prompt: {test_prompt}")
    print(f"Predicted Approach: {predicted_approach}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OptILM classifier")
    parser.add_argument("--model_name", type=str, default="roberta-large", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, help="Model ID for Hugging Face Hub")
    
    args = parser.parse_args()
    main(args)