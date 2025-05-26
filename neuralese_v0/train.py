import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
import wandb
import os
from tqdm import tqdm
import json
#from model import LatentAugmentedQwen

# Hardcode wandb API key
os.environ["WANDB_API_KEY"] = "96abf1612b3254db58323f533eda25e38a5a15aa"

# Training config
TRAIN_CONFIG = {
    "model_name": "Qwen/Qwen3-0.6B",
    "dataset": "openai/gsm8k",
    "batch_size": 4,
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "warmup_steps": 100,
    "max_length": 512,
    "gradient_accumulation_steps": 4,
    "save_steps": 200,  # More frequent saving
    "eval_steps": 50,   # More frequent evaluation
    "logging_steps": 1,  # Log every gradient step (much more frequent)
    "console_log_steps": 10,  # Print to console every 10 steps
}

def format_gsm8k_sample(question, answer):
    """Format GSM8K sample as instruction-following format"""
    prompt = f"Question: {question}\nAnswer: {answer}<|endoftext|>"
    return prompt

def create_dataloader(tokenizer, split="train", max_samples=None):
    """Create dataloader for GSM8K dataset"""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    def tokenize_function(examples):
        # Format the examples - examples is a dict with lists
        texts = [
            format_gsm8k_sample(q, a) 
            for q, a in zip(examples["question"], examples["answer"])
        ]
        
        # Tokenize - DON'T use return_tensors="pt" here
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=TRAIN_CONFIG["max_length"]
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create dataloader with proper collate function
    def collate_fn(batch):
        # Convert lists to tensors
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )
    
    return dataloader

def compute_loss(logits, labels, attention_mask):
    """Compute cross-entropy loss"""
    # Shift logits and labels for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_attention_mask = attention_mask[..., 1:].contiguous()
    
    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_attention_mask = shift_attention_mask.view(-1)
    
    # Compute loss only on non-padded tokens
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits, shift_labels)
    
    # Apply attention mask
    loss = loss * shift_attention_mask
    loss = loss.sum() / shift_attention_mask.sum()
    
    return loss

def compute_adjustment_statistics(model):
    """Compute statistics about latent adjustments"""
    stats = {
        "latent_norms": {},
        "bias_norms": {},
        "original_output_norms": {},
        "adjusted_output_norms": {},
        "adjustment_ratios": {},
        "gate_values": {}
    }
    
    # Check if model has collected statistics
    if hasattr(model, 'adjustment_stats') and model.adjustment_stats:
        for layer_name, layer_stats in model.adjustment_stats.items():
            if 'latent_norm' in layer_stats:
                stats["latent_norms"][layer_name] = layer_stats['latent_norm']
            if 'bias_norm' in layer_stats:
                stats["bias_norms"][layer_name] = layer_stats['bias_norm']
            if 'original_norm' in layer_stats:
                stats["original_output_norms"][layer_name] = layer_stats['original_norm']
            if 'adjusted_norm' in layer_stats:
                stats["adjusted_output_norms"][layer_name] = layer_stats['adjusted_norm']
            if 'adjustment_ratio' in layer_stats:
                stats["adjustment_ratios"][layer_name] = layer_stats['adjustment_ratio']
    
    # Add gate values
    if hasattr(model, 'latent_gates'):
        for gate_name, gate_param in model.latent_gates.items():
            gate_value = torch.sigmoid(gate_param).item()
            stats["gate_values"][gate_name] = gate_value
    
    return stats

def log_adjustment_statistics_to_wandb(stats, prefix="train", global_step=None):
    """Log adjustment statistics to wandb"""
    log_dict = {}
    
    # Log average norms across layers
    if stats["latent_norms"]:
        avg_latent_norm = sum(stats["latent_norms"].values()) / len(stats["latent_norms"])
        log_dict[f"{prefix}/avg_latent_norm"] = avg_latent_norm
        
        # Log per-layer latent norms (only first few to avoid clutter)
        for i, (layer, norm) in enumerate(list(stats["latent_norms"].items())[:5]):
            log_dict[f"{prefix}/latent_norm_{layer}"] = norm
    
    if stats["bias_norms"]:
        avg_bias_norm = sum(stats["bias_norms"].values()) / len(stats["bias_norms"])
        log_dict[f"{prefix}/avg_bias_norm"] = avg_bias_norm
        
        # Log per-layer bias norms (only first few)
        for i, (layer, norm) in enumerate(list(stats["bias_norms"].items())[:5]):
            log_dict[f"{prefix}/bias_norm_{layer}"] = norm
    
    if stats["original_output_norms"]:
        avg_original_norm = sum(stats["original_output_norms"].values()) / len(stats["original_output_norms"])
        log_dict[f"{prefix}/avg_original_output_norm"] = avg_original_norm
    
    if stats["adjusted_output_norms"]:
        avg_adjusted_norm = sum(stats["adjusted_output_norms"].values()) / len(stats["adjusted_output_norms"])
        log_dict[f"{prefix}/avg_adjusted_output_norm"] = avg_adjusted_norm
    
    if stats["adjustment_ratios"]:
        avg_adjustment_ratio = sum(stats["adjustment_ratios"].values()) / len(stats["adjustment_ratios"])
        log_dict[f"{prefix}/avg_adjustment_ratio"] = avg_adjustment_ratio
        
        # Log per-layer adjustment ratios (only first few)
        for i, (layer, ratio) in enumerate(list(stats["adjustment_ratios"].items())[:5]):
            log_dict[f"{prefix}/adjustment_ratio_{layer}"] = ratio
    
    # Log norm changes
    if stats["original_output_norms"] and stats["adjusted_output_norms"]:
        norm_changes = {}
        for layer in stats["original_output_norms"]:
            if layer in stats["adjusted_output_norms"]:
                change = stats["adjusted_output_norms"][layer] - stats["original_output_norms"][layer]
                norm_changes[layer] = change
        
        if norm_changes:
            avg_norm_change = sum(norm_changes.values()) / len(norm_changes)
            log_dict[f"{prefix}/avg_norm_change"] = avg_norm_change
    
    # Add global step if provided
    if global_step is not None:
        log_dict["global_step"] = global_step
    
    # Log to wandb
    if log_dict:
        wandb.log(log_dict)

def evaluate_model(model, eval_dataloader, device):
    """Evaluate model on eval dataset"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    print("Running evaluation...")
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass (two-pass during evaluation too)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Compute loss
            loss = compute_loss(logits, labels, attention_mask)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    print(f"Evaluation complete - Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    
    return {
        "eval_loss": avg_loss,
        "eval_perplexity": perplexity.item()
    }

def train_model():
    """Main training function"""
    
    # Initialize wandb
    wandb.init(
        project="latent-augmented-qwen",
        config=TRAIN_CONFIG,
        name="gsm8k-two-pass-training-v3-fixed"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading model...")
    model = LatentAugmentedQwen(TRAIN_CONFIG["model_name"])
    model.to(device)
    
    # Enable statistics collection in model
    model.collect_adjustment_stats = True
    
    # Freeze base model and get trainable parameters
    model.freeze_base_model()
    trainable_params = list(model.get_trainable_parameters())
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_param_count:,} ({100*trainable_param_count/total_params:.2f}%)")
    
    # Create dataloaders
    print("Loading datasets...")
    train_dataloader = create_dataloader(model.tokenizer, split="train", max_samples=1000)
    eval_dataloader = create_dataloader(model.tokenizer, split="test", max_samples=200)
    
    print(f"Train samples: {len(train_dataloader) * TRAIN_CONFIG['batch_size']}")
    print(f"Eval samples: {len(eval_dataloader) * TRAIN_CONFIG['batch_size']}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(trainable_params, lr=TRAIN_CONFIG["learning_rate"])
    
    total_steps = len(train_dataloader) * TRAIN_CONFIG["num_epochs"] // TRAIN_CONFIG["gradient_accumulation_steps"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
        num_training_steps=total_steps
    )
    
    print(f"Total training steps: {total_steps}")
    
    # Training loop
    print("Starting training...")
    model.train()
    global_step = 0
    running_loss = 0.0
    batch_count_since_log = 0
    
    for epoch in range(TRAIN_CONFIG["num_epochs"]):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch + 1}/{TRAIN_CONFIG['num_epochs']}")
        print(f"{'='*50}")
        
        epoch_loss = 0
        num_batches = 0
        
        # Training
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass (two-pass)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Compute loss
            loss = compute_loss(logits, labels, attention_mask)
            loss = loss / TRAIN_CONFIG["gradient_accumulation_steps"]
            
            # Update progress bar with current loss
            progress_bar.set_postfix({
                'loss': f'{loss.item() * TRAIN_CONFIG["gradient_accumulation_steps"]:.3f}',
                'step': global_step
            })
            
            # Backward pass
            loss.backward()
            
            epoch_loss += loss.item()
            running_loss += loss.item()
            num_batches += 1
            batch_count_since_log += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % TRAIN_CONFIG["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Console logging (FIXED - more visible)
                if batch_count_since_log >= TRAIN_CONFIG["console_log_steps"]:
                    avg_loss = running_loss / batch_count_since_log * TRAIN_CONFIG["gradient_accumulation_steps"]
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Get current adjustment stats
                    adj_stats = compute_adjustment_statistics(model)
                    avg_adj_ratio = "N/A"
                    if adj_stats["adjustment_ratios"]:
                        avg_adj_ratio = f"{sum(adj_stats['adjustment_ratios'].values()) / len(adj_stats['adjustment_ratios']):.2f}"
                    
                    print(f"📊 Step {global_step:4d} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | Adj Ratio: {avg_adj_ratio}")
                    
                    running_loss = 0.0
                    batch_count_since_log = 0
                
                # Wandb logging (every step)
                if global_step % TRAIN_CONFIG["logging_steps"] == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Basic training metrics
                    log_dict = {
                        "train/loss": loss.item() * TRAIN_CONFIG["gradient_accumulation_steps"],
                        "train/learning_rate": current_lr,
                        "global_step": global_step,
                        "epoch": epoch
                    }
                    
                    # Compute and add adjustment statistics
                    adj_stats = compute_adjustment_statistics(model)
                    if adj_stats["latent_norms"]:
                        avg_latent_norm = sum(adj_stats["latent_norms"].values()) / len(adj_stats["latent_norms"])
                        log_dict["train/avg_latent_norm"] = avg_latent_norm
                    if adj_stats["bias_norms"]:
                        avg_bias_norm = sum(adj_stats["bias_norms"].values()) / len(adj_stats["bias_norms"])
                        log_dict["train/avg_bias_norm"] = avg_bias_norm
                    if adj_stats["adjustment_ratios"]:
                        avg_adjustment_ratio = sum(adj_stats["adjustment_ratios"].values()) / len(adj_stats["adjustment_ratios"])
                        log_dict["train/avg_adjustment_ratio"] = avg_adjustment_ratio
                    if adj_stats["gate_values"]:
                        avg_gate_value = sum(adj_stats["gate_values"].values()) / len(adj_stats["gate_values"])
                        log_dict["train/avg_gate_value"] = avg_gate_value
                        
                        # Log first 5 gate values
                        for i, (gate_name, gate_val) in enumerate(list(adj_stats["gate_values"].items())[:5]):
                            log_dict[f"train/{gate_name}"] = gate_val
                    
                    wandb.log(log_dict)
                
                # Evaluation
                if global_step % TRAIN_CONFIG["eval_steps"] == 0:
                    print(f"\n--- Evaluating at step {global_step} ---")
                    eval_metrics = evaluate_model(model, eval_dataloader, device)
                    
                    
                    wandb.log({
                        "eval/loss": eval_metrics["eval_loss"],
                        "eval/perplexity": eval_metrics["eval_perplexity"],
                        "global_step": global_step
                    })
                    
                    model.train()
                
                # Save checkpoint
                if global_step % TRAIN_CONFIG["save_steps"] == 0:
                    checkpoint_dir = f"checkpoints/step_{global_step}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    trainable_state_dict = {
                        name: param for name, param in model.named_parameters() 
                        if param.requires_grad
                    }
                    
                    torch.save({
                        "model_state_dict": trainable_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                        "config": TRAIN_CONFIG
                    }, f"{checkpoint_dir}/checkpoint.pt")
                    
                    print(f"✅ Checkpoint saved at {checkpoint_dir}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / num_batches * TRAIN_CONFIG["gradient_accumulation_steps"]
        print(f"\n🏁 Epoch {epoch + 1} Complete!")
        print(f"Average Loss: {avg_epoch_loss:.4f}")
        
        wandb.log({
            "epoch/loss": avg_epoch_loss,
            "epoch": epoch + 1
        })
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    final_eval_metrics = evaluate_model(model, eval_dataloader, device)
    
    wandb.log({
        "final/eval_loss": final_eval_metrics["eval_loss"],
        "final/eval_perplexity": final_eval_metrics["eval_perplexity"]
    })
    
    print(f"🎉 Training Complete!")
    print(f"Final Eval Loss: {final_eval_metrics['eval_loss']:.4f}")
    print(f"Final Eval Perplexity: {final_eval_metrics['eval_perplexity']:.4f}")
    
    # Save final model
    final_checkpoint_dir = "checkpoints/final"
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    trainable_state_dict = {
        name: param for name, param in model.named_parameters() 
        if param.requires_grad
    }
    
    torch.save({
        "model_state_dict": trainable_state_dict,
        "config": TRAIN_CONFIG,
        "final_metrics": final_eval_metrics
    }, f"{final_checkpoint_dir}/final_model.pt")
    
    print(f"✅ Final model saved at {final_checkpoint_dir}")
    wandb.finish()

if __name__ == "__main__":
    train_model() 