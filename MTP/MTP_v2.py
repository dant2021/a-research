import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset
import wandb
import bitsandbytes as bnb

class MultiTokenPredictor(nn.Module):
    def __init__(self, base_model, n_tokens=4):
        """
        base_model: Pretrained LLaMA model
        n_tokens: Number of future tokens to predict (e.g. 4)
        """
        super().__init__()
        self.n_tokens = n_tokens
        
        # Use pretrained LLaMA as trunk and freeze its parameters
        self.trunk = base_model
        for param in self.trunk.parameters():
            pass
        # Get config from base model
        config = base_model.config
        
        # Create n separate prediction heads
        last_layer = base_model.model.layers[-1]
        self.heads = nn.ModuleList()
        for _ in range(n_tokens):
            new_layer = type(last_layer)(config, layer_idx=15)
            new_layer.load_state_dict(last_layer.state_dict())
            self.heads.append(new_layer)
        
        # Share the unembedding matrix with the trunk
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.unembedding.weight = base_model.model.embed_tokens.weight
        # Freeze the unembedding layer too since it's shared
        self.unembedding.weight.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Get shared representation
        hidden_states = self.trunk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        ).hidden_states[-2]
        
        # Detach hidden_states to prevent gradients flowing back to trunk during head backwards
        head_input = hidden_states.detach()
        head_input.requires_grad = True

        # Prepare causal attention mask
        batch_size, seq_length = input_ids.shape
        causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        
        # Convert attention_mask to 4D boolean mask
        extended_attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [batch_size, 1, 1, seq_length]
        
        total_loss = 0
        grad_outputs = []
        step_losses = []

        # Calculate position IDs for the heads
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Forward through each head separately
        for i in range(len(self.heads)):
            # Adjust attention mask for each head to prevent looking at future tokens
            head_causal_mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device), diagonal=i+1)
            head_causal_mask = head_causal_mask.unsqueeze(0).unsqueeze(0)
            head_attention_mask = extended_attention_mask & ~head_causal_mask
            
            # Forward through head i
            outputs = self.heads[i](
                hidden_states=head_input,
                attention_mask=head_attention_mask,
                position_ids=position_ids,
                use_cache=False
            )
            head_hidden = outputs[0]
            logits = self.unembedding(head_hidden)
            
            # Get targets for position i+1
            shifted_targets = input_ids[:, i+1:]
            
            # Get predictions for tokens at position i+1
            predictions = logits[:, :-i-1, :]  # Trim predictions to match targets
            
            # Compute loss for this head
            loss = nn.functional.cross_entropy(
                predictions.reshape(-1, predictions.size(-1)),
                shifted_targets.reshape(-1),
                ignore_index=-100
            )
            
            # Scale loss for non-first heads (optional, depending on your needs)
            if i > 0:
                loss = loss * 0.1
            
            loss.backward()
            
            total_loss += loss.detach()
            grad_outputs.append(head_input.grad.clone())
            head_input.grad = None
            
            step_losses.append(loss.detach().item())

            if batchidx % 100 == 0:
                # Get top predictions for each position
                top_tokens = torch.argmax(logits, dim=-1)  # [batch, seq]
                
                # Convert to text
                input_text = tokenizer.decode(input_ids[0])
                pred_text = tokenizer.decode(top_tokens[0])
                
                print(f"\nStep {batchidx}, Head {i}")
                if i == 0:
                    print(f"Input: {input_text}")
                print(f"Predictions: {pred_text}")
                print(f"grad_outputs norm: {grad_outputs[i].norm():.4f} std {grad_outputs[i].std():.2f}")
                

        # Sum up all gradients for hidden_states
        final_grad = sum(grad_outputs)
        # Now backward through trunk with accumulated gradient
        hidden_states.backward(gradient=final_grad)

        del hidden_states, head_input, logits, predictions, shifted_targets
        torch.cuda.empty_cache()
        
        return total_loss.item(), step_losses

    def generate(self, input_ids, max_length=100, **kwargs):
        """
        Basic autoregressive generation using only the first head
        """
        # Standard autoregressive generation using first head only
        while input_ids.shape[1] < max_length:
            # Get predictions from model
            outputs = self(input_ids)
            next_token_logits = outputs[0][:, -1, :]
            
            # Sample next token
            next_token = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
        return input_ids

# 1. First set up the model and tokenizer
def setup_model(model_name="meta-llama/Llama-3.2-1B-Instruct"):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, dtype=torch.bfloat16)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = LlamaForCausalLM.from_pretrained(model_name)
    model = MultiTokenPredictor(base_model, n_tokens=4)
    return model, tokenizer

# 2. Training function
def train_mtp(model, tokenizer, train_dataloader, val_dataloader=None, num_epochs=3, 
              learning_rate=1e-4, device="cuda", gradient_accumulation_steps=1):
    
    wandb.init(project="mtp-training", config={
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "model_type": "MTP",
        "n_tokens": model.n_tokens
    })
    
    model = model.to(device)
    optimizer = bnb.optim.AdamW8bit([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        global batchidx
        batchidx = 0

        for batchidx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            
            # Forward pass
            loss, step_losses = model(input_ids, attention_mask=attention_mask)

            if (batchidx + 1) % gradient_accumulation_steps == 0:
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
            
            # Log metrics
            metrics = {
                "avg_loss": loss,
                **{f"step_{i}_loss": l for i, l in enumerate(step_losses)}
            }
            wandb.log(metrics)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss})

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, f'mtp_checkpoint_epoch_{epoch}.pt')

        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "epoch_loss": total_loss / len(train_dataloader)
        })

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx]
        }

    def __len__(self):
        return len(self.encodings["input_ids"])

# 5. Example usage
if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    model, tokenizer = setup_model()

    # Load dataset
    dataset = load_dataset(
        "ant-des/filtered_reasoning_deepseek", 
        split='train'
    ).select(range(5000))

    train_texts = []

    # Correct way to access dataset elements
    for item in dataset:
        # Combine message content and answer
        train_texts.append(str(item['messages'][0]['content'] + item['answer']))

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False
    )
        
    # Train the model
    train_mtp(
        model,
        tokenizer,
        train_dataloader,
        num_epochs=3,
        learning_rate=1e-4,
        device=device
    )