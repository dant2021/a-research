import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import re
import copy
from peft import get_peft_model, LoraConfig
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

class GPRO:
    def __init__(self):

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
            
        # Use PEFT for memory efficiency
        self.step_count = 0  # Critical missing initialization
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            #quantization_config=quant_config,
        )
        self.model = get_peft_model(self.model, peft_config)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            padding_side="left",
            add_eos_token=True,
            pad_token="<|endoftext|>"
        )
        self.old_policy = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        # Pre-allocate buffers (critical for memory efficiency)
        self.response_buffer = torch.empty((8, 512), dtype=torch.long, device=self.model.device)
        self.reward_buffer = torch.empty(8, device=self.model.device)
        
        # Store system prompt as instance variable
        self.system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
        The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
        <answer> answer here </answer>"""

        # Pre-tokenize system prompt with correct chat template
        self.system_prompt_ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}],
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

    def train_step(self, query, correct_answer, accum_steps=4):
        # Generate full prompt with proper template structure
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": ""}  # Required for response generation
        ]
        
        # Tokenize with explicit left-padding
        full_prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            padding="max_length",  # Force fixed-length padding
            max_length=512,  # Match your response buffer size
            truncation=True
        ).to(self.model.device)

        # Create batch using properly padded prompts
        input_ids = full_prompt_ids.repeat(8, 1)
        
        # Generate attention mask once after padding
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Verify pad token compatibility
        assert self.tokenizer.pad_token == "<|end_of_text|>", \
            f"Expected pad token '<|end_of_text|>', got {self.tokenizer.pad_token}"
        
        # Batch generate 8 responses in parallel
        with torch.no_grad():
            outputs = self.old_policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=8,  # Critical for parallel generation
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # Reshape outputs: [8*bs, seq_len] → [bs, 8, seq_len]
        outputs = outputs.reshape(-1, 8, outputs.size(-1))
        
        # Process all 8 responses per sample
        decoded_responses = []
        for batch in outputs:
            batch_responses = [
                self.tokenizer.decode(
                    seq, 
                    skip_special_tokens=False
                ).split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                for seq in batch
            ]
            decoded_responses.extend(batch_responses)
            
        # Vectorized reward computation (MASSIVE speedup)
        with torch.no_grad():
            # Convert to tensor for GPU acceleration
            answer_tensor = torch.tensor(
                [float(correct_answer)]*8, 
                device=self.model.device
            )
            
            # Batch compute rewards (modify compute_reward to use tensors)
            rewards = self.vectorized_reward(decoded_responses, answer_tensor)
            
        rewards = torch.tensor(rewards, device=self.model.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        total_loss = 0
        for resp, adv in zip(decoded_responses, advantages):
            # Process each response individually
            inputs = self.tokenizer(resp, return_tensors="pt", padding=True).to(self.model.device)
            
            # Get logprobs in memory-efficient way
            with torch.no_grad():
                old_outputs = self.old_policy(**inputs)
                old_logprobs = old_outputs.logits.log_softmax(dim=-1)
                old_logprobs = torch.gather(old_logprobs, 2, inputs.input_ids.unsqueeze(2)).squeeze(-1)
                
            new_outputs = self.model(**inputs)
            new_logprobs = new_outputs.logits.log_softmax(dim=-1)
            new_logprobs = torch.gather(new_logprobs, 2, inputs.input_ids.unsqueeze(2)).squeeze(-1)

            # Policy ratio with gradient checkpointing
            ratio = (new_logprobs - old_logprobs.detach()).exp()
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
            
            # Simplified loss without KL term
            loss = -torch.min(ratio * adv, clipped_ratio * adv).mean()
            loss = loss / accum_steps  # Normalize for accumulation
            loss.backward()

            total_loss += loss.item()
            del inputs, old_outputs, new_outputs, old_logprobs, new_logprobs
            torch.cuda.empty_cache()

        if self.step_count % 10 == 0:
            # Debug prints
            print("Raw rewards:", rewards)
            print("Advantages:", advantages)

        # Optimizer step
        if (self.step_count % accum_steps) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.step_count % 16 == 0:
            self.old_policy.load_state_dict(self.model.state_dict())
            
        self.step_count += 1
        return total_loss

    def vectorized_reward(self, responses, correct_answers):
        """GPU-accelerated reward computation with robust parsing"""
        # Convert to numerical comparison
        answer_pattern = r"<answer>\s*([+-]?\d+\.?\d*)\s*</answer>"
        think_pattern = r"<think>.*?</think>"

        # Batch process using list comprehensions
        answer_matches = [re.search(answer_pattern, r, re.DOTALL) for r in responses]
        think_matches = [re.search(think_pattern, r, re.DOTALL) for r in responses]

        # Numerical comparison tensor
        accuracy = torch.zeros(len(responses), device=self.model.device)
        for i, m in enumerate(answer_matches):
            if m:
                try:
                    # Handle numerical equivalence
                    pred = float(m.group(1).strip())
                    accuracy[i] = 1.0 if abs(pred - correct_answers[i].item()) < 1e-3 else 0.0
                except (ValueError, TypeError):
                    accuracy[i] = 0.0

        # Format validation tensor
        format_reward = torch.tensor([
            0.25 if (think and answer) else -0.25
            for think, answer in zip(think_matches, answer_matches)
        ], device=self.model.device)

        return accuracy + format_reward

# Optimized dataset loading
class MathDataset(Dataset):
    def __init__(self, data_path):
        # Preload and preprocess all data (critical for GPU utilization)
        with open(data_path, "r") as f:
            self.questions = []
            self.answers = []
            for line in f:
                entry = json.loads(line)
                self.questions.append(entry["question"])
                self.answers.append(float(entry["answer"]))  # Numeric conversion
                
        # Pre-convert to tensors
        self.answer_tensor = torch.tensor(self.answers, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.questions[idx], self.answer_tensor[idx]

# Initialize with optimized DataLoader
dataset = MathDataset("/kaggle/input/math-grpo/math_tasks.jsonl")
dataloader = DataLoader(
    dataset,
    batch_size=4,  # Process multiple samples per step
    shuffle=True,
    pin_memory=True,  # Critical for GPU transfer speed
    num_workers=4     # Parallel data loading
)

# Initialize and train
trainer = GPRO()
writer = SummaryWriter('runs/grpo')

for question, answer in dataset:
    loss = trainer.train_step(question, answer)
    print(f"Step {trainer.step_count}: Loss {loss:.4f}")

    # Log to TensorBoard
    writer.add_scalar("Loss/train", loss, trainer.step_count)
    writer.flush()

# Close the TensorBoard writer
writer.close()