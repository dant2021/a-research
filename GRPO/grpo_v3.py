import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizerFast
from torch.utils.data import Dataset
import json
import re
import copy
from peft import get_peft_model, LoraConfig
from torch.utils.tensorboard import SummaryWriter

class GPRO:
    def __init__(self):

        #quant_config = BitsAndBytesConfig(
        #    load_in_4bit=True,
        #    bnb_4bit_use_double_quant=True,
        #    bnb_4bit_quant_type="nf4",
        #    bnb_4bit_compute_dtype=torch.bfloat16
        #)
            
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
            pad_token="<|end_of_text|>"
        )
        self.old_policy = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

    def train_step(self, query, correct_answer, accum_steps=4):
        system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        # Apply chat template correctly
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize as batch (even single example)
        inputs = self.tokenizer(
            [formatted_prompt],  # Critical list wrapper
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        ).to(self.model.device)

        # Generate group with old policy
        with torch.no_grad():
            group_responses = []
            
            outputs = self.old_policy.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=8,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            # Properly decode each sequence individually
            for i in range(outputs.sequences.shape[0]):
                decoded_response = self.tokenizer.decode(
                    outputs.sequences[i], 
                    skip_special_tokens=False
                )
                # Extract content after last end_header_id
                split_response = decoded_response.split("<|end_header_id|>")[-1]
                group_responses.append(split_response)

            del inputs, outputs
            torch.cuda.empty_cache()

        # Process one response at a time to save memory
        total_loss = 0
        rewards = []
        for resp in group_responses:
            # Compute reward
            reward = compute_reward(resp, correct_answer)
            rewards.append(reward)
            del resp  # Free memory immediately
            
        rewards = torch.tensor(rewards, device=self.model.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        for resp, adv in zip(group_responses, advantages):
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

# Custom Dataset Class
class MathDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["question"], self.data[idx]["answer"]
       
# Reward Function
def compute_reward(response, correct_answer):
    """
    Compute the reward for a response.
    
    Args:
        response (str): The model's response.
        correct_answer (str): The correct answer to the query.
    
    Returns:
        float: The reward for the response.
    """
    # Accuracy Reward
    answer_match = re.search(r"<answer>(.*?)</answer>", response)
    if answer_match:
        predicted_answer = answer_match.group(1).strip() # here i want to take the last group of the answer fix it
        accuracy_reward = 1.0 if correct_answer in predicted_answer else 0.0
    else:
        accuracy_reward = 0.0

    # Format Reward
    has_think_tag = "<think>" in response and "</think>" in response
    has_answer_tag = "<answer>" in response and "</answer>" in response
    format_reward = 0.25 if has_think_tag and has_answer_tag else -0.25
    
    # Language Consistency Reward
    language_consistency_reward = 0.10 if is_consistent_language(response) else -0.10
    
    # Total Reward
    total_reward = accuracy_reward + format_reward + language_consistency_reward
    return total_reward

def is_consistent_language(text):
    """
    Check if the text is consistent in language (e.g., no mixing of languages).
    
    Args:
        text (str): The text to check.
    
    Returns:
        bool: True if the language is consistent, False otherwise.
    """
    non_english_words = re.findall(r'[^\x00-\x7F]+', text)
    return len(non_english_words) == 0

# Initialize and train
trainer = GPRO()
writer = SummaryWriter('runs/grpo')

dataset = MathDataset("/kaggle/input/math-grpo/math_tasks.jsonl")

for question, answer in dataset:
    loss = trainer.train_step(question, answer)
    print(f"Step {trainer.step_count}: Loss {loss:.4f}")
    writer.add_scalar("Loss/train", loss, trainer.step_count)
    writer.flush()

writer.close()  
