import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

# Config
MODEL_NAME = "Qwen/Qwen3-0.6B"
LATENT_DIM = 128

# Continous or discrete?
CONTINUOUS = True

class LatentAugmentedQwen(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        
        # Define layer mappings: L5→L0, L6→L1, ..., L25→L20
        self.extractor_layers = list(range(5, 26))  # L5 to L25
        self.bias_layers = list(range(0, 21))       # L0 to L20
        
        # Layer-specific latent projection layers with PROPER INITIALIZATION
        self.latent_down_projs = nn.ModuleDict({
            str(layer): nn.Linear(hidden_size, LATENT_DIM) 
            for layer in self.extractor_layers
        })
        self.latent_up_projs = nn.ModuleDict({
            str(layer): nn.Linear(LATENT_DIM, hidden_size) 
            for layer in self.bias_layers
        })
        
        # Add gates for each up projection
        self.latent_gates = nn.ParameterDict({
            f"gate_{target_layer}": nn.Parameter(torch.zeros(1))  # Start at 0 (sigmoid(0) = 0.5)
            for target_layer in range(21)  # L0-L20
        })
        
        # CRITICAL: Initialize projection layers properly
        self._initialize_projection_layers()
        
        # Storage for latents
        self.latents_pass1 = {}      # Current token's latents from pass 1
        self.previous_latents = {}   # Previous token's latents (for inference)
        self.attention_weights = {}
        
        self.hook_handles = []
        self.training_mode = True    # Two passes during training
        
        # Add statistics collection
        self.collect_adjustment_stats = False
        self.adjustment_stats = {}
        
    def _initialize_projection_layers(self):
        """Initialize projection layers to start with minimal impact"""
        for layer_idx in self.extractor_layers:
            # Initialize down projection with small weights
            nn.init.xavier_normal_(self.latent_down_projs[str(layer_idx)].weight, gain=0.1)
            nn.init.zeros_(self.latent_down_projs[str(layer_idx)].bias)
        
        for layer_idx in self.bias_layers:
            # Initialize up projection to output near-zero initially
            nn.init.xavier_normal_(self.latent_up_projs[str(layer_idx)].weight, gain=0.05)
            nn.init.zeros_(self.latent_up_projs[str(layer_idx)].bias)
        
        print("✅ Initialized projection layers with small weights")
        
    def setup_hooks(self, pass_num=1):
        """Setup hooks for either pass 1 (extract) or pass 2 (bias)"""
        self.remove_hooks()
        
        if pass_num == 1:
            # Pass 1: Extract latents from L5, L6, ..., L25
            for layer_idx in self.extractor_layers:
                hook_handle = self.base_model.model.layers[layer_idx].register_forward_hook(
                    self.make_extract_latent_hook(layer_idx)
                )
                self.hook_handles.append(hook_handle)
                
        elif pass_num == 2:
            # Pass 2: Add latent bias to L0, L1, ..., L20
            for bias_layer_idx in self.bias_layers:
                extractor_layer_idx = bias_layer_idx + 5  # L5→L0, L6→L1, etc.
                if extractor_layer_idx <= 25:  # Make sure we don't go beyond L25
                    down_proj = self.base_model.model.layers[bias_layer_idx].mlp.down_proj
                    hook_handle = down_proj.register_forward_hook(
                        self.make_latent_bias_hook(bias_layer_idx, extractor_layer_idx)
                    )
                    self.hook_handles.append(hook_handle)
        
        # Always add attention hooks for visualization
        for i in range(len(self.base_model.model.layers)):
            attention_module = self.base_model.model.layers[i].self_attn
            hook_handle = attention_module.register_forward_hook(self.make_attention_hook(i))
            self.hook_handles.append(hook_handle)
    
    def make_extract_latent_hook(self, layer_idx):
        """Create hook for extracting latents during pass 1"""
        def hook(module, input, output):
            # Extract hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Project to latent space
            latent = self.latent_down_projs[str(layer_idx)](hidden_states)
            self.latents_pass1[layer_idx] = latent
            
            #print(f"Pass 1: Extracted latent from L{layer_idx}, shape: {latent.shape}")
            return output
        return hook
    
    def make_latent_bias_hook(self, bias_layer_idx, extractor_layer_idx):
        """Create hook for adding latent bias during pass 2"""
        def hook(module, input, output):
            # Get the appropriate latent
            if self.training_mode:
                # Training: use current token's pass 1 latents
                latent_source = self.latents_pass1
            else:
                # Inference: use previous token's latents
                latent_source = self.previous_latents
            
            if extractor_layer_idx in latent_source:
                latent = latent_source[extractor_layer_idx]
                latent_bias = self.latent_up_projs[str(bias_layer_idx)](latent)
                
                # Get the gate for this layer
                gate_key = f"gate_{bias_layer_idx}"
                if gate_key in self.latent_gates:
                    gate_value = torch.sigmoid(self.latent_gates[gate_key])  # 0-1 range
                    latent_bias = latent_bias * gate_value  # Scale bias by gate
                
                # Collect statistics if enabled
                if self.collect_adjustment_stats:
                    layer_key = f"L{extractor_layer_idx}→L{bias_layer_idx}"
                    
                    # Compute norms
                    latent_norm = torch.norm(latent).item()
                    bias_norm = torch.norm(latent_bias).item()
                    original_norm = torch.norm(output).item()
                    
                    # Apply adjustment
                    adjusted_output = output + latent_bias
                    adjusted_norm = torch.norm(adjusted_output).item()
                    
                    # Compute adjustment ratio (bias norm / original norm)
                    adjustment_ratio = bias_norm / (original_norm + 1e-8)
                    
                    # Store statistics
                    self.adjustment_stats[layer_key] = {
                        'latent_norm': latent_norm,
                        'bias_norm': bias_norm,
                        'original_norm': original_norm,
                        'adjusted_norm': adjusted_norm,
                        'adjustment_ratio': adjustment_ratio
                    }
                    
                    #print(f"Pass 2: Adding L{extractor_layer_idx}→L{bias_layer_idx} bias, "
                    #      f"bias_norm={bias_norm:.3f}, adj_ratio={adjustment_ratio:.3f}")
                    
                    return adjusted_output
                else:
                    #print(f"Pass 2: Adding L{extractor_layer_idx}→L{bias_layer_idx} bias, shape: {latent_bias.shape}")
                    return output + latent_bias
            else:
                #print(f"Pass 2: No latent available for L{extractor_layer_idx}→L{bias_layer_idx}")
                return output
        return hook
    
    def make_attention_hook(self, layer_idx):
        """Create hook for capturing attention patterns"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            self.attention_weights[f"layer_{layer_idx}"] = {
                "hidden_states": hidden_states.detach().cpu(),
                "layer_idx": layer_idx
            }
            return output
        return hook
    
    def forward(self, input_ids, attention_mask=None):
        if self.training_mode:
            return self.forward_two_pass(input_ids, attention_mask)
        else:
            return self.forward_single_pass(input_ids, attention_mask)
    
    def forward_two_pass(self, input_ids, attention_mask=None):
        """Training: Two-pass forward"""
        #print(f"\n=== TWO-PASS FORWARD ===")
        #print(f"Input shape: {input_ids.shape}")
        
        # Reset storage
        self.latents_pass1 = {}
        self.attention_weights = {}
        
        # Pass 1: Extract latents from L5-L25
        #print("\n--- PASS 1: Extracting Latents ---")
        self.setup_hooks(pass_num=1)
        _ = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pass 2: Use latents as bias in L0-L20
        #print("\n--- PASS 2: Applying Latent Bias ---")
        self.setup_hooks(pass_num=2)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        #print(f"Final output shape: {outputs.logits.shape}")
        #print(f"Latents extracted: {len(self.latents_pass1)}")
        
        return {
            "logits": outputs.logits,
            "latents_extracted": self.latents_pass1,
            "num_latents": len(self.latents_pass1)
        }
    
    def forward_single_pass(self, input_ids, attention_mask=None):
        """Inference: Single pass using previous latents"""
        print(f"\n=== SINGLE-PASS FORWARD (Inference) ===")
        print(f"Using {len(self.previous_latents)} previous latents")
        
        self.attention_weights = {}
        self.setup_hooks(pass_num=2)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        return {
            "logits": outputs.logits
        }
    
    def set_inference_mode(self, inference_mode=True):
        """Switch between training (two-pass) and inference (single-pass)"""
        self.training_mode = not inference_mode
        print(f"Set to {'inference' if inference_mode else 'training'} mode")
    
    def update_previous_latents(self, new_latents):
        """Update previous latents - extract only the last token's latents"""
        self.previous_latents = {}
        for layer_idx, latent in new_latents.items():
            # Extract only the last token's latent: [1, seq_len, 128] → [1, 1, 128]
            self.previous_latents[layer_idx] = latent[:, -1:, :].clone()  
        print(f"Updated previous latents: {len(self.previous_latents)} stored (last token only)")
    
    def get_trainable_parameters(self):
        """Get only the latent projection parameters for training"""
        trainable_params = []
        trainable_params.extend(self.latent_down_projs.parameters())
        trainable_params.extend(self.latent_up_projs.parameters())
        trainable_params.extend(self.latent_gates.parameters())
        return trainable_params
    
    def freeze_base_model(self):
        """Freeze all base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        print("Base model parameters frozen")
    
    def visualize_layer_mapping(self):
        """Visualize the layer mapping"""
        print("\n=== LAYER MAPPING ===")
        for bias_idx in self.bias_layers:
            extractor_idx = bias_idx + 5
            if extractor_idx <= 25:
                print(f"L{extractor_idx} → L{bias_idx}")
    
    def visualize_attention_norms(self, tokens=None):
        """Visualize the norms of hidden states across layers"""
        if not self.attention_weights:
            print("No attention data captured. Run a forward pass first.")
            return
        
        layers = sorted([int(k.split('_')[1]) for k in self.attention_weights.keys()])
        norms_per_layer = []
        
        for layer_idx in layers:
            hidden_states = self.attention_weights[f"layer_{layer_idx}"]["hidden_states"]
            norms = torch.norm(hidden_states, dim=-1).squeeze(0)
            norms_per_layer.append(norms.numpy())
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Heatmap
        norms_matrix = np.array(norms_per_layer)
        im1 = ax1.imshow(norms_matrix, aspect='auto', cmap='viridis')
        ax1.set_xlabel('Token Position')
        ax1.set_ylabel('Layer')
        ax1.set_title('Hidden State Norms Across Layers (Two-Pass Architecture)')
        plt.colorbar(im1, ax=ax1, label='Norm')
        
        # Mark bias receiving layers (0-20)
        for i in range(21):
            ax1.axhline(y=i, color='red', linestyle=':', alpha=0.3)
        ax1.text(0.02, 0.98, 'Red lines: Bias receiving layers (L0-L20)', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Plot 2: Average norms
        avg_norms = np.mean(norms_matrix, axis=1)
        ax2.plot(layers, avg_norms, 'o-', linewidth=2, markersize=6)
        
        # Mark layer ranges
        ax2.axvspan(0, 20, alpha=0.2, color='red', label='Bias Receivers (L0-L20)')
        ax2.axvspan(5, 25, alpha=0.2, color='blue', label='Latent Extractors (L5-L25)')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Average Hidden State Norm')
        ax2.set_title('Average Hidden State Norm per Layer')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        if tokens:
            print(f"Tokens: {tokens}")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
    
    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self.remove_hooks()

if __name__ == "__main__":
    model = LatentAugmentedQwen()
    tokenizer = model.tokenizer
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Freeze base model and show trainable parameters
    model.freeze_base_model()
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Show layer mapping
    model.visualize_layer_mapping()
    
    # Test training mode (two passes)
    text = "LoRA (Low-Rank Adaptation) and hooks in PyTorch serve different purposes and operate at different levels of abstraction. LoRA is a parameter-efficient fine-tuning technique that modifies model weights by adding low-rank decomposition matrices. Instead of updating all parameters during fine-tuning, LoRA freezes the original weights and learns small rank decomposition matrices (A and B) such that the weight update is W + BA, where B and A have much lower rank than the original weight matrix W. Hooks are PyTorch's mechanism for intercepting and modifying the forward and backward passes of neural network modules. They allow you to execute custom code at specific points during computation without modifying the model architecture directly."
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    print("\n" + "="*50)
    print("TESTING TRAINING MODE")
    print("="*50)
    out = model(**inputs)
    print("Logits shape:", out["logits"].shape)
    print("Latents extracted:", out["num_latents"])
    
    # Test inference mode (single pass with previous latents)
    print("\n" + "="*50)
    print("TESTING INFERENCE MODE")
    print("="*50)
    model.set_inference_mode(True)
    model.update_previous_latents(out["latents_extracted"])
    
    # New token for inference
    new_text = "This is a test."
    new_inputs = tokenizer(new_text, return_tensors="pt")
    out_inference = model(**new_inputs)
    print("Inference logits shape:", out_inference["logits"].shape)
    
    # Visualize
    model.visualize_attention_norms(tokens)
