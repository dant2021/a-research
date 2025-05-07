#!/usr/bin/env python3
"""

Extract image embeddings from Qwen2.5‑VL‑3B‑Instruct.

"""

from sklearn.manifold import TSNE
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from PIL import Image, ImageFont, ImageDraw
import os


def extract_embedding_from_image(image, model, processor, device, path):

    # 1. Create chat template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        },
    ]
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2. Process both text and image
    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt"
    ).to(device)

    # 3. Forward pass (pass all relevant tensors)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            pixel_values=inputs.pixel_values,
            image_grid_thw=inputs.image_grid_thw,
            return_dict=True,
            output_hidden_states=True,  # To get all hidden states
        )

    # 4. Extract the image embedding
    # The embedding for the image token is at position 0 in the sequence
    last_hidden = outputs.hidden_states[-1]  # shape: (batch, seq_len, hidden_size)
    # save to disk
    out_path = f"debug/image_embedding_{path}.pt"
    torch.save(last_hidden.cpu(), out_path)
    print(f"Saved image embedding (shape={last_hidden.shape}) to {out_path}")
    return last_hidden


def create_font_grid(font_path, chars, grid_size=8, font_size=64):
    """Generate grid image of font characters with validation"""
    print(f"\nCreating font grid for: {font_path}")
    print(f"Grid size: {grid_size}x{grid_size}, Font size: {font_size}px")
    
    try:
        print("Loading font...")
        font = ImageFont.truetype(font_path, font_size-4)
        print(f"Font loaded: {font.getname()}")
    except Exception as e:
        print(f"Font loading error: {str(e)}")
        return None

    img_size = font_size * grid_size
    print(f"Creating image canvas: {img_size}x{img_size}px")
    img = Image.new('RGB', (img_size, img_size), 'white')
    draw = ImageDraw.Draw(img)

    print("Rendering characters...")
    for i, char in enumerate(chars):
        if i >= grid_size**2:
            print(f"Warning: Too many characters ({len(chars)}), truncating to {grid_size**2}")
            break
            
        row, col = divmod(i, grid_size)
        x = col * font_size 
        y = row * font_size 
        draw.text((x, y), char, font=font, fill='black')

    print(f"Rendered {min(len(chars), grid_size**2)} characters")
    font_name = os.path.basename(font_path).replace('.ttf', '')
    path = f"debug/font_grid_{font_name}.png"
    img.save(path)
    return img

def visualize_embeddings(dict_embeddings):
    """
    Visualize font embeddings using UMAP.
    Each font is represented by the first 4 principal components 
    of their embeddings to capture the most variance.
    """
    import numpy as np
    import umap
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    for path, embedding in dict_embeddings.items():
        print(f"path: {path}")
        print(f"embedding.shape: {embedding.shape}")
        #print(f"embedding.mean: {torch.mean(embedding, dim=0)}")
        #print(f"embedding.l2: {torch.norm(embedding, p=2, dim=1)}")
    
    # Extract embedding features from each font
    font_features = []
    font_names = []
    
    for font_name, embedding in dict_embeddings.items():
        # Take the mean across the sequence dimension to get font-level features
        # embedding shape: [1, seq_len, 2048]
        
        avg_embedding = embedding.squeeze(0).float().mean(dim=0).cpu().numpy()  # [2048]
        font_features.append(avg_embedding)
        font_names.append(font_name)
    
    # Convert to numpy array
    font_features = np.array(font_features)
    
    # Use PCA to reduce dimensionality to manageable components first
    pca = PCA(n_components=min(4, len(font_features)))
    reduced_features = pca.fit_transform(font_features)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Apply UMAP for visualization
    reducer = umap.UMAP(
        n_neighbors=min(5, len(font_features)-1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    #embedding_2d = reducer.fit_transform(reduced_features)
    tsne = TSNE(n_components=2, perplexity=min(5, len(font_features)-1), random_state=42)
    embedding_2d = tsne.fit_transform(reduced_features)
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot each font as a point
    for i, font_name in enumerate(font_names):
        plt.scatter(embedding_2d[i, 0], embedding_2d[i, 1], s=100, alpha=0.8)
        plt.text(embedding_2d[i, 0]+0.01, embedding_2d[i, 1]+0.01, font_name, 
                 fontsize=12, fontweight='bold')
    
    plt.title("UMAP Visualization of Font Embeddings")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("font_embeddings_umap.png", dpi=300, bbox_inches='tight')
    plt.show()

    

if __name__ == "__main__":
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Qwen2_5_VLProcessor.from_pretrained(model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    dict_embeddings = {}

    fonts = [
        # Sans-serif fonts
        "arial", "arialbd", "ariali",  # Regular, Bold, Italic
        "calibri", "calibrib", "calibril",  # Regular, Bold, Light
        "tahoma", "tahomabd",  # Regular, Bold
        "verdana", "verdanab",  # Regular, Bold
        "segoeui", "segoeuib", "segoeuil",  # Regular, Bold, Light
        
        # Serif fonts
        "georgia", "georgiab",  # Regular, Bold
        "times", "timesbd",  # Regular, Bold
        
        # Monospace fonts
        "consola", "consolab",  # Regular, Bold
        "cour", "courbd",  # Regular, Bold
        
        # Decorative/Display fonts
        "comic", "comicbd",  # Regular, Bold
        "impact"  # Display font
    ]

    for path in fonts:
        font_path=f"yourpath/{path}.ttf"
        chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

        img = create_font_grid(font_path, chars, 8, 64)
        if img is not None:
            last_hidden = extract_embedding_from_image(img, model, processor, device, path)

            print(f"path: {path}")

            dict_embeddings[path] = last_hidden
    visualize_embeddings(dict_embeddings)
