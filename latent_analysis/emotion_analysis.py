import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from datasets import load_dataset
from zonos.model import Zonos
from zonos.utils import DEFAULT_DEVICE as device
import io
import librosa
from scipy.spatial.distance import cosine
import umap

# Load the Zonos model
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)

# Load the RAVDESS dataset from Hugging Face
ravdess = load_dataset("narad/ravdess")

# Get the train split (the only split in this dataset)
dataset = ravdess["train"]

# Lists to store data
embeddings = []
emotions = []
actors = []
intensities = []

# Define minimum audio length (in samples)
MIN_AUDIO_LENGTH = 16000  # 1 second at 16kHz

# Process all audio files in the dataset
for i, sample in enumerate(dataset):
    audio = sample["audio"]
    emotion = sample["labels"]
    actor_id = sample["speaker_id"]
    gender = sample["speaker_gender"]
    
    # Convert emotion index to name
    emotion_names = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    emotion_name = emotion_names[emotion]
    
    # Extract intensity and statement from filename
    filename = os.path.basename(audio["path"])
    parts = filename.split("-")
    intensity = "normal" if parts[3] == "01" else "strong"
    
    print(f"Processing {i+1}/{len(dataset)}: {filename} | {emotion_name} | Actor {actor_id}")
    
    try:
        # Load the audio data
        if "array" in audio and audio["array"] is not None:
            waveform_np = audio["array"].astype(np.float32)  # Convert to float32
            sample_rate = audio["sampling_rate"]
        else:
            # Load from file with explicit dtype
            waveform_np, sample_rate = librosa.load(audio["path"], sr=None, dtype=np.float32)
        
        # Ensure minimum length
        if len(waveform_np) < MIN_AUDIO_LENGTH:
            # Pad with silence if too short
            padding = np.zeros(MIN_AUDIO_LENGTH - len(waveform_np), dtype=np.float32)
            waveform_np = np.concatenate([waveform_np, padding])
        
        # Convert to torch tensor with explicit float32 type
        waveform = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform).to(torch.float32)  # Ensure float32 after resampling
            sample_rate = 16000
        
        # Extract speaker embedding and explicitly convert to Float32
        speaker = model.make_speaker_embedding(waveform, sample_rate)
        # Convert from BFloat16 to Float32 before any other operations
        speaker = speaker.to(torch.float32)
        speaker_np = speaker.squeeze().cpu().detach().numpy()
        embeddings.append(speaker_np)
        
        # Store metadata
        emotions.append(emotion_name)
        actors.append(f"{actor_id}-{gender}")
        intensities.append(intensity)
        
        print(f"Successfully processed sample {i}")
        
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

# Check if we have any embeddings
if len(embeddings) == 0:
    print("No embeddings were successfully processed. Cannot generate visualizations.")
else:
    # Convert to numpy arrays
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create a dictionary to store embeddings by actor
    actor_embeddings = {}
    for i, actor in enumerate(actors):
        if actor not in actor_embeddings:
            actor_embeddings[actor] = []
        actor_embeddings[actor].append(embeddings_array[i])
    
    # Calculate average embedding for each actor
    actor_averages = {}
    for actor, actor_embs in actor_embeddings.items():
        actor_averages[actor] = np.mean(actor_embs, axis=0)
    
    # Normalize embeddings by subtracting the actor's average
    normalized_embeddings = []
    for i, emb in enumerate(embeddings_array):
        actor = actors[i]
        normalized_emb = emb - actor_averages[actor]
        normalized_embeddings.append(normalized_emb)
    
    # Group normalized embeddings by emotion
    emotion_normalized_embeddings = {}
    for i, emotion in enumerate(emotions):
        if emotion not in emotion_normalized_embeddings:
            emotion_normalized_embeddings[emotion] = []
        emotion_normalized_embeddings[emotion].append(normalized_embeddings[i])
    
    # Calculate average embedding for each emotion
    emotion_averages = {}
    for emotion, embs in emotion_normalized_embeddings.items():
        emotion_averages[emotion] = np.mean(embs, axis=0)
    
    # Calculate cosine similarity between emotion averages
    emotion_similarity = {}
    unique_emotions = sorted(list(emotion_averages.keys()))
    
    print("\nCosine Similarity Between Emotions (1.0 = identical, 0.0 = completely different):")
    for i, emotion1 in enumerate(unique_emotions):
        for emotion2 in unique_emotions[i:]:  # Start from i to avoid duplicates
            similarity = 1.0 - cosine(emotion_averages[emotion1], emotion_averages[emotion2])
            print(f"{emotion1} vs {emotion2}: {similarity:.3f}")
            emotion_similarity[(emotion1, emotion2)] = similarity
    
    # Use normalized embeddings for TSNE
    normalized_embeddings_array = np.array(normalized_embeddings, dtype=np.float32)
    
    # Apply both TSNE and UMAP
    print("Running dimensionality reduction...")
    
    # t-SNE
    perplexity = min(30, max(1, len(normalized_embeddings)-1))
    print(f"Using perplexity value: {perplexity}")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_tsne = tsne.fit_transform(normalized_embeddings_array)
    
    # UMAP
    reducer = umap.UMAP(random_state=42)
    embeddings_umap = reducer.fit_transform(normalized_embeddings_array)
    
    # After normalization and UMAP calculation:
    if len(embeddings) > 0:
        # Create color map
        unique_emotions = sorted(list(set(emotions)))
        emotion_colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_emotions)))
        emotion_color_map = {emotion: emotion_colors[i] for i, emotion in enumerate(unique_emotions)}

        # Simplified UMAP visualization
        plt.figure(figsize=(12, 8))
        for i, (x, y) in enumerate(embeddings_umap):
            plt.scatter(x, y, color=emotion_color_map[emotions[i]], 
                       alpha=0.7, s=50, edgecolor='none')
        plt.title("UMAP Projection of Speaker-Normalized Embeddings")
        plt.savefig("umap_embeddings.png")
        plt.close()

        # Mirror pattern investigation
        plt.figure(figsize=(12, 8))
        for i, (x, y) in enumerate(embeddings_umap):
            # Color by emotion, shape by gender
            marker = 'o' if 'female' in actors[i] else 's'
            plt.scatter(x, y, color=emotion_color_map[emotions[i]],
                       marker=marker, s=60, alpha=0.7, 
                       edgecolor='black' if intensities[i] == 'strong' else 'none')
        
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Female',
                  markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='s', color='w', label='Male',
                  markerfacecolor='grey', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Strong Intensity',
                  markerfacecolor='none', markeredgecolor='black', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title("Emotion Embeddings: Shape=Gender, Border=Intensity")
        plt.savefig("mirror_pattern_analysis.png")
        plt.close()

        # Add after UMAP visualization
        print("\nGender-Emotion Similarity Analysis:")

        # Create gender-emotion groups
        gender_emotion_embeddings = {}
        for i, (emotion, actor) in enumerate(zip(emotions, actors)):
            gender = 'female' if 'female' in actor else 'male'
            key = (gender, emotion)
            if key not in gender_emotion_embeddings:
                gender_emotion_embeddings[key] = []
            gender_emotion_embeddings[key].append(normalized_embeddings[i])

        # Calculate average embeddings for each group
        gender_emotion_averages = {}
        for key, embs in gender_emotion_embeddings.items():
            if len(embs) >= 5:  # Only consider groups with at least 5 samples
                gender_emotion_averages[key] = np.mean(embs, axis=0)

        # Compare all pairs
        print("\n{:<25} {:<25} {:<10}".format("Group 1", "Group 2", "Similarity"))
        for (gender1, emotion1), avg1 in gender_emotion_averages.items():
            for (gender2, emotion2), avg2 in gender_emotion_averages.items():
                if (gender1, emotion1) < (gender2, emotion2):  # Avoid duplicates
                    similarity = 1 - cosine(avg1, avg2)
                    print("{:<25} {:<25} {:.3f}".format(
                        f"{gender1} {emotion1}", 
                        f"{gender2} {emotion2}", 
                        similarity
                    ))
