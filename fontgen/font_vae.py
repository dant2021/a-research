import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from fontTools.ttLib import TTFont
from tqdm import tqdm
import freetype
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Constants
GLYPH_SIZE = 64  # Size of glyph image
BASIC_LATIN = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
BATCH_SIZE = 1  # Reduced batch size since we have fewer fonts
LATENT_DIM = 256
LEARNING_RATE = 1e-4

class FontDataset(Dataset):
    def __init__(self, font_dir='extracted_fonts', glyph_size=GLYPH_SIZE):
        self.font_dir = Path(font_dir)
        self.glyph_size = glyph_size
        # Include both .ttf and .otf files
        self.fonts = list(self.font_dir.glob('*.ttf')) + list(self.font_dir.glob('*.otf'))
        self.char_to_idx = {char: idx for idx, char in enumerate(BASIC_LATIN)}
        print(f"Found {len(self.fonts)} fonts in {font_dir}")
        
    def __len__(self):
        return len(self.fonts)
    
    def render_glyph(self, font_path, char):
        """Render a single glyph as a bitmap image"""
        try:
            face = freetype.Face(str(font_path))
            face.set_pixel_sizes(0, self.glyph_size)
            
            # Load the character with FT_LOAD_RENDER flag to get bitmap
            face.load_char(char, freetype.FT_LOAD_RENDER)
            
            # Get the glyph's bitmap
            bitmap = face.glyph.bitmap
            
            # If no valid bitmap, return None
            if bitmap.width == 0 or bitmap.rows == 0:
                return None
                
            # Create an array from the bitmap
            image = np.array(bitmap.buffer, dtype=np.uint8).reshape(bitmap.rows, bitmap.width)
            
            # Pad or crop to make square of size glyph_size x glyph_size
            padded_image = np.zeros((self.glyph_size, self.glyph_size), dtype=np.float32)
            
            # Calculate centering offsets
            x_offset = (self.glyph_size - bitmap.width) // 2
            y_offset = (self.glyph_size - bitmap.rows) // 2
            
            # Ensure positive offsets
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)
            
            # Calculate the region to copy
            copy_width = min(bitmap.width, self.glyph_size)
            copy_height = min(bitmap.rows, self.glyph_size)
            
            # Copy the bitmap to the center of our square image
            padded_image[y_offset:y_offset+copy_height, x_offset:x_offset+copy_width] = image[:copy_height, :copy_width]
            
            # Normalize to [0, 1] range
            padded_image = padded_image / 255.0
            
            return padded_image
            
        except Exception as e:
            print(f"Error rendering glyph '{char}' from {font_path}: {str(e)}")
            return None
    
    def __getitem__(self, idx):
        font_path = self.fonts[idx]
        glyph_data = []
        glyph_ids = []
        
        for char in BASIC_LATIN:
            image = self.render_glyph(font_path, char)
            if image is not None:
                glyph_data.append(image)
                glyph_ids.append(self.char_to_idx[char])
                
        if not glyph_data:
            # Handle empty font by returning None and filtering in collate_fn
            print(f"Warning: No valid glyphs found in {font_path}")
            return None
            
        return {
            'images': torch.tensor(np.stack(glyph_data), dtype=torch.float32).unsqueeze(1),  # Add channel dimension [G, 1, H, W]
            'glyph_ids': torch.tensor(glyph_ids, dtype=torch.long),
            'font_path': str(font_path)
        }

def collate_fn(batch):
    """Filter None values and collate valid samples"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        'images': torch.stack([item['images'] for item in batch]),
        'glyph_ids': torch.stack([item['glyph_ids'] for item in batch]),
        'font_path': [item['font_path'] for item in batch]
    }

class FontVAE(nn.Module):
    def __init__(self, glyph_size=GLYPH_SIZE, latent_dim=LATENT_DIM):
        super().__init__()
        self.glyph_size = glyph_size
        self.latent_dim = latent_dim
        
        # Encoder - Using CNN for image processing
        self.encoder = nn.Sequential(
            # Input: [1, glyph_size, glyph_size]
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> [32, glyph_size/2, glyph_size/2]
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> [64, glyph_size/4, glyph_size/4]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> [128, glyph_size/8, glyph_size/8]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> [256, glyph_size/16, glyph_size/16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # Calculate the flattened size
        self.flatten_size = 256 * (glyph_size // 16) * (glyph_size // 16)
        
        # Latent space mapping
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Glyph embedding
        self.glyph_embedding = nn.Embedding(len(BASIC_LATIN), 64)
        
        # Decoder - input is latent vector + glyph embedding
        self.fc_decoder = nn.Linear(latent_dim + 64, self.flatten_size)
        
        self.decoder = nn.Sequential(
            # Input: [256, glyph_size/16, glyph_size/16]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # -> [128, glyph_size/8, glyph_size/8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> [64, glyph_size/4, glyph_size/4]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [32, glyph_size/2, glyph_size/2]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> [1, glyph_size, glyph_size]
            nn.Sigmoid()  # Output between [0, 1] for image data
        )
    
    def encode(self, x, glyph_ids):
        batch_size, num_glyphs, channels, h, w = x.shape
        
        # Process each glyph through the encoder
        x = x.reshape(-1, channels, h, w)  # [B*G, C, H, W]
        features = self.encoder(x)  # [B*G, 256, H/16, W/16]
        
        # Flatten
        features = features.view(-1, self.flatten_size)  # [B*G, flatten_size]
        
        # Get mu and logvar
        mu = self.fc_mu(features)  # [B*G, latent_dim]
        logvar = self.fc_logvar(features)  # [B*G, latent_dim]
        
        # Reshape to [batch, glyphs, latent_dim]
        mu = mu.view(batch_size, num_glyphs, -1)
        logvar = logvar.view(batch_size, num_glyphs, -1)
        
        return mu, logvar
    
    def decode(self, z, glyph_ids):
        batch_size, num_glyphs = glyph_ids.shape
        
        # Get glyph embeddings
        glyph_emb = self.glyph_embedding(glyph_ids)  # [B, G, 64]
        
        # Prepare inputs for decoder
        z = z.view(-1, self.latent_dim)  # [B*G, latent_dim]
        glyph_emb = glyph_emb.view(-1, glyph_emb.size(-1))  # [B*G, 64]
        decoder_input = torch.cat([z, glyph_emb], dim=1)  # [B*G, latent_dim+64]
        
        # Map to feature space
        features = self.fc_decoder(decoder_input)  # [B*G, flatten_size]
        features = features.view(-1, 256, self.glyph_size // 16, self.glyph_size // 16)  # [B*G, 256, H/16, W/16]
        
        # Decode to images
        images = self.decoder(features)  # [B*G, 1, H, W]
        
        # Reshape to [batch, glyphs, channels, height, width]
        return images.view(batch_size, num_glyphs, 1, self.glyph_size, self.glyph_size)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, glyph_ids):
        mu, logvar = self.encode(x, glyph_ids)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, glyph_ids), mu, logvar

def visualize_image_grid(images, title=None):
    """Visualize a grid of glyph images"""
    fig, axs = plt.subplots(1, len(images), figsize=(2*len(images), 2))
    if len(images) == 1:
        axs = [axs]
    
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze(), cmap='gray')
        axs[i].axis('off')
    
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def train_model():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = FontDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          collate_fn=collate_fn, num_workers=0)
    
    model = FontVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # For visualization
    sample_batch = next(iter(dataloader))
    if sample_batch:
        sample_images = sample_batch['images'][:1].to(device)  # First font only
        sample_glyph_ids = sample_batch['glyph_ids'][:1].to(device)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        batch_count = 0
        
        for batch in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            if batch is None:
                continue
                
            images = batch['images'].to(device)
            glyph_ids = batch['glyph_ids'].to(device)
            
            # Forward pass
            recon, mu, logvar = model(images, glyph_ids)

            recon_loss = F.binary_cross_entropy(recon, images, reduction='none')
            recon_loss = recon_loss.sum(dim=[1, 2, 3, 4]).mean()
            
            # KL loss - also properly normalized
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2]).mean()
            
            loss = recon_loss + kl_loss 
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        avg_recon_loss = total_recon_loss / batch_count
        avg_kl_loss = total_kl_loss / batch_count
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')
        
        # Visualize reconstruction every 10 epochs
        if (epoch + 1) % 10 == 0 and sample_batch:
            model.eval()
            with torch.no_grad():
                recon, _, _ = model(sample_images, sample_glyph_ids)
                
                # Visualize original vs reconstruction
                fig, axes = plt.subplots(2, 13, figsize=(20, 5))
                fig.suptitle(f'Epoch {epoch+1} Reconstruction')
                
                for i in range(min(13, sample_images.shape[1])):
                    # Original
                    orig_img = sample_images[0, i, 0].cpu().numpy()
                    axes[0, i].imshow(orig_img, cmap='gray')
                    axes[0, i].set_title(BASIC_LATIN[i])
                    axes[0, i].set_xticks([])
                    axes[0, i].set_yticks([])
                    
                    # Reconstruction
                    recon_img = recon[0, i, 0].cpu().numpy()
                    axes[1, i].imshow(recon_img, cmap='gray')
                    axes[1, i].set_xticks([])
                    axes[1, i].set_yticks([])
                
                plt.tight_layout()
                plt.savefig(f'recon_epoch_{epoch+1}.png')
                plt.close()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'font_vae_checkpoint_{epoch+1}.pt')

if __name__ == '__main__':
    train_model()
