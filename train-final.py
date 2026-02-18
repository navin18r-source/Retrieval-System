import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import math
import gc
from sklearn.model_selection import train_test_split

class Config:
    MODEL_NAME = "google/siglip-so400m-patch14-384"
    TRAINING_DATA_CSV = "/workspace/datasets/jewelry/train.csv"
    IMAGE_BASE_DIR = "/workspace/multimodal-dataset/"
    OUTPUT_DIR = "/workspace/models/siglip2-jewelry-lora-final"
    
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    EPOCHS = 5
    BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 4
    
    LEARNING_RATE = 1e-4
    LR_TEMPERATURE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_TEXT_LENGTH = 64
    MIXED_PRECISION = True
    VAL_SPLIT = 0.05

# Set CUDA memory allocation config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class JewelryDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_path = str(row['image_path'])
            
            # Handle path construction
            if not image_path.startswith("/") and not image_path.startswith("http"):
                image_path = os.path.join(Config.IMAGE_BASE_DIR, image_path)
            
            # Validate image exists
            if not os.path.exists(image_path):
                return None
            
            image = Image.open(image_path).convert('RGB')
            caption = str(row['description'])
            
            # Validate caption
            if not caption or caption == 'nan' or len(caption.strip()) == 0:
                return None
            
            # Use processor for BOTH image and text
            image_inputs = self.processor(images=image, return_tensors="pt")
            text_inputs = self.processor(
                text=caption, 
                padding="max_length", 
                truncation=True, 
                max_length=Config.MAX_TEXT_LENGTH, 
                return_tensors="pt"
            )
            
            # Get input_ids from text_inputs
            input_ids = text_inputs['input_ids'].squeeze(0)
            
            # Create attention_mask (SigLIP doesn't return it, so we create it)
            # For SigLIP, we can use all 1s since it doesn't use padding
            attention_mask = torch.ones_like(input_ids)
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        except Exception as e:
            # Silently skip problematic samples
            return None
    
    @staticmethod
    def collate_fn(batch):
        # Filter out None values
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        
        return {
            'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
            'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        }

class SigLIPLoss(nn.Module):
    """SigLIP loss with proper numerical stability"""
    def __init__(self, init_temperature=10.0, init_bias=-10.0):
        super().__init__()
        # Use log of temperature for numerical stability
        self.log_temperature = nn.Parameter(torch.tensor(math.log(init_temperature)))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        
    def forward(self, img_emb, txt_emb):
        # Normalize embeddings
        img_emb = F.normalize(img_emb, p=2, dim=-1)
        txt_emb = F.normalize(txt_emb, p=2, dim=-1)
        
        # Compute similarity matrix
        batch_size = img_emb.shape[0]
        temperature = self.log_temperature.exp()
        
        # Compute logits
        logits = (img_emb @ txt_emb.T) * temperature + self.bias
        
        # Create labels: +1 for diagonal, -1 for off-diagonal
        labels = 2 * torch.eye(batch_size, device=logits.device, dtype=logits.dtype) - 1
        
        # SigLIP loss
        loss = -F.logsigmoid(labels * logits).mean()
        
        return loss

class Trainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Using device: {self.device}")
        
        # Load processor (handles both image and text for SigLIP)
        print(f"📦 Loading processor from {Config.MODEL_NAME}")
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        
        # Load base model
        print(f"📦 Loading base model...")
        base_model = AutoModel.from_pretrained(Config.MODEL_NAME)
        
        # Configure LoRA
        print("🔧 Configuring LoRA...")
        lora_config = LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.LORA_TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False  # CRITICAL: Must be False for training
        )
        
        # Apply LoRA
        print("✨ Applying LoRA adapters...")
        self.model = get_peft_model(base_model, lora_config)
        self.model.to(self.device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Initialize loss function
        self.loss_fn = SigLIPLoss().to(self.device)
        
        # Optimizer with separate learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': [p for p in self.model.parameters() if p.requires_grad], 'lr': Config.LEARNING_RATE},
            {'params': list(self.loss_fn.parameters()), 'lr': Config.LR_TEMPERATURE},
        ], weight_decay=Config.WEIGHT_DECAY)
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if Config.MIXED_PRECISION and self.device == "cuda" else None
        
        print(f"✅ Trainer initialized")

    def extract_embeddings(self, outputs):
        """Extract embeddings from model outputs"""
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0]
        else:
            return outputs

    def train(self, train_loader, val_loader):
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        # Calculate total steps for scheduler
        total_steps = (len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS) * Config.EPOCHS
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[Config.LEARNING_RATE, Config.LR_TEMPERATURE],
            total_steps=total_steps,
            pct_start=Config.WARMUP_RATIO,
            anneal_strategy='cos'
        )
        
        best_loss = float('inf')
        
        print("\n" + "=" * 80)
        print("🎯 Starting Training")
        print("=" * 80 + "\n")
        
        for epoch in range(Config.EPOCHS):
            self.model.train()
            self.loss_fn.train()
            
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
            
            for idx, batch in enumerate(pbar):
                if batch is None:
                    continue
                
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass with mixed precision
                with torch.amp.autocast(self.device, enabled=Config.MIXED_PRECISION):
                    # Get embeddings - Call model directly (LoRA is applied automatically)
                    vision_outputs = self.model.get_image_features(pixel_values=pixel_values)
                    text_outputs = self.model.get_text_features(
                        input_ids=input_ids, 
                        attention_mask=attention_mask
                    )
                    
                    # Extract embeddings
                    vision_emb = self.extract_embeddings(vision_outputs)
                    text_emb = self.extract_embeddings(text_outputs)
                    
                    # Compute loss
                    loss = self.loss_fn(vision_emb, text_emb)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / Config.GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass
                if self.scaler:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Update weights
                if (idx + 1) % Config.GRADIENT_ACCUMULATION_STEPS == 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()
                    
                    scheduler.step()
                    self.optimizer.zero_grad()
                
                # Track metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg': f"{epoch_loss/num_batches:.4f}",
                    'temp': f"{self.loss_fn.log_temperature.exp().item():.2f}"
                })
            
            # Validation
            val_loss = self.validate(val_loader)
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS}:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"   Temp:       {self.loss_fn.log_temperature.exp().item():.2f}")
            print(f"   Bias:       {self.loss_fn.bias.item():.2f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save(os.path.join(Config.OUTPUT_DIR, "best"))
                print(f"   ✅ New best model saved!")
            
            # Save epoch checkpoint
            self.save(os.path.join(Config.OUTPUT_DIR, f"epoch_{epoch+1}"))
            
            # Clear cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            print()

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        self.loss_fn.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(loader, desc="Validating", leave=False):
            if batch is None:
                continue
            
            # Move to device
            pixel_values = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            vision_outputs = self.model.get_image_features(pixel_values=pixel_values)
            text_outputs = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Extract embeddings
            vision_emb = self.extract_embeddings(vision_outputs)
            text_emb = self.extract_embeddings(text_outputs)
            
            # Compute loss
            loss = self.loss_fn(vision_emb, text_emb)
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def save(self, path):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA adapter weights
        self.model.save_pretrained(path)
        
        # Save processor
        self.processor.save_pretrained(path)
        
        # Save loss parameters
        torch.save({
            'log_temperature': self.loss_fn.log_temperature.item(),
            'bias': self.loss_fn.bias.item()
        }, os.path.join(path, "loss_params.pt"))

def main():
    print("=" * 80)
    print("🚀 SigLIP LoRA Fine-tuning for Jewelry Search")
    print("=" * 80)
    
    # Load data
    print(f"\n📂 Loading data from: {Config.TRAINING_DATA_CSV}")
    df = pd.read_csv(Config.TRAINING_DATA_CSV)
    print(f"   Total samples: {len(df):,}")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=Config.VAL_SPLIT, random_state=42)
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    # Initialize trainer
    trainer = Trainer()
    
    # Create datasets
    print("\n📦 Creating datasets...")
    train_ds = JewelryDataset(train_df, trainer.processor)
    val_ds = JewelryDataset(val_df, trainer.processor)
    
    # Create dataloaders
    print("🔄 Creating dataloaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=JewelryDataset.collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=JewelryDataset.collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"   Train batches: {len(train_loader):,}")
    print(f"   Val batches:   {len(val_loader):,}")
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print(f"📁 Models saved to: {Config.OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()