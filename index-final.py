#!/usr/bin/env python3
"""
FAST QDRANT INDEXING SCRIPT FOR JEWELRY SEARCH
Uses GPU batching for 3-5x speedup over single-image processing
Matches training and search scripts for proper model access
"""

import torch
from transformers import AutoProcessor, AutoModel
from peft import PeftModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image
import pandas as pd
from tqdm import tqdm
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import time

class Config:
    # Model configuration
    MODEL_NAME = "google/siglip-so400m-patch14-384"
    LORA_WEIGHTS_PATH = "/workspace/models/siglip2-jewelry-lora-final/best"
    
    # Data paths
    CSV_PATH = "/workspace/datasets/jewelry/siglip_training_metadata.csv"
    IMAGE_BASE_DIR = "/workspace/multimodal-dataset/"
    
    # Qdrant configuration
    QDRANT_PATH = "/workspace/qdrant_db_jewelry_v2"
    COLLECTION_NAME = "jewelry_collection"
    
    # Processing configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Performance tuning
    BATCH_SIZE = 64              # Number of images to process at once on GPU
    UPLOAD_BATCH_SIZE = 500      # Number of points to upload to Qdrant at once
    NUM_WORKERS = 8              # Number of parallel threads for image loading
    
    # Model selection
    USE_LORA = True              # Set to False to use base model only

class FastJewelryIndexer:
    def __init__(self):
        print("=" * 80)
        print("🚀 FAST JEWELRY INDEXING WITH GPU BATCHING")
        print("=" * 80)
        
        # Load processor
        print("\n1️⃣  Loading Processor...")
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        print("   ✅ Processor loaded")
        
        # Load model
        print("\n2️⃣  Loading Model...")
        base_model = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16 if Config.DEVICE == "cuda" else torch.float32
        ).to(Config.DEVICE)
        
        # Load LoRA if available
        if Config.USE_LORA and os.path.exists(Config.LORA_WEIGHTS_PATH):
            print(f"   📦 Loading LoRA from: {Config.LORA_WEIGHTS_PATH}")
            self.model = PeftModel.from_pretrained(
                base_model, 
                Config.LORA_WEIGHTS_PATH
            ).to(Config.DEVICE)
            print("   ✅ Fine-tuned LoRA model loaded")
        else:
            if Config.USE_LORA:
                print(f"   ⚠️  LoRA not found at: {Config.LORA_WEIGHTS_PATH}")
            print("   ✅ Using base model")
            self.model = base_model
        
        self.model.eval()
        
        # Connect to Qdrant
        print("\n3️⃣  Connecting to Qdrant...")
        self.client = QdrantClient(path=Config.QDRANT_PATH)
        print(f"   ✅ Connected to: {Config.QDRANT_PATH}")
        
        print("\n" + "=" * 80)
        print("✅ READY TO INDEX!")
        print("=" * 80 + "\n")

    def _extract_embeddings(self, outputs):
        """Helper to extract embeddings from model outputs"""
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state[:, 0]
        else:
            return outputs

    def load_single_image(self, path):
        """Load a single image safely"""
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            return None

    def load_images_parallel(self, paths):
        """Load multiple images in parallel using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
            images = list(executor.map(self.load_single_image, paths))
        
        # Filter out None values (failed loads)
        valid_images = []
        valid_indices = []
        for i, img in enumerate(images):
            if img is not None:
                valid_images.append(img)
                valid_indices.append(i)
        
        return valid_images, valid_indices

    def get_image_embeddings_batch(self, images):
        """
        Process a batch of images on GPU and return embeddings
        This is the key optimization - processes multiple images at once
        """
        if not images:
            return None
        
        try:
            # Process all images at once with the processor
            inputs = self.processor(images=images, return_tensors="pt").to(Config.DEVICE)
            
            with torch.no_grad():
                # Get image features using the correct method
                outputs = self.model.get_image_features(**inputs)
                
                # Extract embeddings
                embeddings = self._extract_embeddings(outputs)
                
                # Normalize
                normalized = torch.nn.functional.normalize(embeddings, dim=-1)
                
                # Return as numpy array
                return normalized.cpu().numpy()
                
        except Exception as e:
            print(f"\n   ⚠️  Batch processing error: {e}")
            return None

    def create_collection(self, vector_size):
        """Create or recreate the Qdrant collection"""
        # Try to delete existing collection
        try:
            self.client.delete_collection(collection_name=Config.COLLECTION_NAME)
            print(f"🗑️  Deleted existing collection: {Config.COLLECTION_NAME}")
        except:
            print(f"ℹ️  No existing collection to delete")
        
        # Create new collection
        self.client.create_collection(
            collection_name=Config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Created collection: {Config.COLLECTION_NAME} (dimension={vector_size})")

    def index_dataset(self):
        """Main indexing function with batched GPU processing"""
        
        # Load dataset
        print("📂 Loading dataset...")
        df = pd.read_csv(Config.CSV_PATH)
        total_records = len(df)
        print(f"   Total records: {total_records:,}")
        
        # Get vector dimension from a sample
        print("\n🔍 Determining vector dimension...")
        sample_image = Image.new('RGB', (384, 384))  # Create dummy image
        sample_embeddings = self.get_image_embeddings_batch([sample_image])
        
        if sample_embeddings is None:
            print("❌ Failed to get sample embeddings!")
            return 0
        
        vector_size = sample_embeddings.shape[1]
        print(f"   Vector dimension: {vector_size}")
        
        # Create collection
        print("\n🏗️  Creating Qdrant collection...")
        self.create_collection(vector_size)
        
        # Initialize counters
        points_buffer = []
        total_indexed = 0
        total_failed = 0
        
        # Start timer
        start_time = time.time()
        
        # Process in batches
        print(f"\n⚙️  Indexing {total_records:,} items in batches of {Config.BATCH_SIZE}...")
        print(f"   Upload batch size: {Config.UPLOAD_BATCH_SIZE}")
        print(f"   Parallel workers: {Config.NUM_WORKERS}")
        
        num_batches = (total_records + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
        
        with tqdm(total=total_records, desc="Indexing", unit="items") as pbar:
            for batch_start in range(0, total_records, Config.BATCH_SIZE):
                batch_end = min(batch_start + Config.BATCH_SIZE, total_records)
                batch_df = df.iloc[batch_start:batch_end]
                
                # Prepare batch data
                batch_paths = []
                batch_indices = []
                batch_payloads = []
                
                for idx, row in batch_df.iterrows():
                    # Get image path
                    image_path = str(row['image_path'])
                    if not image_path.startswith('/'):
                        image_path = os.path.join(Config.IMAGE_BASE_DIR, image_path)
                    
                    # Check if file exists
                    if not os.path.exists(image_path):
                        total_failed += 1
                        continue
                    
                    # Add to batch
                    batch_paths.append(image_path)
                    batch_indices.append(idx)
                    
                    # Prepare payload
                    payload = {
                        'product_id': str(row.get('product_id', f'item_{idx}')),
                        'path': image_path,
                        'semantic_description': str(row.get('description', ''))
                    }
                    batch_payloads.append(payload)
                
                if not batch_paths:
                    pbar.update(len(batch_df))
                    continue
                
                # Load images in parallel
                loaded_images, valid_indices = self.load_images_parallel(batch_paths)
                
                if not loaded_images:
                    total_failed += len(batch_paths)
                    pbar.update(len(batch_df))
                    continue
                
                # Get embeddings for the entire batch (GPU acceleration!)
                batch_embeddings = self.get_image_embeddings_batch(loaded_images)
                
                if batch_embeddings is None:
                    total_failed += len(loaded_images)
                    pbar.update(len(batch_df))
                    continue
                
                # Create points for valid embeddings
                for i, valid_idx in enumerate(valid_indices):
                    if i < len(batch_embeddings):
                        point = PointStruct(
                            id=batch_indices[valid_idx],
                            vector=batch_embeddings[i].tolist(),
                            payload=batch_payloads[valid_idx]
                        )
                        points_buffer.append(point)
                
                # Update failed count
                total_failed += (len(batch_paths) - len(loaded_images))
                
                # Upload to Qdrant when buffer is full
                if len(points_buffer) >= Config.UPLOAD_BATCH_SIZE:
                    try:
                        self.client.upsert(
                            collection_name=Config.COLLECTION_NAME,
                            points=points_buffer
                        )
                        total_indexed += len(points_buffer)
                        points_buffer = []
                    except Exception as e:
                        print(f"\n   ⚠️  Upload error: {e}")
                
                # Update progress bar
                pbar.update(len(batch_df))
        
        # Upload remaining points
        if points_buffer:
            try:
                self.client.upsert(
                    collection_name=Config.COLLECTION_NAME,
                    points=points_buffer
                )
                total_indexed += len(points_buffer)
            except Exception as e:
                print(f"\n   ⚠️  Final upload error: {e}")
        
        # Calculate stats
        elapsed_time = time.time() - start_time
        items_per_second = total_indexed / elapsed_time if elapsed_time > 0 else 0
        
        # Get final collection info
        try:
            collection_info = self.client.get_collection(Config.COLLECTION_NAME)
            final_count = collection_info.points_count
        except:
            final_count = total_indexed
        
        # Print summary
        print("\n" + "=" * 80)
        print("✅ INDEXING COMPLETE!")
        print("=" * 80)
        print(f"   📊 Total indexed:     {total_indexed:,} items")
        print(f"   ❌ Failed:            {total_failed:,} items")
        print(f"   💾 Collection count:  {final_count:,} points")
        print(f"   ⏱️  Time elapsed:      {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        print(f"   🚀 Speed:             {items_per_second:.1f} items/second")
        print(f"   💾 Database:          {Config.QDRANT_PATH}")
        print("=" * 80)
        
        return total_indexed

    def test_search(self, query="gold necklace", top_k=5):
        """Test the indexed collection with a sample query"""
        print(f"\n🧪 Testing search: '{query}'")
        
        # Get text embedding for query
        try:
            inputs = self.processor(
                text=query,
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(Config.DEVICE)
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                embeddings = self._extract_embeddings(outputs)
                query_vector = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy().squeeze()
        except Exception as e:
            print(f"❌ Failed to create query embedding: {e}")
            return
        
        # Search
        try:
            results = self.client.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=query_vector.tolist(),
                limit=top_k
            ).points
            
            print(f"\n🏆 Top {top_k} Results:")
            for i, result in enumerate(results, 1):
                prod_id = result.payload.get('product_id', 'N/A')
                score = result.score
                print(f"   {i}. Score: {score:.4f} | ID: {prod_id}")
            
        except Exception as e:
            print(f"❌ Search failed: {e}")

def main():
    print("\n" + "=" * 80)
    print("FAST JEWELRY SEARCH INDEXING")
    print("GPU-Accelerated Batch Processing")
    print("=" * 80 + "\n")
    
    # Initialize indexer
    indexer = FastJewelryIndexer()
    
    # Index the dataset
    total_indexed = indexer.index_dataset()
    
    # Run test searches if successful
    if total_indexed > 0:
        print("\n" + "=" * 80)
        print("RUNNING TEST SEARCHES")
        print("=" * 80)
        
        indexer.test_search(query="gold necklace", top_k=5)
        indexer.test_search(query="diamond ring", top_k=5)
        indexer.test_search(query="silver earrings", top_k=5)
        
        print("\n" + "=" * 80)
        print(f"✅ SUCCESS! Indexed {total_indexed:,} items")
        print(f"You can now use the search script to query the database.")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("❌ INDEXING FAILED - No items were indexed")
        print("Please check the errors above and try again.")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()