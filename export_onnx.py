"""
ONNX Model Export Script
Exports sentence-transformers model to ONNX format with quantization.
"""

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

# Model to export
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = Path("models/semantic_onnx")

print(f"Exporting {MODEL_NAME} to ONNX...")
print(f"Output directory: {OUTPUT_DIR}")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load and export
try:
    # Export to ONNX with optimization
    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_NAME,
        export=True,  # Export to ONNX
    )
    
    # Save ONNX model
    model.save_pretrained(OUTPUT_DIR)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"✅ Successfully exported to {OUTPUT_DIR}")
    print(f"   Files: {list(OUTPUT_DIR.glob('*'))}")
    
except Exception as e:
    print(f"❌ Export failed: {e}")
    import traceback
    traceback.print_exc()
