#!/bin/bash

echo "========================================"
echo "  DoctorG Medical LLM Training Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU detected\"}')"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: PyTorch not installed or CUDA not available"
    echo ""
    echo "Please install PyTorch with CUDA support:"
    echo "pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo "  Step 1: Preparing Training Data"
echo "========================================"
python scripts/prepare_training_data.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Data preparation failed"
    exit 1
fi

echo ""
echo "========================================"
echo "  Step 2: Starting Model Training"
echo "========================================"
echo ""
echo "This will take 2-6 hours depending on your GPU"
echo "Training: Mistral-7B with LoRA"
echo "Epochs: 3"
echo ""

python scripts/train_llm.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed"
    exit 1
fi

echo ""
echo "========================================"
echo "  Training Complete!"
echo "========================================"
echo ""
echo "Model saved to: backend/models/doctorg-medical-llm"
echo ""
echo "Next steps:"
echo "1. Test the model: python -c \"from scripts.train_llm import MedicalLLMTrainer; trainer = MedicalLLMTrainer(); trainer.test_inference('test')\""
echo "2. Start backend: uvicorn app.main:app --reload"
echo "3. Start frontend: cd ../frontend && npm run dev"
echo ""
