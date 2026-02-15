# DoctorG 2.0 - AI Medical Reasoning Assistant

Transform from a static prediction tool into an intelligent Medical AI Assistant with fine-tuned LLM, RAG memory, and real-time streaming.

## 🎯 Phase 1 Features

- **Fine-tuned Medical LLM** (Mistral-7B + LoRA)
- **RAG Memory Engine** (FAISS + PostgreSQL)
- **Real-time SSE Streaming**
- **Subscription Logic** (Free/Premium tiers)
- **Modern Dark UI** (ChatGPT-style)
- **Feedback Learning System**
- **Production-Ready** (Docker + GPU support)

## 🏗️ Architecture

```
User → Next.js Frontend (React + Zustand)
  ↓
FastAPI Backend (Python + Async)
  ↓
Medical LLM (Mistral-7B-LoRA) + RAG (FAISS)
  ↓
PostgreSQL + Redis
```

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Docker & Docker Compose**
- **NVIDIA GPU** (for training, optional for inference)
- **CUDA 11.8+** (for GPU training)
- **16GB+ RAM** (32GB recommended)
- **50GB+ Disk Space**

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd doctorg

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

### 2. Configure Environment Variables

Edit `.env` file:

```bash
# Required - Add your OpenAI API key
OPENAI_API_KEY=sk-proj-your_key_here

# Database (auto-configured in Docker)
POSTGRES_PASSWORD=your_secure_password_here
JWT_SECRET=your_jwt_secret_min_32_chars

# Optional - for dataset augmentation
GOOGLE_API_KEY=your_google_key_here
PUBMED_EMAIL=your_email@example.com
```

### 3. Run with Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### 4. Manual Setup (Development)

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run database migrations
python -c "from app.db.database import init_db; init_db()"

# Start backend server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Access at http://localhost:3000
```

## 🎓 Training the Medical LLM

### Step 1: Prepare Training Data

```bash
cd backend

# Activate virtual environment
source venv/bin/activate

# Run data preparation (converts CSV to instruction format)
python scripts/prepare_training_data.py
```

This creates:
- `backend/data/training/train.jsonl` - Training data
- `backend/data/training/val.jsonl` - Validation data

### Step 2: (Optional) Augment with External Data

```bash
# Fetch PubMed abstracts and Clinical QA datasets
python scripts/web_agent.py

# This downloads:
# - PubMed medical abstracts (1000+)
# - MedQA clinical questions
# - PubMedQA dataset
```

### Step 3: Fine-tune with GPU

**Requirements:**
- NVIDIA GPU with 16GB+ VRAM (RTX 3090, A100, etc.)
- CUDA 11.8+ installed
- PyTorch with CUDA support

```bash
# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Start fine-tuning (takes 2-6 hours depending on GPU)
python scripts/train_llm.py
```

**Training Configuration:**
- Base Model: Mistral-7B-v0.1
- Method: LoRA (Low-Rank Adaptation)
- Epochs: 3
- Batch Size: 4 (adjust based on VRAM)
- Learning Rate: 2e-4
- Quantization: 8-bit (reduces VRAM usage)

**Expected Output:**
```
Loading model: mistralai/Mistral-7B-v0.1
Model loaded successfully
LoRA configuration created
trainable params: 4,194,304 || all params: 7,241,732,096 || trainable%: 0.0579
Starting training...
Epoch 1/3: 100%|██████████| 500/500 [1:23:45<00:00]
Saving model to backend/models/doctorg-medical-llm
Training completed successfully!
```

### Step 4: Test the Model

```bash
# Test inference
python -c "
from backend.scripts.train_llm import MedicalLLMTrainer

trainer = MedicalLLMTrainer()
prompt = '''You are a medical AI assistant. Analyze the symptoms and provide a structured medical assessment.

Symptoms: headache, fever, fatigue

Provide your response in JSON format:'''

response = trainer.test_inference(prompt)
print(response)
"
```

### Training on Cloud GPU (Alternative)

If you don't have a local GPU:

**Google Colab (Free GPU):**
```python
# Upload your code to Google Drive
# Open Google Colab notebook
# Mount Drive and run:

!pip install -r requirements.txt
!python scripts/prepare_training_data.py
!python scripts/train_llm.py
```

**AWS/GCP/Azure:**
- Launch GPU instance (g4dn.xlarge on AWS, n1-standard-4 with T4 on GCP)
- Clone repository
- Run training scripts
- Download trained model

## 🐳 Docker Deployment

### Production Deployment

```bash
# Build for production
docker-compose -f docker-compose.yml up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### GPU Support in Docker

Edit `docker-compose.yml` to enable GPU:

```yaml
services:
  backend:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Then run:
```bash
# Requires nvidia-docker2 installed
docker-compose up --build
```

### Environment-Specific Configs

```bash
# Development
docker-compose -f docker-compose.yml up

# Production with GPU
docker-compose -f docker-compose.prod.yml up

# Staging
docker-compose -f docker-compose.staging.yml up
```

## 📊 Using the Application

### 1. Register an Account

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "full_name": "John Doe"
  }'
```

### 2. Login

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 3. Get Medical Consultation

**Via Web UI:**
1. Open http://localhost:3000
2. Login with your credentials
3. Describe your symptoms
4. Get real-time streaming response

**Via API:**
```bash
curl -X POST http://localhost:8000/api/v1/chat/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "symptoms": ["headache", "fever", "fatigue"]
  }'
```

### 4. Submit Feedback

```bash
curl -X POST http://localhost:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "session_id": "session-id-here",
    "rating": 5,
    "helpful": true,
    "comments": "Very helpful diagnosis!"
  }'
```

## 🔧 Configuration

### Subscription Tiers

**Free Tier:**
- 5 sessions per month
- No memory/history
- Basic medical insights

**Premium Tier:**
- Unlimited sessions
- Full RAG memory (past consultations)
- Detailed follow-up questions
- Priority support

### Adjusting Limits

Edit `backend/app/core/constants.py`:

```python
class SubscriptionLimits:
    FREE_SESSION_LIMIT = 5  # Change to desired limit
    PREMIUM_SESSION_LIMIT = -1  # -1 = unlimited
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

### Frontend Tests

```bash
cd frontend
npm test
```

### End-to-End Test

```bash
# Start all services
docker-compose up -d

# Run E2E tests
npm run test:e2e
```

## 📈 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-02-15T10:30:00",
  "services": {
    "database": "connected",
    "llm": "ready",
    "rag": "ready"
  }
}
```

### Logs

```bash
# Backend logs
docker-compose logs -f backend

# Frontend logs
docker-compose logs -f frontend

# Database logs
docker-compose logs -f postgres
```

## 🔒 Security

- ✅ No hardcoded secrets (all in .env)
- ✅ Bcrypt password hashing
- ✅ JWT authentication with expiration
- ✅ SQL injection prevention (ORM)
- ✅ XSS protection (React escaping)
- ✅ CORS configured
- ✅ Security headers enabled
- ✅ Rate limiting implemented

### Security Best Practices

1. **Change default passwords** in `.env`
2. **Use strong JWT secret** (min 32 characters)
3. **Enable HTTPS** in production
4. **Regular dependency updates**: `pip list --outdated`
5. **Backup database** regularly

## 🐛 Troubleshooting

### GPU Not Detected

```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Docker Issues

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up

# Check container logs
docker-compose logs backend
```

### Database Connection Error

```bash
# Reset database
docker-compose down -v
docker-compose up postgres -d
sleep 10
docker-compose up backend
```

### Port Already in Use

```bash
# Find and kill process on port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

## 📚 API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

- **Developer**: Abhishek Gupta
- **GitHub**: [@cosmos-dx](https://github.com/cosmos-dx)
- **LinkedIn**: [abhishek-gupta](https://www.linkedin.com/in/abhishek-gupta-a1a44a203/)

## 🙏 Acknowledgments

- Mistral AI for the base model
- Hugging Face for transformers library
- OpenAI for API integration
- FastAPI and Next.js communities

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/cosmos-dx/doctorg/issues)
- **Email**: support@doctorg.ai
- **Discord**: [Join our community](https://discord.gg/doctorg)

---

**⚠️ Medical Disclaimer**: DoctorG is an AI assistant for educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions regarding medical conditions.
