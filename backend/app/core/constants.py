"""
Constants for DoctorG backend.
All static strings, error messages, and configuration values.
"""


class ErrorMessages:
    """Error messages used throughout the application."""
    INVALID_SYMPTOMS = "Please provide valid symptoms"
    UNAUTHORIZED = "Authentication required"
    INVALID_CREDENTIALS = "Invalid email or password"
    SESSION_LIMIT_REACHED = "Free tier session limit reached. Upgrade to premium for unlimited access"
    USER_NOT_FOUND = "User not found"
    DATABASE_ERROR = "Database operation failed"
    LLM_GENERATION_ERROR = "Failed to generate medical response"
    MEMORY_RETRIEVAL_ERROR = "Failed to retrieve user history"
    INVALID_FEEDBACK = "Invalid feedback data"
    TOKEN_EXPIRED = "Authentication token expired"
    INSUFFICIENT_PERMISSIONS = "Insufficient permissions for this operation"


class SuccessMessages:
    """Success messages."""
    FEEDBACK_RECEIVED = "Feedback received successfully"
    SESSION_CREATED = "Session created successfully"
    USER_REGISTERED = "User registered successfully"
    USER_LOGIN = "Login successful"


class APIEndpoints:
    """API endpoint paths."""
    HEALTH = "/health"
    PREDICT = "/api/v1/predict"
    CHAT_STREAM = "/api/v1/chat/stream"
    FEEDBACK = "/api/v1/feedback"
    AUTH_REGISTER = "/api/v1/auth/register"
    AUTH_LOGIN = "/api/v1/auth/login"
    AUTH_LOGOUT = "/api/v1/auth/logout"
    USER_PROFILE = "/api/v1/user/profile"
    USER_SESSIONS = "/api/v1/user/sessions"


class SubscriptionTiers:
    """Subscription tier identifiers."""
    FREE = "free"
    PREMIUM = "premium"


class SubscriptionLimits:
    """Subscription limits."""
    FREE_SESSION_LIMIT = 5
    FREE_MEMORY_ENABLED = False
    PREMIUM_SESSION_LIMIT = -1  # @INFO: -1 means unlimited
    PREMIUM_MEMORY_ENABLED = True


class ModelPaths:
    """Paths to trained models."""
    LLM_BASE_MODEL = "models/doctorg-medical-llm"
    LLM_LORA_ADAPTER = "models/doctorg-lora-adapter"
    SENTENCE_TRANSFORMER = "sentence-transformers/all-MiniLM-L6-v2"
    FAISS_INDEX = "models/faiss_index"


class DatabaseTables:
    """Database table names."""
    USERS = "users"
    SESSIONS = "user_sessions"
    FEEDBACK = "feedback"
    CONVERSATIONS = "conversations"


class LLMConfig:
    """LLM configuration constants."""
    MAX_LENGTH = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    NUM_RETURN_SEQUENCES = 1
    LOAD_IN_8BIT = True
    
    
class RAGConfig:
    """RAG configuration constants."""
    EMBEDDING_DIMENSION = 384
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7


class SecurityConfig:
    """Security configuration constants."""
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 60
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    PASSWORD_MIN_LENGTH = 8
    BCRYPT_ROUNDS = 12


class PubMedConfig:
    """PubMed API configuration."""
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    SEARCH_ENDPOINT = "esearch.fcgi"
    FETCH_ENDPOINT = "efetch.fcgi"
    DATABASE = "pubmed"
    RETURN_TYPE = "abstract"
    RETURN_MODE = "xml"
    MAX_RESULTS_PER_QUERY = 1000


class DatasetSources:
    """External dataset sources."""
    MEDQA = "bigbio/med_qa"
    PUBMEDQA = "pubmed_qa"
    BIOASQ = "bioasq"
