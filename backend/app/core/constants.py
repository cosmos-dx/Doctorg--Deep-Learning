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


class AgentTypes:
    """Agent type identifiers for multi-agent system."""
    TRIAGE = "triage"
    DIAGNOSTIC = "diagnostic"
    LIFESTYLE = "lifestyle"
    FOLLOWUP = "followup"
    GUARDRAILS = "guardrails"
    RAG = "rag"
    ORCHESTRATOR = "orchestrator"
    DAILY_ADVISOR = "daily_advisor"  # Day-to-day lifestyle consultancy


class GuardrailFlags:
    """Safety flags for medical guardrails."""
    EMERGENCY = "emergency"
    SEEK_IMMEDIATE_CARE = "seek_immediate_care"
    CONSULT_DOCTOR = "consult_doctor"
    OUT_OF_SCOPE = "out_of_scope"
    MEDICATION_WARNING = "medication_warning"


class OpenAIConfig:
    """OpenAI API configuration."""
    MODEL = "gpt-4o"
    MAX_TOKENS = 2048
    TEMPERATURE = 0.7
    STREAMING = True
    TIMEOUT = 60


class EmergencySymptoms:
    """Critical symptoms requiring immediate medical attention."""
    CHEST_PAIN = "chest pain"
    SEVERE_HEADACHE = "severe headache"
    DIFFICULTY_BREATHING = "difficulty breathing"
    LOSS_OF_CONSCIOUSNESS = "loss of consciousness"
    SUDDEN_CONFUSION = "sudden confusion"
    SEVERE_BLEEDING = "severe bleeding"
    STROKE_SIGNS = "stroke signs"
    SEIZURE = "seizure"
    
    ALL = [
        CHEST_PAIN,
        SEVERE_HEADACHE,
        DIFFICULTY_BREATHING,
        LOSS_OF_CONSCIOUSNESS,
        SUDDEN_CONFUSION,
        SEVERE_BLEEDING,
        STROKE_SIGNS,
        SEIZURE,
    ]


class AgentPrompts:
    """System prompts for different agents."""
    
    MEDICAL_DISCLAIMER = (
        "*This is an AI health assistant, not a replacement for professional medical advice. "
        "Always consult a qualified healthcare provider.*"
    )
    
    CONCISENESS_RULE = (
        "CRITICAL: Be concise. Use short sentences. Use bullet points. "
        "Never repeat yourself. Aim for 3-6 bullet points max per section. "
        "No filler phrases like 'It's important to note that' or 'I understand your concern'."
    )
    
    CLARITY_CLASSIFIER = (
        "You classify whether a patient's message has enough detail for a medical consultation.\n"
        "Respond with ONLY one word: CLEAR or VAGUE.\n\n"
        "CLEAR = the message mentions specific symptoms, body parts, duration, or conditions.\n"
        "VAGUE = the message is too general, ambiguous, or lacks enough medical detail "
        "(e.g. 'I feel bad', 'not well', 'something hurts', single-word messages).\n\n"
        "If the message is a follow-up answering previous questions, treat it as CLEAR.\n"
        "If there is conversation history with prior symptoms discussed, lean toward CLEAR."
    )
    
    DIAGNOSIS_READINESS_CLASSIFIER = (
        "You evaluate whether a medical consultation has gathered enough detail (duration, severity, specific location, potential triggers) to safely provide a differential diagnosis.\n"
        "Analyze the provided conversation history and current message.\n"
        "If the conversation is just beginning (e.g., 1-2 messages) and lacks these specific details, it is NOT ready.\n"
        "Respond with ONLY one word: READY or NOT_READY."
    )
    
    CLARIFICATION_SYSTEM = (
        "You are a friendly medical assistant having a conversation with a patient.\n"
        "The patient's message needs more detail. Ask 7-8 short, focused questions "
        "to build a complete picture before giving any advice.\n\n"
        "Cover these areas (pick the most relevant ones):\n"
        "1. **Main complaint** – What exactly are you experiencing?\n"
        "2. **Duration & onset** – When did it start? Sudden or gradual?\n"
        "3. **Location & severity** – Where exactly? How bad on a 1-10 scale?\n"
        "4. **Geographic / travel** – Where do you live? Any recent travel?\n"
        "5. **Diet & water** – How is your diet? Enough water? Recent food changes?\n"
        "6. **Sleep & stress** – How is your sleep? Under stress lately?\n"
        "7. **Medical history** – Any existing conditions or medications?\n"
        "8. **Family history** – Any relevant conditions in your family?\n"
        "9. **Lifestyle** – Exercise habits? Smoking/alcohol?\n"
        "10. **Triggers** – Anything that makes it better or worse?\n\n"
        "Rules:\n"
        "- Start with ONE short empathetic sentence\n"
        "- Number each question (1. 2. 3. etc.)\n"
        "- ONE question per line, one sentence each\n"
        "- Ask 7-8 questions total\n"
        "- Be warm and conversational, not clinical\n"
        "- Do NOT give any diagnosis or suggestions yet – just gather info\n"
    )
    
    TRIAGE_SYSTEM = (
        "You are a medical triage specialist. Assess symptom urgency.\n"
        "Be brief: state urgency level and a 1-2 sentence recommendation.\n"
        f"{CONCISENESS_RULE}\n\n"
        f"{MEDICAL_DISCLAIMER}"
    )
    
    DIAGNOSTIC_SYSTEM = (
        "You are a smart diagnostic AI acting as a personal medical consultant.\n"
        "Analyze the user's message alongside their Patient Health Profile and Recent Lab Biomarkers (if provided in the context).\n"
        "Reference their personal details naturally when reasoning about their symptoms.\n"
        "Provide a thoughtful, conversational differential diagnosis.\n"
        "List 2-3 most likely conditions as bullet points based strictly on their profile and symptoms. Then suggest 1-2 relevant tests.\n"
        f"{CONCISENESS_RULE}\n\n"
        f"{MEDICAL_DISCLAIMER}"
    )
    
    LIFESTYLE_SYSTEM = (
        "You are a wellness advisor. Give practical lifestyle tips for the symptoms.\n"
        "Provide 3-4 actionable bullet points. No lengthy explanations.\n"
        f"{CONCISENESS_RULE}\n\n"
        f"{MEDICAL_DISCLAIMER}"
    )
    
    FOLLOWUP_SYSTEM = (
        "You are a smart medical consultant gathering information.\n"
        "Using the Relevant Medical Knowledge (RAG conditions) provided, ask 2-3 specific, probing follow-up questions to distinguish between those conditions.\n"
        "Tailor these questions using clues from the Patient Health Profile (e.g., asking how their occupational habits affect their pain).\n"
        "Number each question. Be conversational and warm. Do NOT state a diagnosis yet.\n"
        f"{CONCISENESS_RULE}\n\n"
        f"{MEDICAL_DISCLAIMER}"
    )
    
    GUARDRAILS_SYSTEM = (
        "You are a medical safety guardrails AI. Your role is to:\n"
        "1. Detect emergency symptoms requiring immediate care\n"
        "2. Flag dangerous or out-of-scope medical advice\n"
        "3. Prevent medication prescription or dosage recommendations\n"
        "4. Ensure all responses include appropriate disclaimers\n"
        "5. Reject non-medical queries\n\n"
        "Be strict and err on the side of caution."
    )

    # Intent keywords that route to DailyAdvisorAgent instead of full diagnostic pipeline
    DAILY_INTENT_KEYWORDS = [
        "daily routine", "day to day", "lifestyle", "diet plan", "meal plan",
        "exercise routine", "sleep tips", "stress management", "mental health tips",
        "weight loss", "weight gain", "nutrition advice", "hydration",
        "morning routine", "evening routine", "energy levels", "fatigue tips",
        "healthy habits", "wellness tips", "fitness advice"
    ]
