export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const API_ENDPOINTS = {
  AUTH: {
    REGISTER: '/api/v1/auth/register',
    LOGIN: '/api/v1/auth/login',
    LOGOUT: '/api/v1/auth/logout',
  },
  CHAT: {
    STREAM: '/api/v1/chat/stream',
    PREDICT: '/api/v1/chat/predict',
  },
  USER: {
    PROFILE: '/api/v1/user/profile',
    SESSIONS: '/api/v1/user/sessions',
  },
  FEEDBACK: '/api/v1/feedback',
  HEALTH: '/health',
} as const

export const STORAGE_KEYS = {
  AUTH_TOKEN: 'doctorg_auth_token',
  USER_DATA: 'doctorg_user_data',
} as const

export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network connection failed. Please try again.',
  INVALID_INPUT: 'Please check your input and try again.',
  AUTH_REQUIRED: 'Please log in to continue.',
  SESSION_LIMIT: 'Free tier session limit reached. Upgrade to premium for unlimited access.',
} as const
