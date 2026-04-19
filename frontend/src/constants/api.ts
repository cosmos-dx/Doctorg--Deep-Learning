export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export const API_ENDPOINTS = {
  AUTH: {
    REGISTER: '/api/v1/auth/register',
    LOGIN: '/api/v1/auth/login',
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
