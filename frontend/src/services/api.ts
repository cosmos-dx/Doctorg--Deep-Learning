import axios from 'axios'
import { API_BASE_URL, API_ENDPOINTS } from '@/constants/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  }
})

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('doctorg-auth')
  if (token) {
    try {
      const authData = JSON.parse(token)
      if (authData.state?.token) {
        config.headers.Authorization = `Bearer ${authData.state.token}`
      }
    } catch (e) {
      console.error('Error parsing auth token:', e)
    }
  }
  return config
})

export const authAPI = {
  register: async (data: { email: string; password: string; full_name?: string }) => {
    const response = await apiClient.post(API_ENDPOINTS.AUTH.REGISTER, data)
    return response.data
  },
  
  login: async (data: { email: string; password: string }) => {
    const response = await apiClient.post(API_ENDPOINTS.AUTH.LOGIN, data)
    return response.data
  }
}

export const userAPI = {
  getProfile: async () => {
    const response = await apiClient.get(API_ENDPOINTS.USER.PROFILE)
    return response.data
  },
  
  getSessions: async (limit: number = 10) => {
    const response = await apiClient.get(API_ENDPOINTS.USER.SESSIONS, {
      params: { limit }
    })
    return response.data
  }
}

export const feedbackAPI = {
  submit: async (data: {
    session_id: string
    rating: number
    correct_diagnosis?: string
    helpful: boolean
    comments?: string
  }) => {
    const response = await apiClient.post(API_ENDPOINTS.FEEDBACK, data)
    return response.data
  }
}

export default apiClient
