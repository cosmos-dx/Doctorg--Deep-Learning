import { create } from 'zustand'

export interface MedicalResponse {
  possible_conditions: string[]
  confidence_level: 'low' | 'medium' | 'high'
  follow_up_questions: string[]
  risk_factors: string[]
  suggested_tests: string[]
  lifestyle_recommendations: string[]
  severity: 'mild' | 'moderate' | 'severe'
  should_see_doctor: boolean
  reasoning?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  structuredData?: MedicalResponse
  timestamp: Date
}

interface ChatStore {
  messages: Message[]
  isStreaming: boolean
  currentStreamContent: string
  sessionsRemaining: number
  
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  updateStreamContent: (content: string) => void
  setStreaming: (value: boolean) => void
  setSessionsRemaining: (count: number) => void
  clearChat: () => void
}

export const useChatStore = create<ChatStore>((set) => ({
  messages: [],
  isStreaming: false,
  currentStreamContent: '',
  sessionsRemaining: 5,
  
  addMessage: (message) => 
    set((state) => ({
      messages: [
        ...state.messages,
        {
          ...message,
          id: Math.random().toString(36).substr(2, 9),
          timestamp: new Date()
        }
      ]
    })),
  
  updateStreamContent: (content) =>
    set({ currentStreamContent: content }),
  
  setStreaming: (value) => 
    set({ isStreaming: value, currentStreamContent: value ? '' : '' }),
  
  setSessionsRemaining: (count) =>
    set({ sessionsRemaining: count }),
  
  clearChat: () => 
    set({
      messages: [],
      currentStreamContent: '',
      isStreaming: false
    })
}))
