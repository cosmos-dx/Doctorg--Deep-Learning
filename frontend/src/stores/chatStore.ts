import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type AgentType = 'triage' | 'diagnostic' | 'lifestyle' | 'followup' | 'guardrails' | 'rag'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  isEmergency?: boolean
}

interface ChatStore {
  messages: Message[]
  sessionId: string | null
  isStreaming: boolean
  currentStreamContent: string
  currentAgent: AgentType | null

  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  updateStreamContent: (content: string) => void
  appendStreamContent: (content: string) => void
  setStreaming: (value: boolean) => void
  setCurrentAgent: (agent: AgentType | null) => void
  setSessionId: (id: string) => void
  resetSession: () => void
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      messages: [],
      sessionId: null,
      isStreaming: false,
      currentStreamContent: '',
      currentAgent: null,

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

      appendStreamContent: (content) =>
        set((state) => ({
          currentStreamContent: state.currentStreamContent + content
        })),

      setStreaming: (value) =>
        set({ isStreaming: value, currentStreamContent: '' }),

      setCurrentAgent: (agent) =>
        set({ currentAgent: agent }),

      setSessionId: (id) =>
        set({ sessionId: id }),

      resetSession: () =>
        set({
          sessionId: null,
          messages: [],
          currentStreamContent: '',
          isStreaming: false,
          currentAgent: null
        })
    }),
    {
      name: 'doctorg-chat',
      partialize: (state) => ({
        messages: state.messages,
        sessionId: state.sessionId
      })
    }
  )
)
