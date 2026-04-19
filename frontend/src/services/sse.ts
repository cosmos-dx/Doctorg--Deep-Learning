import { API_BASE_URL, API_ENDPOINTS } from '@/constants/api'
import { AgentType } from '@/stores/chatStore'

interface StreamChunk {
  type: string
  agent?: AgentType
  content?: string
  metadata?: Record<string, any>
  session_id?: string
  symptoms?: string[]
  error_code?: string
}

interface StreamCallbacks {
  onAgentStart?: (agent: AgentType, message?: string) => void
  onChunk?: (chunk: string, agent?: AgentType) => void
  onEmergency?: (message: string, symptoms?: string[]) => void
  onOutOfScope?: (message: string) => void
  onComplete?: (sessionId?: string) => void
  onError?: (error: string, errorCode?: string) => void
}

export class SSEChatService {
  private token: string | null = null

  setToken(token: string) {
    this.token = token
  }

  async streamChat(
    message: string,
    symptoms: string[],
    sessionId: string | null,
    callbacks: StreamCallbacks
  ): Promise<void> {
    const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.CHAT.STREAM}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.token}`
      },
      body: JSON.stringify({ message, symptoms, session_id: sessionId })
    })

    if (!response.ok) {
      const text = await response.text().catch(() => '')
      callbacks.onError?.(`HTTP ${response.status}`, text)
      return
    }

    const reader = response.body?.getReader()
    if (!reader) {
      callbacks.onError?.('Stream not available')
      return
    }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const data: StreamChunk = JSON.parse(line.slice(6))
          this.dispatch(data, callbacks)
        } catch {
          // @INFO: skip malformed SSE lines
        }
      }
    }
  }

  private dispatch(data: StreamChunk, cb: StreamCallbacks) {
    switch (data.type) {
      case 'agent_start':
        if (data.agent) cb.onAgentStart?.(data.agent, data.content)
        else if (data.content) cb.onChunk?.(data.content)
        break
      case 'content':
      case 'disclaimer':
        if (data.content) cb.onChunk?.(data.content, data.agent)
        break
      case 'emergency':
        cb.onEmergency?.(data.content || '', data.symptoms)
        break
      case 'out_of_scope':
        cb.onOutOfScope?.(data.content || '')
        break
      case 'complete':
      case 'done':
        cb.onComplete?.(data.session_id)
        break
      case 'error':
        cb.onError?.(data.content || 'Server error', data.error_code)
        break
    }
  }
}

export const sseService = new SSEChatService()
