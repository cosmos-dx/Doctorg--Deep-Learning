import { API_BASE_URL, API_ENDPOINTS } from '@/constants/api'
import { MedicalResponse } from '@/stores/chatStore'

interface StreamChunk {
  content?: string
  done: boolean
  error?: string
  structured_data?: MedicalResponse
}

export class SSEChatService {
  private token: string | null = null

  setToken(token: string) {
    this.token = token
  }

  async streamChat(
    symptoms: string[],
    onChunk: (chunk: string) => void,
    onComplete: (data: MedicalResponse) => void,
    onError: (error: string) => void
  ): Promise<void> {
    try {
      const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.CHAT.STREAM}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.token}`
        },
        body: JSON.stringify({ symptoms })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('Stream not available')
      }

      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        const chunk = decoder.decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data: StreamChunk = JSON.parse(line.slice(6))
              
              if (data.error) {
                onError(data.error)
                return
              }
              
              if (data.content) {
                onChunk(data.content)
              }
              
              if (data.done) {
                if (data.structured_data) {
                  onComplete(data.structured_data)
                }
                return
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('SSE stream error:', error)
      onError(error instanceof Error ? error.message : 'Stream connection failed')
    }
  }
}

export const sseService = new SSEChatService()
