'use client'

import { useState, useRef, useEffect } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { useAuthStore } from '@/stores/authStore'
import { sseService } from '@/services/sse'
import MessageList from './MessageList'
import InputArea from './InputArea'
import MedicalResponse from '../MedicalResponse'

export default function ChatInterface() {
  const { messages, addMessage, isStreaming, setStreaming, currentStreamContent, updateStreamContent } = useChatStore()
  const { token } = useAuthStore()
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (token) {
      sseService.setToken(token)
    }
  }, [token])

  const handleSubmit = async (input: string) => {
    if (!input.trim() || isStreaming) return

    setError(null)
    
    addMessage({
      role: 'user',
      content: input
    })

    setStreaming(true)
    let streamedContent = ''

    try {
      await sseService.streamChat(
        [input],
        (chunk) => {
          streamedContent += chunk
          updateStreamContent(streamedContent)
        },
        (structuredData) => {
          addMessage({
            role: 'assistant',
            content: streamedContent,
            structuredData
          })
          setStreaming(false)
        },
        (errorMsg) => {
          setError(errorMsg)
          setStreaming(false)
        }
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed')
      setStreaming(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-bg-primary">
      <header className="bg-bg-secondary border-b border-border px-6 py-4">
        <h1 className="text-2xl font-semibold text-text-primary">DoctorG Medical AI</h1>
        <p className="text-sm text-text-secondary mt-1">AI-powered medical consultation assistant</p>
      </header>

      <div className="flex-1 overflow-y-auto">
        <MessageList 
          messages={messages} 
          currentStreamContent={currentStreamContent}
          isStreaming={isStreaming}
        />
      </div>

      {error && (
        <div className="bg-red-900 border border-red-700 text-red-200 px-4 py-3 mx-6">
          <p className="text-sm">{error}</p>
        </div>
      )}

      <InputArea onSubmit={handleSubmit} disabled={isStreaming} />
    </div>
  )
}
