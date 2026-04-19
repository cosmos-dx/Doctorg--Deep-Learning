'use client'

import { useState, useEffect, useRef } from 'react'
import { useChatStore } from '@/stores/chatStore'
import { useAuthStore } from '@/stores/authStore'
import { sseService } from '@/services/sse'
import MessageList from './MessageList'
import InputArea from './InputArea'

export default function ChatInterface() {
  const {
    messages, addMessage,
    isStreaming, setStreaming,
    currentStreamContent, updateStreamContent, appendStreamContent,
    currentAgent, setCurrentAgent,
    sessionId, setSessionId,
    resetSession
  } = useChatStore()

  const { token, logout } = useAuthStore()
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef('')

  useEffect(() => {
    if (token) sseService.setToken(token)
  }, [token])

  useEffect(() => {
    streamRef.current = currentStreamContent
  }, [currentStreamContent])

  const handleSubmit = async (input: string) => {
    if (!input.trim() || isStreaming) return

    setError(null)
    addMessage({ role: 'user', content: input })
    setStreaming(true)
    updateStreamContent('')
    streamRef.current = ''

    try {
      await sseService.streamChat(input, [input], sessionId, {
        onAgentStart: (_agent, message) => {
          setCurrentAgent(_agent)
          if (message) appendStreamContent(message)
        },
        onChunk: (chunk) => appendStreamContent(chunk),
        onEmergency: (message) => {
          addMessage({ role: 'system', content: message, isEmergency: true })
          setStreaming(false)
          setCurrentAgent(null)
        },
        onOutOfScope: (message) => {
          addMessage({ role: 'system', content: message })
          setStreaming(false)
          setCurrentAgent(null)
        },
        onComplete: (newSessionId) => {
          const finalContent = streamRef.current
          if (finalContent) {
            addMessage({ role: 'assistant', content: finalContent })
          }
          if (newSessionId) setSessionId(newSessionId)
          setStreaming(false)
          setCurrentAgent(null)
          updateStreamContent('')
        },
        onError: (msg) => {
          if (msg.includes('401') || msg.includes('expired')) {
            logout()
            return
          }
          setError(msg)
          setStreaming(false)
          setCurrentAgent(null)
        }
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connection failed')
      setStreaming(false)
      setCurrentAgent(null)
    }
  }

  const AGENT_LABELS: Record<string, string> = {
    triage: 'Triage Assessment',
    diagnostic: 'Diagnostic Analysis',
    lifestyle: 'Lifestyle Recommendations',
    followup: 'Follow-Up Questions',
    guardrails: 'Safety Check',
    rag: 'Knowledge Retrieval'
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-emerald-500 text-white px-6 py-3 shadow-md flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold">DoctorG</h1>
          <p className="text-xs text-emerald-100">AI Medical Consultation</p>
        </div>
        <div className="flex items-center gap-3">
          {messages.length > 0 && (
            <button
              onClick={() => { resetSession(); setError(null) }}
              className="px-3 py-1.5 text-sm bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
            >
              New Chat
            </button>
          )}
          <button
            onClick={logout}
            className="px-3 py-1.5 text-sm bg-white/20 hover:bg-white/30 rounded-lg transition-colors"
          >
            Logout
          </button>
        </div>
      </header>

      {currentAgent && (
        <div className="bg-emerald-50 border-b border-emerald-100 px-6 py-1.5 flex items-center gap-2">
          <div className="animate-pulse w-2 h-2 bg-emerald-500 rounded-full" />
          <span className="text-xs text-emerald-700 font-medium">
            {AGENT_LABELS[currentAgent] || currentAgent}
          </span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto">
        <MessageList
          messages={messages}
          currentStreamContent={currentStreamContent}
          isStreaming={isStreaming}
        />
      </div>

      {error && (
        <div className="mx-4 mb-2 bg-red-50 border border-red-200 text-red-700 px-4 py-2 rounded-lg text-sm">
          {error}
        </div>
      )}

      <InputArea onSubmit={handleSubmit} disabled={isStreaming} />
    </div>
  )
}
