'use client'

import { useEffect, useRef } from 'react'
import { useChatStore } from '@/stores/chatStore'
import MedicalResponse from '../MedicalResponse'

interface MessageListProps {
  messages: any[]
  currentStreamContent: string
  isStreaming: boolean
}

export default function MessageList({ messages, currentStreamContent, isStreaming }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentStreamContent])

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      {messages.length === 0 && !isStreaming && (
        <div className="text-center py-12">
          <h2 className="text-xl font-medium text-text-primary mb-4">
            Welcome to DoctorG
          </h2>
          <p className="text-text-secondary">
            Describe your symptoms to get AI-powered medical insights
          </p>
        </div>
      )}

      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-3xl px-6 py-4 rounded-lg ${
              message.role === 'user'
                ? 'bg-accent text-white'
                : 'bg-bg-secondary text-text-primary'
            }`}
          >
            <p className="whitespace-pre-wrap">{message.content}</p>
            
            {message.structuredData && (
              <div className="mt-4">
                <MedicalResponse data={message.structuredData} />
              </div>
            )}
          </div>
        </div>
      ))}

      {isStreaming && currentStreamContent && (
        <div className="flex justify-start">
          <div className="max-w-3xl px-6 py-4 rounded-lg bg-bg-secondary text-text-primary">
            <p className="whitespace-pre-wrap">{currentStreamContent}</p>
            <div className="flex items-center mt-2">
              <div className="animate-pulse flex space-x-1">
                <div className="w-2 h-2 bg-accent rounded-full"></div>
                <div className="w-2 h-2 bg-accent rounded-full"></div>
                <div className="w-2 h-2 bg-accent rounded-full"></div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  )
}
