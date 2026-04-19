'use client'

import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MessageListProps {
  messages: any[]
  currentStreamContent: string
  isStreaming: boolean
}

const markdownComponents = {
  h3: (props: any) => <h3 className="text-base font-semibold mt-3 mb-1.5 text-gray-900" {...props} />,
  p: (props: any) => <p className="mb-2 last:mb-0" {...props} />,
  ul: (props: any) => <ul className="list-disc pl-4 mb-2 space-y-0.5" {...props} />,
  ol: (props: any) => <ol className="list-decimal pl-4 mb-2 space-y-0.5" {...props} />,
  li: (props: any) => <li className="leading-relaxed" {...props} />,
  strong: (props: any) => <strong className="font-semibold text-gray-900" {...props} />,
  em: (props: any) => <em className="text-gray-500 text-xs" {...props} />,
  hr: () => <hr className="my-3 border-gray-200" />,
  a: (props: any) => <a className="text-emerald-600 underline" target="_blank" rel="noopener noreferrer" {...props} />,
  pre: (props: any) => <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto mb-2" {...props} />,
  code: (props: any) => <code className="bg-gray-100 px-1 py-0.5 rounded text-xs" {...props} />,
}

function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
      {content}
    </ReactMarkdown>
  )
}

export default function MessageList({ messages, currentStreamContent, isStreaming }: MessageListProps) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, currentStreamContent])

  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">
      {messages.length === 0 && !isStreaming && (
        <div className="text-center py-20">
          <div className="inline-flex items-center justify-center w-14 h-14 bg-emerald-100 rounded-2xl mb-4">
            <span className="text-2xl">🩺</span>
          </div>
          <h2 className="text-lg font-semibold text-gray-900 mb-2">
            Welcome to DoctorG
          </h2>
          <p className="text-gray-500 text-sm max-w-md mx-auto">
            Describe your symptoms to get AI-powered medical insights.
          </p>
        </div>
      )}

      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
        >
          <div
            className={`max-w-[85%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
              message.role === 'user'
                ? 'bg-emerald-500 text-white rounded-br-md'
                : message.isEmergency
                ? 'bg-red-50 text-red-800 border border-red-200 rounded-bl-md'
                : message.role === 'system'
                ? 'bg-amber-50 text-amber-800 border border-amber-200 rounded-bl-md'
                : 'bg-white text-gray-800 shadow-sm border border-gray-100 rounded-bl-md'
            }`}
          >
            {message.role === 'user' ? (
              message.content
            ) : (
              <MarkdownContent content={message.content} />
            )}
          </div>
        </div>
      ))}

      {isStreaming && currentStreamContent && (
        <div className="flex justify-start">
          <div className="max-w-[85%] px-4 py-3 rounded-2xl rounded-bl-md bg-white text-gray-800 shadow-sm border border-gray-100 text-sm leading-relaxed">
            <MarkdownContent content={currentStreamContent} />
            <span className="inline-block w-1.5 h-4 bg-emerald-500 ml-0.5 animate-pulse rounded-sm" />
          </div>
        </div>
      )}

      {isStreaming && !currentStreamContent && (
        <div className="flex justify-start">
          <div className="px-4 py-3 rounded-2xl bg-white shadow-sm border border-gray-100">
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}

      <div ref={endRef} />
    </div>
  )
}
