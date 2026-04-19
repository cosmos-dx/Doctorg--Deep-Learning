import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/stores/authStore'
import { Send, Bot, User, ShieldAlert, Activity, Coffee } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import toast from 'react-hot-toast'

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
  agentType?: string
}

export default function ChatPage() {
  const { token } = useAuthStore()
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'init',
      role: 'assistant',
      content: "Hello! I am DoctorG, your AI health assistant. You can ask me about symptoms you're experiencing, or get daily lifestyle, sleep, and nutrition advice.",
      agentType: 'orchestrator'
    }
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!input.trim() || isLoading) return

    const userMsg: Message = { id: Date.now().toString(), role: 'user', content: input }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setIsLoading(true)

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    let fullResponse = ''
    let currentAgent = 'orchestrator'

    const astMsgId = (Date.now() + 1).toString()
    setMessages(prev => [...prev, { id: astMsgId, role: 'assistant', content: '', isStreaming: true, agentType: currentAgent }])

    try {
      const response = await fetch(`${apiUrl}/api/v1/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ symptoms: [], message: userMsg.content })
      })

      if (!response.ok) throw new Error('API Error')

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()

      if (reader) {
        let buffer = ''
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n\n')
          buffer = lines.pop() || ''

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.slice(6)
              if (dataStr === '[DONE]') continue
              try {
                const data = JSON.parse(dataStr)
                if (data.type === 'agent_start' || (data.type === 'content' && data.agent)) {
                  if (data.agent && data.agent !== currentAgent) {
                     currentAgent = data.agent
                  }
                }
                
                if (data.content !== undefined) {
                  fullResponse += data.content
                  setMessages(prev => prev.map(m => 
                    m.id === astMsgId ? { ...m, content: fullResponse, agentType: currentAgent } : m
                  ))
                }
                if (data.error) throw new Error(data.error)
              } catch (e) {
                console.error("Stream parse error", e)
              }
            }
          }
        }
      }
    } catch (error: any) {
      toast.error(error.message || 'Failed to get response')
      setMessages(prev => prev.map(m => m.id === astMsgId ? { ...m, content: m.content || 'An error occurred. Please try again.' } : m))
    } finally {
      setIsLoading(false)
      setMessages(prev => prev.map(m => m.id === astMsgId ? { ...m, isStreaming: false } : m))
    }
  }

  const getAgentBadge = (agentType?: string) => {
    switch (agentType) {
      case 'daily_advisor':
        return <span className="bg-orange-500/20 text-orange-400 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full flex items-center gap-1"><Coffee size={10}/> Daily Advisor</span>
      case 'triage':
      case 'guardrails':
        return <span className="bg-red-500/20 text-red-400 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full flex items-center gap-1"><ShieldAlert size={10}/> Guardrails</span>
      case 'diagnostic':
        return <span className="bg-teal-500/20 text-teal-400 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full flex items-center gap-1"><Activity size={10}/> Medical AI</span>
      default:
        return null
    }
  }

  return (
    <div className="h-[calc(100vh-6rem)] flex flex-col glass-panel rounded-2xl overflow-hidden">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6">
        {messages.map((msg) => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex gap-4 max-w-4xl mx-auto ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 shadow-lg ${
              msg.role === 'user' 
                ? 'bg-gradient-to-br from-blue-500 to-indigo-600 outline outline-2 outline-offset-2 outline-blue-500/30' 
                : 'bg-gradient-to-br from-teal-400 to-emerald-600 outline outline-2 outline-offset-2 outline-teal-500/30'
            }`}>
              {msg.role === 'user' ? <User size={20} className="text-white" /> : <Bot size={20} className="text-white" />}
            </div>

            <div className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'} max-w-[80%]`}>
              {msg.role === 'assistant' && msg.agentType && (
                <div className="mb-2 ml-1">
                  {getAgentBadge(msg.agentType)}
                </div>
              )}
              
              <div className={`p-4 rounded-2xl ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-tr-sm'
                  : 'bg-slate-800/80 border border-slate-700/50 text-slate-200 rounded-tl-sm'
              }`}>
                {msg.role === 'assistant' ? (
                  <div className="prose prose-invert prose-teal max-w-none prose-p:leading-relaxed prose-pre:bg-slate-900 prose-pre:border prose-pre:border-slate-800">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content + (msg.isStreaming ? ' ▍' : '')}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <p>{msg.content}</p>
                )}
              </div>
            </div>
          </motion.div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-4 md:p-6 bg-slate-900/80 border-t border-slate-800 backdrop-blur-md">
        <form onSubmit={handleSend} className="max-w-4xl mx-auto relative flex items-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe your symptoms or ask for daily wellness advice..."
            className="w-full bg-slate-800/50 border border-slate-700 rounded-full py-4 pl-6 pr-16 text-slate-200 focus:outline-none focus:border-teal-500 focus:ring-1 focus:ring-teal-500 shadow-inner"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 bg-teal-500 hover:bg-teal-400 text-slate-900 rounded-full flex items-center justify-center transition-colors disabled:opacity-50 disabled:hover:bg-teal-500 shadow-lg shadow-teal-500/20"
          >
            <Send size={18} className={input.trim() ? "translate-x-[-1px] translate-y-[1px]" : ""} />
          </button>
        </form>
        <p className="text-center text-xs text-slate-500 mt-3 flex items-center justify-center gap-1.5 font-medium">
          <ShieldAlert size={12} className="text-red-400/70" />
          DoctorG provides informational advice only. Always consult a real doctor for medical emergencies.
        </p>
      </div>
    </div>
  )
}
