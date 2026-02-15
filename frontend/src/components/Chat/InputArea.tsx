'use client'

import { useState, KeyboardEvent } from 'react'
import { PaperAirplaneIcon } from '@heroicons/react/24/solid'

interface InputAreaProps {
  onSubmit: (input: string) => void
  disabled: boolean
}

export default function InputArea({ onSubmit, disabled }: InputAreaProps) {
  const [input, setInput] = useState('')

  const handleSubmit = () => {
    if (input.trim() && !disabled) {
      onSubmit(input.trim())
      setInput('')
    }
  }

  const handleKeyPress = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-border bg-bg-secondary px-6 py-4">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end space-x-4">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Describe your symptoms..."
            disabled={disabled}
            rows={3}
            className="flex-1 bg-bg-tertiary text-text-primary rounded-lg px-4 py-3 
                     focus:outline-none focus:ring-2 focus:ring-accent resize-none
                     disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSubmit}
            disabled={disabled || !input.trim()}
            className="bg-accent text-white rounded-lg p-3 hover:bg-opacity-90 
                     disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            <PaperAirplaneIcon className="w-6 h-6" />
          </button>
        </div>
        <p className="text-xs text-text-secondary mt-2">
          Press Enter to send, Shift+Enter for new line
        </p>
      </div>
    </div>
  )
}
