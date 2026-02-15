'use client'

import { useState } from 'react'
import { feedbackAPI } from '@/services/api'
import { StarIcon } from '@heroicons/react/24/solid'
import { StarIcon as StarOutline } from '@heroicons/react/24/outline'

interface FeedbackFormProps {
  sessionId: string
  onSuccess?: () => void
}

export default function FeedbackForm({ sessionId, onSuccess }: FeedbackFormProps) {
  const [rating, setRating] = useState(0)
  const [hoveredRating, setHoveredRating] = useState(0)
  const [comments, setComments] = useState('')
  const [correctDiagnosis, setCorrectDiagnosis] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (rating === 0) return

    setSubmitting(true)

    try {
      await feedbackAPI.submit({
        session_id: sessionId,
        rating,
        correct_diagnosis: correctDiagnosis || undefined,
        helpful: rating >= 3,
        comments: comments || undefined
      })

      setSuccess(true)
      onSuccess?.()

      setTimeout(() => {
        setSuccess(false)
        setRating(0)
        setComments('')
        setCorrectDiagnosis('')
      }, 2000)
    } catch (error) {
      console.error('Error submitting feedback:', error)
    } finally {
      setSubmitting(false)
    }
  }

  if (success) {
    return (
      <div className="bg-green-900 bg-opacity-30 border border-green-700 rounded-lg p-6 text-center">
        <p className="text-green-200">Thank you for your feedback!</p>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="bg-bg-secondary rounded-lg p-6 space-y-4">
      <h3 className="text-lg font-semibold text-text-primary">Rate this consultation</h3>

      <div className="flex items-center space-x-1">
        {[1, 2, 3, 4, 5].map((star) => {
          const Icon = star <= (hoveredRating || rating) ? StarIcon : StarOutline
          return (
            <button
              key={star}
              type="button"
              onClick={() => setRating(star)}
              onMouseEnter={() => setHoveredRating(star)}
              onMouseLeave={() => setHoveredRating(0)}
              className="text-yellow-400 hover:text-yellow-300 transition-colors"
            >
              <Icon className="w-8 h-8" />
            </button>
          )
        })}
      </div>

      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          If you received a diagnosis, please share:
        </label>
        <input
          type="text"
          value={correctDiagnosis}
          onChange={(e) => setCorrectDiagnosis(e.target.value)}
          placeholder="Actual diagnosis (optional)"
          className="w-full bg-bg-tertiary text-text-primary rounded-lg px-4 py-2 
                   focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-text-primary mb-2">
          Additional comments:
        </label>
        <textarea
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          placeholder="Was this consultation helpful? Any suggestions?"
          rows={3}
          className="w-full bg-bg-tertiary text-text-primary rounded-lg px-4 py-2 
                   focus:outline-none focus:ring-2 focus:ring-accent resize-none"
        />
      </div>

      <button
        type="submit"
        disabled={rating === 0 || submitting}
        className="w-full bg-accent text-white rounded-lg px-6 py-3 font-medium
                 hover:bg-opacity-90 disabled:opacity-50 disabled:cursor-not-allowed
                 transition-all"
      >
        {submitting ? 'Submitting...' : 'Submit Feedback'}
      </button>
    </form>
  )
}
