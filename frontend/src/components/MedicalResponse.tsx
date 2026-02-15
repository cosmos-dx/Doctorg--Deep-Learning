'use client'

import { MedicalResponse as MedicalResponseType } from '@/stores/chatStore'
import { ExclamationTriangleIcon, CheckCircleIcon } from '@heroicons/react/24/outline'

interface MedicalResponseProps {
  data: MedicalResponseType
}

export default function MedicalResponse({ data }: MedicalResponseProps) {
  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high':
        return 'text-green-400'
      case 'medium':
        return 'text-yellow-400'
      case 'low':
        return 'text-red-400'
      default:
        return 'text-text-secondary'
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'severe':
        return 'text-red-400'
      case 'moderate':
        return 'text-yellow-400'
      case 'mild':
        return 'text-green-400'
      default:
        return 'text-text-secondary'
    }
  }

  return (
    <div className="space-y-4 bg-bg-tertiary rounded-lg p-4 mt-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-text-primary">Medical Assessment</h3>
        <div className="flex items-center space-x-4">
          <span className={`text-sm font-medium ${getConfidenceColor(data.confidence_level)}`}>
            Confidence: {data.confidence_level}
          </span>
          <span className={`text-sm font-medium ${getSeverityColor(data.severity)}`}>
            Severity: {data.severity}
          </span>
        </div>
      </div>

      {data.should_see_doctor && (
        <div className="bg-yellow-900 bg-opacity-30 border border-yellow-700 rounded-lg p-3 flex items-start space-x-2">
          <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-yellow-200">
            Please consult a healthcare professional for proper diagnosis and treatment
          </p>
        </div>
      )}

      <Section title="Possible Conditions" items={data.possible_conditions} />
      
      <Section title="Follow-up Questions" items={data.follow_up_questions} />
      
      <Section title="Risk Factors" items={data.risk_factors} />
      
      <Section title="Suggested Tests" items={data.suggested_tests} />
      
      <Section title="Lifestyle Recommendations" items={data.lifestyle_recommendations} icon={<CheckCircleIcon className="w-5 h-5" />} />

      {data.reasoning && (
        <div className="pt-4 border-t border-border">
          <h4 className="text-sm font-medium text-text-primary mb-2">Reasoning</h4>
          <p className="text-sm text-text-secondary">{data.reasoning}</p>
        </div>
      )}
    </div>
  )
}

function Section({ title, items, icon }: { title: string; items: string[]; icon?: React.ReactNode }) {
  if (!items || items.length === 0) return null

  return (
    <div>
      <h4 className="text-sm font-medium text-text-primary mb-2 flex items-center space-x-2">
        {icon}
        <span>{title}</span>
      </h4>
      <ul className="list-disc list-inside space-y-1">
        {items.map((item, index) => (
          <li key={index} className="text-sm text-text-secondary">
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}
