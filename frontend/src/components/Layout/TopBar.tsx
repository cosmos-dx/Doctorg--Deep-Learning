import React from 'react'
import { useRouter } from 'next/router'
import { useAuthStore } from '@/stores/authStore'
import { Sparkles, Crown } from 'lucide-react'

export default function TopBar() {
  const router = useRouter()
  const { user } = useAuthStore()

  // Get a readable title for the current route
  const getPageTitle = () => {
    const path = router.pathname
    if (path.startsWith('/dashboard')) return 'Dashboard'
    if (path.startsWith('/chat')) return 'AI Consultation'
    if (path.startsWith('/reports')) return 'Medical Reports'
    if (path.startsWith('/profile')) return 'Health Profile'
    return 'DoctorG'
  }

  return (
    <header className="h-20 w-full glass-panel border-t-0 border-x-0 flex items-center justify-between px-8 z-30 sticky top-0 bg-slate-900/60">
      <div className="flex items-center gap-4">
        <h1 className="text-2xl font-bold text-slate-100">{getPageTitle()}</h1>
      </div>

      <div className="flex items-center gap-6">
        {user?.subscription_tier === 'premium' ? (
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-amber-500/10 to-orange-400/10 border border-amber-500/20 rounded-full">
            <Crown size={16} className="text-amber-400" />
            <span className="text-sm font-medium text-amber-400">Premium</span>
          </div>
        ) : (
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-slate-800 border border-slate-700 rounded-full">
            <Sparkles size={16} className="text-slate-400" />
            <span className="text-sm font-medium text-slate-300">Free Tier</span>
          </div>
        )}

        <div className="flex items-center gap-3 pl-6 border-l border-slate-700">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-teal-400 to-blue-500 p-0.5">
            <div className="w-full h-full bg-slate-900 rounded-full flex items-center justify-center text-sm font-bold text-slate-200">
              {user?.full_name?.charAt(0).toUpperCase() || user?.email.charAt(0).toUpperCase() || 'U'}
            </div>
          </div>
          <div className="hidden md:block">
            <p className="text-sm font-medium text-slate-200">{user?.full_name || 'Patient'}</p>
            <p className="text-xs text-slate-500 truncate max-w-[150px]">{user?.email}</p>
          </div>
        </div>
      </div>
    </header>
  )
}
