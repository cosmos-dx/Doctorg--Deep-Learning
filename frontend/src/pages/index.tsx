import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useRouter } from 'next/router'
import { useAuthStore } from '@/stores/authStore'
import { Activity, Mail, Lock, User, ArrowRight, ActivitySquare } from 'lucide-react'
import toast from 'react-hot-toast'
import axios from 'axios'

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()
  const { login } = useAuthStore()

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    full_name: ''
  })

  // Redirect if already logged in
  React.useEffect(() => {
    if (useAuthStore.getState().isAuthenticated) {
      router.push('/dashboard')
    }
  }, [router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)

    try {
      const endpoint = isLogin ? '/api/v1/auth/login' : '/api/v1/auth/register'
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
      
      const payload = isLogin 
        ? { email: formData.email, password: formData.password }
        : formData

      const response = await axios.post(`${apiUrl}${endpoint}`, payload, {
        headers: {
          'Content-Type': 'application/json'
        }
      })

      const token = response.data.access_token
      
      // Fetch profile
      const profileRes = await axios.get(`${apiUrl}/api/v1/user/profile`, {
        headers: { Authorization: `Bearer ${token}` }
      })

      login(profileRes.data, token)
      toast.success(isLogin ? 'Welcome back!' : 'Account created successfully!')
      router.push('/dashboard')
    } catch (err: any) {
      toast.error(err.response?.data?.detail?.[0]?.msg || err.response?.data?.detail || 'Authentication failed')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col md:flex-row">
      <div className="flex-1 p-8 md:p-12 lg:p-24 flex flex-col justify-center relative overflow-hidden">
        <div className="absolute top-0 left-0 w-full h-full bg-slate-900 z-0">
           <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-teal-500/20 blur-[120px] rounded-full" />
           <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-purple-500/20 blur-[120px] rounded-full" />
        </div>
        
        <div className="relative z-10 max-w-md mx-auto w-full">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-3 mb-12"
          >
            <div className="w-12 h-12 bg-gradient-to-br from-teal-400 to-teal-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/20 text-white">
              <ActivitySquare size={24} />
            </div>
            <span className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-cyan-300">
              DoctorG
            </span>
          </motion.div>

          <motion.div
            key={isLogin ? 'login' : 'register'}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className="glass-panel p-8 rounded-2xl"
          >
            <h2 className="text-2xl font-bold text-white mb-2">
              {isLogin ? 'Welcome back' : 'Create an account'}
            </h2>
            <p className="text-slate-400 mb-8">
              {isLogin ? 'Enter your details to access your health profile.' : 'Join DoctorG to start your personalized health journey.'}
            </p>

            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && (
                <div>
                  <label className="block text-sm font-medium text-slate-300 mb-1">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                    <input 
                      required
                      type="text"
                      className="w-full bg-slate-900 border border-slate-700 rounded-xl py-3 pl-10 pr-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all"
                      placeholder="John Doe"
                      value={formData.full_name}
                      onChange={e => setFormData({ ...formData, full_name: e.target.value })}
                    />
                  </div>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                  <input 
                    required
                    type="email"
                    className="w-full bg-slate-900 border border-slate-700 rounded-xl py-3 pl-10 pr-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all"
                    placeholder="you@example.com"
                    value={formData.email}
                    onChange={e => setFormData({ ...formData, email: e.target.value })}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Password</label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                  <input 
                    required
                    type="password"
                    className="w-full bg-slate-900 border border-slate-700 rounded-xl py-3 pl-10 pr-4 text-white focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500 transition-all"
                    placeholder="••••••••"
                    value={formData.password}
                    onChange={e => setFormData({ ...formData, password: e.target.value })}
                  />
                </div>
              </div>

              <button 
                type="submit" 
                disabled={isLoading}
                className="w-full bg-gradient-to-r from-teal-500 to-teal-400 hover:from-teal-400 hover:to-teal-300 text-slate-900 font-bold py-3 px-4 rounded-xl transition-all flex items-center justify-center gap-2 mt-6 disabled:opacity-70"
              >
                {isLoading ? 'Processing...' : isLogin ? 'Sign In' : 'Sign Up'}
                {!isLoading && <ArrowRight size={18} />}
              </button>
            </form>

            <div className="mt-6 text-center">
              <button 
                onClick={() => setIsLogin(!isLogin)}
                className="text-slate-400 hover:text-teal-400 text-sm font-medium transition-colors"
                type="button"
              >
                {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
              </button>
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Decorative right side graphic */}
      <div className="hidden md:flex flex-1 bg-slate-900 border-l border-slate-800 items-center justify-center p-12 relative overflow-hidden">
        <div className="absolute inset-0 opacity-20" style={{ backgroundImage: 'radial-gradient(#14B8A6 1px, transparent 1px)', backgroundSize: '32px 32px' }} />
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="relative z-10 glass-panel p-8 rounded-2xl max-w-lg shadow-2xl border-teal-500/20"
        >
          <div className="flex items-start gap-4 mb-6">
            <div className="w-10 h-10 rounded-full bg-teal-500/20 flex items-center justify-center shrink-0">
              <Activity className="text-teal-400" size={20} />
            </div>
            <div>
              <h3 className="text-xl font-bold font-syne text-white mb-2">Advanced Agentic Medical AI</h3>
              <p className="text-slate-400 leading-relaxed">
                Experience clinical-grade consultation powered by multi-agent Retrieval-Augmented Generation. Upload reports, get daily advice, and track 
                biomarker trends in one streamlined health hub.
              </p>
            </div>
          </div>
          <div className="space-y-3 mt-8">
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-teal-500 to-cyan-400 w-[85%]" />
            </div>
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-purple-500 to-pink-400 w-[60%]" />
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
