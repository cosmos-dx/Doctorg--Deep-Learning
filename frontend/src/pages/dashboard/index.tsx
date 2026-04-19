import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Activity, Clock, FileText, TrendingUp, AlertCircle, Calendar } from 'lucide-react'
import { useAuthStore } from '@/stores/authStore'
import Link from 'next/link'
import axios from 'axios'
import { format } from 'date-fns'

export default function Dashboard() {
  const { user, token } = useAuthStore()
  const [data, setData] = useState({
    summary: '',
    reportsCount: 0,
    recentConsults: [],
    loading: true
  })

  useEffect(() => {
    const fetchDashboardData = async () => {
      if (!token) return
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        
        const [profileRes, summaryRes, reportsRes] = await Promise.all([
          axios.get(`${apiUrl}/api/v1/profile/history?limit=3`, { headers: { Authorization: `Bearer ${token}` } }),
          axios.get(`${apiUrl}/api/v1/reports/summary`, { headers: { Authorization: `Bearer ${token}` } }),
          axios.get(`${apiUrl}/api/v1/reports/?limit=1`, { headers: { Authorization: `Bearer ${token}` } })
        ])

        setData({
          recentConsults: profileRes.data,
          summary: summaryRes.data?.summary || 'Upload medical reports to get your personalized health snapshot.',
          reportsCount: reportsRes.data?.length > 0 ? 1 : 0, // Simplified for UI
          loading: false
        })
      } catch (err) {
        console.error('Failed to load dashboard', err)
        setData(prev => ({ ...prev, loading: false }))
      }
    }
    fetchDashboardData()
  }, [token])

  if (data.loading) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500"></div>
      </div>
    )
  }

  return (
    <div className="space-y-8 pb-12">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-white font-syne capitalize">Good morning, {user?.full_name?.split(' ')[0] || 'Patient'}</h1>
        <p className="text-slate-400 mt-2">Here is your daily health overview.</p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Metric Cards */}
        <div className="glass-panel p-6 rounded-2xl flex items-center gap-5">
          <div className="w-14 h-14 rounded-full bg-teal-500/20 flex items-center justify-center">
            <Activity className="text-teal-400" size={24} />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Account Status</p>
            <h3 className="text-2xl font-bold text-white capitalize">{user?.subscription_tier || 'Free'}</h3>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl flex items-center gap-5">
          <div className="w-14 h-14 rounded-full bg-purple-500/20 flex items-center justify-center">
            <Clock className="text-purple-400" size={24} />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Consultations</p>
            <h3 className="text-2xl font-bold text-white">{user?.sessions_used || 0}</h3>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl flex items-center gap-5">
          <div className="w-14 h-14 rounded-full bg-blue-500/20 flex items-center justify-center">
            <TrendingUp className="text-blue-400" size={24} />
          </div>
          <div>
            <p className="text-slate-400 text-sm font-medium">Reports Processed</p>
            <h3 className="text-2xl font-bold text-white">{data.reportsCount}+</h3>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Column */}
        <div className="lg:col-span-2 space-y-8">
          <motion.div 
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} 
            className="glass-panel rounded-2xl p-8"
          >
            <div className="flex items-center gap-3 mb-6">
              <FileText className="text-teal-400" />
              <h2 className="text-xl font-bold text-white">AI Health Narrative</h2>
            </div>
            {data.summary.includes('Upload') ? (
              <div className="bg-slate-800/50 border border-slate-700 p-6 rounded-xl text-center">
                <AlertCircle className="text-slate-500 mx-auto mb-3" size={32} />
                <p className="text-slate-300 mb-4">{data.summary}</p>
                <Link href="/reports" className="text-sm font-medium text-teal-400 hover:text-teal-300">
                  Upload a report →
                </Link>
              </div>
            ) : (
              <div className="prose prose-invert max-w-none text-slate-300">
                {data.summary.split('\n').map((line, i) => (
                  <p key={i} className="mb-2">{line}</p>
                ))}
              </div>
            )}
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
            className="glass-panel rounded-2xl p-8"
          >
            <h2 className="text-xl font-bold text-white mb-6">Recent Consultations</h2>
            {data.recentConsults.length === 0 ? (
              <p className="text-slate-400 text-sm">No recent consultations.</p>
            ) : (
              <div className="space-y-4">
                {data.recentConsults.map((c: any) => (
                  <div key={c.id} className="p-4 bg-slate-800/40 rounded-xl border border-slate-700/50 hover:border-slate-600 transition-colors">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center gap-2">
                        <Calendar size={14} className="text-slate-500" />
                        <span className="text-xs text-slate-400">{format(new Date(c.timestamp), 'MMM d, yyyy - h:mm a')}</span>
                      </div>
                      <span className="px-2 py-1 bg-slate-700 rounded-md text-[10px] uppercase font-bold text-slate-300">
                        {c.diagnosis.severity || 'UNKNOWN'}
                      </span>
                    </div>
                    <div className="mb-2">
                      <p className="text-sm text-slate-300">
                        <span className="font-semibold text-white">Symptoms:</span> {c.symptoms.join(', ')}
                      </p>
                    </div>
                    <div className="bg-slate-900/50 p-3 rounded-lg text-sm text-slate-400 line-clamp-2">
                      {c.diagnosis.possible_conditions?.[0] || 'Consultation record'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>

        {/* Side Column */}
        <div className="space-y-8">
          <motion.div 
            initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}
            className="glass-panel rounded-2xl p-6 bg-gradient-to-b from-teal-900/20 to-slate-900/40"
          >
            <h3 className="font-bold text-white mb-4">Quick Actions</h3>
            <div className="space-y-3">
              <Link href="/chat" className="flex items-center justify-between p-4 bg-slate-800/50 hover:bg-teal-500/20 border border-slate-700 hover:border-teal-500/50 rounded-xl transition-all group">
                <span className="font-medium text-slate-200 group-hover:text-teal-400 transition-colors">New Consultation</span>
                <Activity size={18} className="text-slate-500 group-hover:text-teal-400" />
              </Link>
              <Link href="/reports" className="flex items-center justify-between p-4 bg-slate-800/50 hover:bg-purple-500/20 border border-slate-700 hover:border-purple-500/50 rounded-xl transition-all group">
                <span className="font-medium text-slate-200 group-hover:text-purple-400 transition-colors">Upload Lab Report</span>
                <FileText size={18} className="text-slate-500 group-hover:text-purple-400" />
              </Link>
            </div>
          </motion.div>

          <div className="glass-panel rounded-2xl p-6 border-dashed border-teal-500/30">
            <h3 className="font-bold text-teal-400 mb-2 font-syne">DoctorG Tip</h3>
            <p className="text-sm text-slate-300 leading-relaxed">
              Ask DoctorG anything about your daily wellness! Head to Consultation and ask for a personalized meal plan or sleep advice.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
