import React, { useState, useEffect } from 'react'
import { useAuthStore } from '@/stores/authStore'
import { motion } from 'framer-motion'
import axios from 'axios'
import toast from 'react-hot-toast'
import { User, Activity, AlertCircle, HeartPulse, Save } from 'lucide-react'

export default function ProfilePage() {
  const { user, token } = useAuthStore()
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [profile, setProfile] = useState<any>({
    age: '', gender: '', blood_group: '', height_cm: '', weight_kg: '',
    allergies: '', chronic_conditions: '', current_medications: '',
    family_history: '', lifestyle_notes: ''
  })

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  useEffect(() => {
    const fetchProfile = async () => {
      if (!token) return
      try {
        const res = await axios.get(`${apiUrl}/api/v1/profile/health`, {
          headers: { Authorization: `Bearer ${token}` }
        })
        const data = res.data
        // Convert arrays to comma strings for simple text area editing
        setProfile({
          ...data,
          age: data.age || '',
          height_cm: data.height_cm || '',
          weight_kg: data.weight_kg || '',
          allergies: data.allergies?.join(', ') || '',
          chronic_conditions: data.chronic_conditions?.join(', ') || '',
          current_medications: data.current_medications?.join(', ') || '',
          family_history: data.family_history?.join(', ') || '',
        })
      } catch (err) {
        toast.error('Failed to load profile')
      } finally {
        setLoading(false)
      }
    }
    fetchProfile()
  }, [apiUrl, token])

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault()
    setSaving(true)

    const payload = {
      ...profile,
      age: profile.age ? parseInt(profile.age) : null,
      height_cm: profile.height_cm ? parseFloat(profile.height_cm) : null,
      weight_kg: profile.weight_kg ? parseFloat(profile.weight_kg) : null,
      // Convert comma strings back to arrays
      allergies: profile.allergies ? profile.allergies.split(',').map((s: string) => s.trim()).filter(Boolean) : [],
      chronic_conditions: profile.chronic_conditions ? profile.chronic_conditions.split(',').map((s: string) => s.trim()).filter(Boolean) : [],
      current_medications: profile.current_medications ? profile.current_medications.split(',').map((s: string) => s.trim()).filter(Boolean) : [],
      family_history: profile.family_history ? profile.family_history.split(',').map((s: string) => s.trim()).filter(Boolean) : [],
    }

    try {
      const res = await axios.put(`${apiUrl}/api/v1/profile/health`, payload, {
        headers: { Authorization: `Bearer ${token}` }
      })
      // Update with computed BMI
      setProfile(prev => ({ ...prev, bmi: res.data.bmi }))
      toast.success('Health profile updated successfully! DoctorG will use this in future consultations.')
    } catch (err) {
      toast.error('Failed to update profile')
    } finally {
      setSaving(false)
    }
  }

  if (loading) return null

  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <header className="mb-8 text-center sm:text-left">
        <h1 className="text-3xl font-bold text-white font-syne">Health Profile</h1>
        <p className="text-slate-400 mt-2">The AI uses this data to provide highly personalized guidance.</p>
      </header>

      {profile.bmi && (
        <motion.div initial={{opacity:0, y:-10}} animate={{opacity:1, y:0}} className="glass-panel p-6 rounded-2xl flex items-center justify-between border border-teal-500/30 bg-teal-500/5">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-teal-500/20 rounded-full flex items-center justify-center"><Activity className="text-teal-400"/></div>
            <div>
              <p className="text-slate-400 text-sm font-medium mb-1">Body Mass Index (BMI)</p>
              <p className="text-2xl font-bold text-white">{profile.bmi}</p>
            </div>
          </div>
          <div className="text-right hidden sm:block">
            <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase ${
              profile.bmi < 18.5 ? 'bg-blue-500/20 text-blue-400' :
              profile.bmi < 25 ? 'bg-emerald-500/20 text-emerald-400' :
              profile.bmi < 30 ? 'bg-orange-500/20 text-orange-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {profile.bmi < 18.5 ? 'Underweight' : profile.bmi < 25 ? 'Healthy' : profile.bmi < 30 ? 'Overweight' : 'Obese'}
            </span>
          </div>
        </motion.div>
      )}

      <form onSubmit={handleSave} className="space-y-8">
        <div className="glass-panel p-8 rounded-2xl space-y-6">
          <div className="flex items-center gap-2 border-b border-slate-700/50 pb-4 mb-6">
            <User className="text-teal-400" size={20} />
            <h2 className="text-xl font-bold text-white">Basic Vitals</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Age</label>
              <input type="number" value={profile.age} onChange={e=>setProfile({...profile, age: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" placeholder="e.g. 30" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-400 mb-2">Gender</label>
              <select value={profile.gender || ''} onChange={e=>setProfile({...profile, gender: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors">
                <option value="">Select...</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2">Weight (kg)</label>
               <input type="number" step="0.1" value={profile.weight_kg} onChange={e=>setProfile({...profile, weight_kg: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" placeholder="e.g. 70.5" />
            </div>
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2">Height (cm)</label>
               <input type="number" step="0.1" value={profile.height_cm} onChange={e=>setProfile({...profile, height_cm: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" placeholder="e.g. 175" />
            </div>
          </div>
        </div>

        <div className="glass-panel p-8 rounded-2xl space-y-6">
          <div className="flex items-center gap-2 border-b border-slate-700/50 pb-4 mb-6">
            <HeartPulse className="text-purple-400" size={20} />
            <h2 className="text-xl font-bold text-white">Medical History</h2>
          </div>
          
          <div className="space-y-6">
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2 flex items-center gap-2"><AlertCircle size={14}/> Allergies <span className="text-xs text-slate-500 font-normal">(comma separated)</span></label>
               <input type="text" value={profile.allergies} onChange={e=>setProfile({...profile, allergies: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors" placeholder="e.g. Penicillin, Peanuts" />
            </div>
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2 mt-4">Chronic Conditions & Past Surgeries</label>
               <textarea rows={2} value={profile.chronic_conditions} onChange={e=>setProfile({...profile, chronic_conditions: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors resize-none" placeholder="e.g. Type 2 Diabetes, Hypertension" />
            </div>
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2 mt-4">Current Medications</label>
               <textarea rows={2} value={profile.current_medications} onChange={e=>setProfile({...profile, current_medications: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors resize-none" placeholder="e.g. Metformin 500mg daily" />
            </div>
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2 mt-4">Family Medical History</label>
               <textarea rows={2} value={profile.family_history} onChange={e=>setProfile({...profile, family_history: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors resize-none" placeholder="e.g. Mother had breast cancer" />
            </div>
            <div>
               <label className="block text-sm font-medium text-slate-400 mb-2 mt-4">Lifestyle Notes (Diet, Sleep, Exercise)</label>
               <textarea rows={3} value={profile.lifestyle_notes || ''} onChange={e=>setProfile({...profile, lifestyle_notes: e.target.value})} className="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-purple-500 transition-colors resize-none" placeholder="e.g. I run 3x a week, sleep 6 hours, mostly vegetarian..." />
            </div>
          </div>
        </div>

        <div className="flex justify-end pt-4">
          <button 
            type="submit" 
            disabled={saving}
            className="bg-gradient-to-r from-teal-500 to-emerald-400 hover:from-teal-400 hover:to-emerald-300 text-slate-900 font-bold py-3 px-8 rounded-xl transition-all flex items-center gap-2 disabled:opacity-70 shadow-lg shadow-teal-500/20"
          >
            {saving ? <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-slate-900"></div> : <Save size={18} />}
            {saving ? 'Saving Profile...' : 'Save Profile Base'}
          </button>
        </div>
      </form>
    </div>
  )
}
