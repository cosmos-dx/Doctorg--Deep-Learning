import React, { useState, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuthStore } from '@/stores/authStore'
import axios from 'axios'
import { format } from 'date-fns'
import toast from 'react-hot-toast'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { UploadCloud, FileType, CheckCircle, AlertTriangle, FileText, ChevronRight, Activity, Beaker } from 'lucide-react'

export default function ReportsPage() {
  const { token } = useAuthStore()
  const [reports, setReports] = useState<any[]>([])
  const [trends, setTrends] = useState<any>(null)
  const [selectedReport, setSelectedReport] = useState<any>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [activeTab, setActiveTab] = useState<'list' | 'trends'>('list')
  const [selectedBiomarker, setSelectedBiomarker] = useState<string | null>(null)

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

  const fetchData = useCallback(async () => {
    try {
      const [repRes, trendRes] = await Promise.all([
        axios.get(`${apiUrl}/api/v1/reports/`, { headers: { Authorization: `Bearer ${token}` } }),
        axios.get(`${apiUrl}/api/v1/reports/trends`, { headers: { Authorization: `Bearer ${token}` } })
      ])
      setReports(repRes.data)
      setTrends(trendRes.data.trends)
      if (trendRes.data.trends && Object.keys(trendRes.data.trends).length > 0) {
        setSelectedBiomarker(Object.keys(trendRes.data.trends)[0])
      }
    } catch (err) {
      toast.error('Failed to load reports')
    }
  }, [apiUrl, token])

  useEffect(() => {
    fetchData()
  }, [fetchData])

  const onDrop = async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post(`${apiUrl}/api/v1/reports/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data', Authorization: `Bearer ${token}` }
      })
      toast.success(res.data.message)
      await fetchData()
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png']
    },
    maxSize: 20 * 1024 * 1024,
    multiple: false
  })

  const loadReportDetail = async (id: str) => {
    try {
      const res = await axios.get(`${apiUrl}/api/v1/reports/${id}`, { headers: { Authorization: `Bearer ${token}` } })
      setSelectedReport(res.data)
    } catch (err) {
      toast.error('Could not load report detail')
    }
  }

  return (
    <div className="space-y-6 pb-10">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-white font-syne">Medical Records</h1>
          <p className="text-slate-400 mt-1">Upload labs and track biomarker trends over time.</p>
        </div>
        <div className="flex bg-slate-900/50 p-1 rounded-xl border border-slate-800">
          <button 
            onClick={() => setActiveTab('list')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'list' ? 'bg-teal-500 text-slate-900 shadow-md' : 'text-slate-400 hover:text-white'}`}
          >
            Documents
          </button>
          <button 
            onClick={() => setActiveTab('trends')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === 'trends' ? 'bg-teal-500 text-slate-900 shadow-md' : 'text-slate-400 hover:text-white'}`}
          >
            Biomarker Trends
          </button>
        </div>
      </div>

      {activeTab === 'list' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            {/* Upload Zone */}
            <div 
              {...getRootProps()} 
              className={`glass-panel border-2 border-dashed p-10 rounded-2xl text-center cursor-pointer transition-all ${
                isDragActive ? 'border-teal-400 bg-teal-500/10' : 'border-slate-700/50 hover:border-teal-500/50 hover:bg-slate-800/80'
              } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
            >
              <input {...getInputProps()} />
              <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-400 shadow-inner">
                {isUploading ? (
                   <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-500" />
                ) : <UploadCloud size={32} />}
              </div>
              <h3 className="text-lg font-bold text-white mb-2">
                {isUploading ? 'Analyzing Document with AI...' : isDragActive ? 'Drop file here' : 'Select or drag & drop file'}
              </h3>
              <p className="text-slate-400 text-sm">PDF, JPG, PNG (max 20MB). AI will automatically extract biomarkers.</p>
            </div>

            {/* Reports List */}
            <div className="space-y-4">
              {reports.map((report) => (
                <div 
                  key={report.id} 
                  onClick={() => loadReportDetail(report.id)}
                  className={`glass-panel p-5 rounded-2xl cursor-pointer hover:border-teal-500/30 transition-all flex items-center justify-between group ${selectedReport?.id === report.id ? 'border-teal-500/50 shadow-[0_0_15px_rgba(20,184,166,0.1)]' : ''}`}
                >
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-slate-800 flex items-center justify-center text-teal-400 group-hover:scale-110 transition-transform">
                      <FileText size={20} />
                    </div>
                    <div>
                      <h4 className="font-bold text-white mb-1 truncate max-w-[200px] sm:max-w-xs">{report.filename}</h4>
                      <div className="flex items-center gap-3 text-xs text-slate-400">
                        <span>{report.report_date ? format(new Date(report.report_date), 'MMM d, yyyy') : format(new Date(report.uploaded_at), 'MMM d, yyyy')}</span>
                        {report.is_medical ? (
                           <span className="flex items-center gap-1 text-emerald-400"><CheckCircle size={12}/> {report.biomarker_count} biomarkers</span>
                        ) : (
                           <span className="flex items-center gap-1 text-yellow-500"><AlertTriangle size={12}/> Non-medical</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <ChevronRight className="text-slate-600 group-hover:text-teal-400 transition-colors" />
                </div>
              ))}
              {reports.length === 0 && !isUploading && (
                <div className="text-center p-8 text-slate-500 border border-dashed border-slate-800 rounded-2xl">
                  No medical reports uploaded yet.
                </div>
              )}
            </div>
          </div>

          {/* Details Sidebar */}
          <div>
            <AnimatePresence mode="popLayout">
              {selectedReport ? (
                <motion.div 
                  initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, scale: 0.95 }}
                  className="glass-panel p-6 rounded-2xl sticky top-24"
                >
                  <div className="flex items-start justify-between mb-6 pb-6 border-b border-slate-800">
                    <div>
                      <h3 className="font-bold text-white text-lg pr-4">{selectedReport.filename}</h3>
                      <p className="text-slate-400 text-sm mt-1">{format(new Date(selectedReport.uploaded_at), 'MMM d, yyyy')}</p>
                    </div>
                    <span className="px-2 py-1 bg-slate-800 text-slate-300 text-[10px] uppercase font-bold rounded-md">
                      {selectedReport.file_type}
                    </span>
                  </div>

                  <div className="mb-6">
                    <h4 className="text-sm font-bold text-teal-400 mb-2 font-syne">AI Summary</h4>
                    <p className="text-sm text-slate-300 leading-relaxed bg-slate-900/50 p-4 rounded-xl border border-slate-800/50">
                      {selectedReport.ai_summary || 'No summary available.'}
                    </p>
                  </div>

                  {selectedReport.biomarkers?.length > 0 && (
                    <div>
                      <h4 className="text-sm font-bold text-white mb-3 flex items-center gap-2"><Beaker size={14} className="text-purple-400"/> Extracted Biomarkers</h4>
                      <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                        {selectedReport.biomarkers.map((b: any) => (
                          <div key={b.id} className="flex justify-between items-center p-3 bg-slate-800/30 rounded-lg border border-slate-700/30 hover:bg-slate-800/50 transition-colors">
                            <span className="text-sm text-slate-200">{b.name}</span>
                            <div className="text-right">
                              <span className={`font-bold ${b.status === 'high' ? 'text-orange-400' : b.status === 'low' ? 'text-blue-400' : 'text-emerald-400'}`}>
                                {b.value !== null ? b.value : '-'} {b.unit}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="glass-panel p-8 rounded-2xl text-center hidden lg:block sticky top-24">
                  <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto mb-4 text-slate-600">
                    <FileType size={24} />
                  </div>
                  <p className="text-slate-400 text-sm">Select a report from the list to view its AI summary and extracted biomarkers.</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>
      )}

      {activeTab === 'trends' && (
        <div className="glass-panel p-8 rounded-2xl min-h-[500px]">
          {!trends || Object.keys(trends).length === 0 ? (
            <div className="text-center pt-20">
              <Activity className="mx-auto text-slate-600 mb-4" size={48} />
              <h3 className="text-xl font-bold text-white mb-2">No Trend Data</h3>
              <p className="text-slate-400">Upload multiple lab reports containing the same biomarkers to see historical trends.</p>
            </div>
          ) : (
            <div className="flex flex-col lg:flex-row gap-8 h-full">
              <div className="w-full lg:w-64 shrink-0 border-r border-slate-800/50 pr-4">
                <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Tracked Biomarkers</h3>
                <div className="space-y-1 max-h-[500px] overflow-y-auto pr-2 custom-scrollbar">
                  {Object.keys(trends).map(name => (
                    <button
                      key={name}
                      onClick={() => setSelectedBiomarker(name)}
                      className={`w-full text-left px-4 py-3 rounded-xl text-sm transition-all flex justify-between items-center ${
                        selectedBiomarker === name 
                          ? 'bg-teal-500/10 text-teal-400 font-bold border border-teal-500/20' 
                          : 'text-slate-300 hover:bg-slate-800/50 hover:text-white'
                      }`}
                    >
                      <span className="truncate max-w-[150px]">{name}</span>
                      <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-400">{trends[name].length}</span>
                    </button>
                  ))}
                </div>
              </div>
              
              <div className="flex-1 flex flex-col items-center justify-center p-4">
                {selectedBiomarker && trends[selectedBiomarker] ? (
                  <div className="w-full h-full max-w-4xl pt-8">
                    <div className="mb-8 text-center">
                      <h2 className="text-2xl font-bold text-white">{selectedBiomarker} Trend</h2>
                      <p className="text-slate-400 mt-1">
                        Historical data across {trends[selectedBiomarker].length} reports
                        {trends[selectedBiomarker][0]?.unit ? ` • Unit: ${trends[selectedBiomarker][0].unit}` : ''}
                      </p>
                    </div>
                    <div className="h-[400px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={trends[selectedBiomarker]} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
                          <XAxis 
                            dataKey="date" 
                            stroke="#64748b" 
                            tickFormatter={(val) => format(new Date(val), 'MMM yyyy')}
                            dy={10}
                          />
                          <YAxis stroke="#64748b" dx={-10} />
                          <RechartsTooltip 
                            contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', color: '#f8fafc' }}
                            labelFormatter={(label) => format(new Date(label), 'MMMM d, yyyy')}
                          />
                          {trends[selectedBiomarker][0]?.reference_high && (
                            <ReferenceLine y={trends[selectedBiomarker][0].reference_high} stroke="#ef4444" strokeDasharray="3 3" label={{ position: 'top', value: 'Upper Limit', fill: '#ef4444', fontSize: 12 }} />
                          )}
                          {trends[selectedBiomarker][0]?.reference_low && (
                            <ReferenceLine y={trends[selectedBiomarker][0].reference_low} stroke="#3b82f6" strokeDasharray="3 3" label={{ position: 'bottom', value: 'Lower Limit', fill: '#3b82f6', fontSize: 12 }} />
                          )}
                          <Line 
                            type="monotone" 
                            dataKey="value" 
                            stroke="#14b8a6" 
                            strokeWidth={3}
                            activeDot={{ r: 8, fill: '#14b8a6', stroke: '#022c22', strokeWidth: 2 }}
                            dot={{ fill: '#0f172a', stroke: '#14b8a6', strokeWidth: 2, r: 4 }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
