import React from 'react'
import { useRouter } from 'next/router'
import Link from 'next/link'
import { motion } from 'framer-motion'
import { 
  LayoutDashboard, 
  MessageSquare, 
  FileText, 
  User as UserIcon,
  LogOut
} from 'lucide-react'
import { useAuthStore } from '@/stores/authStore'

const navItems = [
  { name: 'Dashboard', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Consultation', path: '/chat', icon: MessageSquare },
  { name: 'Medical Reports', path: '/reports', icon: FileText },
  { name: 'Health Profile', path: '/profile', icon: UserIcon },
]

export default function Sidebar() {
  const router = useRouter()
  const { logout } = useAuthStore()

  const handleLogout = () => {
    logout()
    router.push('/')
  }

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 glass-panel border-l-0 border-y-0 flex flex-col z-40 bg-slate-900/60">
      <div className="p-6">
        <Link href="/dashboard" className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500 flex items-center justify-center font-bold text-white shadow-lg shadow-teal-500/20">
            D
          </div>
          <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-teal-400 to-cyan-300">
            DoctorG
          </span>
        </Link>
      </div>

      <nav className="flex-1 px-4 py-6 space-y-2">
        {navItems.map((item) => {
          const isActive = router.pathname.startsWith(item.path)
          const Icon = item.icon

          return (
            <Link key={item.name} href={item.path} className="block relative">
              {isActive && (
                <motion.div 
                  layoutId="sidebar-active"
                  className="absolute inset-0 bg-teal-500/10 rounded-xl border border-teal-500/20"
                  transition={{ type: "spring", stiffness: 300, damping: 30 }}
                />
              )}
              <div className={`relative flex items-center gap-3 px-4 py-3 rounded-xl transition-colors duration-200 z-10 ${
                isActive ? 'text-teal-400' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}>
                <Icon size={20} className={isActive ? 'drop-shadow-[0_0_8px_rgba(20,184,166,0.5)]' : ''} />
                <span className="font-medium">{item.name}</span>
              </div>
            </Link>
          )
        })}
      </nav>

      <div className="p-4 border-t border-slate-700/50 mt-auto">
        <button 
          onClick={handleLogout}
          className="w-full flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-red-400 hover:bg-red-400/10 rounded-xl transition-colors duration-200"
        >
          <LogOut size={20} />
          <span className="font-medium">Sign Out</span>
        </button>
      </div>
    </aside>
  )
}
