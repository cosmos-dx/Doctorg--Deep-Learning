import '@/styles/globals.css'
import type { AppProps } from 'next/app'
import Head from 'next/head'
import { Toaster } from 'react-hot-toast'
import Layout from '@/components/Layout/Layout'
import { useRouter } from 'next/router'

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter()
  // Pages that don't need the sidebar layout (like login)
  const isAuthPage = router.pathname === '/' || router.pathname === '/auth'

  return (
    <>
      <Head>
        <title>DoctorG - AI Medical Consultation</title>
        <meta name="description" content="AI-powered medical consultation platform" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#14B8A6" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      {isAuthPage ? (
        <Component {...pageProps} />
      ) : (
        <Layout>
          <Component {...pageProps} />
        </Layout>
      )}
      
      <Toaster 
        position="top-right"
        toastOptions={{
          className: 'glass-panel text-slate-100',
          style: {
            background: 'rgba(30, 41, 59, 0.8)',
            backdropFilter: 'blur(12px)',
            color: '#fff',
            border: '1px solid rgba(51, 65, 85, 0.5)'
          }
        }} 
      />
    </>
  )
}
