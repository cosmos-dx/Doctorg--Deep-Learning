import Head from 'next/head'
import ChatInterface from '@/components/Chat/ChatInterface'

export default function Home() {
  return (
    <>
      <Head>
        <title>DoctorG - AI Medical Assistant</title>
        <meta name="description" content="AI-powered medical consultation assistant" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <ChatInterface />
      </main>
    </>
  )
}
