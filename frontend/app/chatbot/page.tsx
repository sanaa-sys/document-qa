import type { Metadata } from 'next'
import { ChatInterface } from '@/components/chat-interface'

export const metadata: Metadata = {
    title: 'Chatbot — One Third',
    description: 'Ask One Third your health questions and get answers grounded in trusted medical sources.',
}

export default function ChatbotPage() {
    return (
        <main className="mx-auto flex h-[calc(100dvh-4rem)] w-full max-w-4xl flex-col px-4 py-5 md:px-5 md:py-6">
            <div className="mb-4 flex flex-col gap-1">
                <h1 className="font-heading text-2xl font-bold tracking-tight">Chatbot</h1>
                <p className="text-sm text-muted-foreground text-pretty">
                    A warm, retrieval-grounded assistant for your health questions.
                </p>
            </div>
            <div className="min-h-0 flex-1">
                <ChatInterface />
            </div>
        </main>
    )
}
