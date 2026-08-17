'use client'

import { useEffect, useRef, useState } from 'react'
import { ArrowUp, BookText, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { OneThirdMark } from '@/components/one-third-mark'
import { stripThinkTags } from '@/lib/strip-think'
import { cn } from '@/lib/utils'

type Source = { title?: string; source?: string; page?: number | string } | string

type Message = {
    id: string
    role: 'user' | 'assistant'
    content: string
    sources?: Source[]
    error?: boolean
}

const suggestions = [
    'What are the benefits of eating dates?',
    'How to treat PCOS (PMOS)?',
    'How is Salah healing and therapeutic?',
]

function sourceLabel(s: Source): string {
    if (typeof s === 'string') return s
    const base = s.title ?? s.source ?? 'Source'
    return s.page != null ? `${base} · p.${s.page}` : base
}

export function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [loading, setLoading] = useState(false)
    const scrollRef = useRef<HTMLDivElement>(null)
    const composingRef = useRef(false)

    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
    }, [messages, loading])

    async function send(text: string) {
        const trimmed = text.trim()
        if (!trimmed || loading) return

        const userMessage: Message = {
            id: crypto.randomUUID(),
            role: 'user',
            content: trimmed,
        }
        const history = [...messages, userMessage].map((m) => ({
            role: m.role,
            content: m.content,
        }))

        setMessages((prev) => [...prev, userMessage])
        setInput('')
        setLoading(true)

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: trimmed, history }),
            })
            const data = await res.json()

            if (!res.ok) {
                setMessages((prev) => [
                    ...prev,
                    {
                        id: crypto.randomUUID(),
                        role: 'assistant',
                        content: data?.error ?? 'Something went wrong. Please try again.',
                        error: true,
                    },
                ])
            } else {
                setMessages((prev) => [
                    ...prev,
                    {
                        id: crypto.randomUUID(),
                        role: 'assistant',
                        content:
                            stripThinkTags(data.answer || '') ||
                            'I could not find an answer for that.',
                        sources: Array.isArray(data.sources) ? data.sources : [],
                    },
                ])
            }
        } catch {
            setMessages((prev) => [
                ...prev,
                {
                    id: crypto.randomUUID(),
                    role: 'assistant',
                    content: 'Could not reach the chatbot. Please try again shortly.',
                    error: true,
                },
            ])
        } finally {
            setLoading(false)
        }
    }

    function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
        if (
            e.key === 'Enter' &&
            !e.shiftKey &&
            !composingRef.current &&
            e.nativeEvent.keyCode !== 229
        ) {
            e.preventDefault()
            send(input)
        }
    }

    return (
        <div className="flex h-full flex-col overflow-hidden rounded-[1.75rem] border border-border/70 bg-card">
            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 md:px-6">
                {messages.length === 0 ? (
                    <div className="mx-auto flex max-w-md flex-col items-center gap-5 py-10 text-center">
                        <OneThirdMark className="size-16" />
                        <div className="flex flex-col gap-1.5">
                            <h2 className="font-heading text-xl font-semibold text-primary">
                                Hi, I&apos;m ONE-THIRD
                            </h2>
                            <p className="leading-relaxed text-muted-foreground text-pretty">
                                Ask me a health question and I&apos;ll answer with information
                                drawn from trusted medical sources.
                            </p>
                        </div>
                        <div className="flex w-full flex-col gap-2">
                            {suggestions.map((s) => (
                                <button
                                    key={s}
                                    type="button"
                                    onClick={() => send(s)}
                                    className="rounded-xl border border-border/70 bg-background px-4 py-2.5 text-left text-sm text-foreground transition-colors hover:bg-muted"
                                >
                                    {s}
                                </button>
                            ))}
                        </div>
                    </div>
                ) : (
                    <ul className="mx-auto flex max-w-2xl flex-col gap-5">
                        {messages.map((m) => (
                            <li
                                key={m.id}
                                className={cn(
                                    'flex',
                                    m.role === 'user' ? 'justify-end' : 'justify-start',
                                )}
                            >
                                <div
                                    className={cn(
                                        'max-w-[85%] rounded-2xl px-4 py-3 leading-relaxed',
                                        m.role === 'user'
                                            ? 'bg-primary text-primary-foreground'
                                            : m.error
                                                ? 'border border-destructive/30 bg-destructive/10 text-foreground'
                                                : 'bg-muted text-foreground',
                                    )}
                                >
                                    <p className="whitespace-pre-wrap text-pretty">{m.content}</p>
                                    {m.sources && m.sources.length > 0 && (
                                        <div className="mt-3 flex flex-col gap-1.5 border-t border-border/60 pt-3">
                                            <span className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
                                                <BookText className="size-3.5" />
                                                Sources
                                            </span>
                                            <ul className="flex flex-wrap gap-1.5">
                                                {m.sources.map((s, i) => (
                                                    <li
                                                        key={i}
                                                        className="rounded-md bg-background px-2 py-1 text-xs text-muted-foreground"
                                                    >
                                                        {sourceLabel(s)}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    )}
                                </div>
                            </li>
                        ))}
                        {loading && (
                            <li className="flex justify-start">
                                <div className="flex items-center gap-2 rounded-2xl bg-muted px-4 py-3 text-muted-foreground">
                                    <Loader2 className="size-4 animate-spin" />
                                    <span className="text-sm">One Third is thinking…</span>
                                </div>
                            </li>
                        )}
                    </ul>
                )}
            </div>

            {/* Composer */}
            <div className="border-t border-border/70 bg-card px-4 py-4 md:px-6">
                <form
                    onSubmit={(e) => {
                        e.preventDefault()
                        send(input)
                    }}
                    className="mx-auto flex max-w-2xl items-end gap-2"
                >
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        onCompositionStart={() => (composingRef.current = true)}
                        onCompositionEnd={() => (composingRef.current = false)}
                        rows={1}
                        placeholder="Ask a health question…"
                        aria-label="Message"
                        className="max-h-40 min-h-11 flex-1 resize-none rounded-2xl border border-input bg-background px-4 py-3 text-sm leading-relaxed outline-none transition-colors focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
                    />
                    <Button
                        type="submit"
                        size="icon-lg"
                        disabled={!input.trim() || loading}
                        aria-label="Send message"
                        className="size-11 rounded-2xl"
                    >
                        <ArrowUp className="size-5" />
                    </Button>
                </form>
                <p className="mx-auto mt-2 max-w-2xl text-center text-xs text-muted-foreground text-pretty">
                    One Third can make mistakes. Always confirm important decisions with a
                    qualified healthcare professional.
                </p>
            </div>
        </div>
    )
}
