import Image from 'next/image'
import Link from 'next/link'
import type { Metadata } from 'next'
import { Button } from '@/components/ui/button'
import {
    BookOpenCheck,
    HeartHandshake,
    MessageCircleHeart,
    ShieldCheck,
    Sparkles,
} from 'lucide-react'

export const metadata: Metadata = {
    title: 'About',
    description:
       'Your caring health companion — answers with warmth and clarity, grounded in authentic Prophetic wisdom and modern medical sources so you can trust what you read.',
}

const features = [
    {
        icon: BookOpenCheck,
        title: 'Grounded in real sources',
        body: 'Every answer is retrieved from your trusted medical documents, so responses stay accurate and citeable — not made up.',
    },
    {
        icon: MessageCircleHeart,
        title: 'Plain, kind language',
        body: 'One Third explains conditions, medications, and next steps in warm, everyday words anyone can follow.',
    },
    {
        icon: ShieldCheck,
        title: 'Careful by design',
        body: 'When something needs a real clinician, One Third says so and points you toward professional care.',
    },
]

const steps = [
    {
        title: 'Ask a question',
        body: 'Type your health question in plain language, just like you would ask a friendly nurse.',
    },
    {
        title: 'We retrieve the facts',
        body: 'One Third searches your medical knowledge base for the most relevant, trustworthy passages.',
    },
    {
        title: 'You get a clear answer',
        body: 'A warm, easy-to-read response grounded in those sources — with gentle reminders when to see a doctor.',
    },
]

export default function AboutPage() {
    return (
        <main>
            {/* Hero */}
            <section className="mx-auto grid w-full max-w-6xl items-center gap-10 px-5 py-14 md:grid-cols-2 md:py-20">
                <div className="flex flex-col items-start gap-6">
                    <span className="inline-flex items-center gap-2 rounded-full bg-accent/35 px-3.5 py-1.5 text-sm font-semibold text-accent-foreground">
                        <Sparkles className="size-4 text-primary" />
                        Meet ONE-THIRD
                    </span>
                    <h1 className="font-heading text-4xl font-bold leading-tight tracking-tight text-balance text-primary md:text-5xl">
                        ONE-THIRD
                    </h1>
                    <p className="max-w-md text-lg leading-relaxed text-muted-foreground text-pretty">
                        Your caring health companion — answers with warmth and clarity, grounded in authentic Prophetic wisdom and modern medical sources so you can trust what you read.
                    </p>
                    <div className="flex flex-wrap items-center gap-3">
                        <Button size="lg" className="h-11 rounded-xl px-6 text-base" render={<Link href="/chatbot" />}>
                            Start chatting
                        </Button>
                        <Button
                            size="lg"
                            variant="outline"
                            className="h-11 rounded-xl px-6 text-base"
                            render={<Link href="/feedback" />}
                        >
                            Share feedback
                        </Button>
                    </div>
                </div>

                <div className="relative flex items-center justify-center">
                    <div className="absolute -inset-6 -z-10 rounded-[2rem] bg-gradient-to-br from-accent/50 via-brand-sky/25 to-primary/20 blur-2xl" />
                    <div className="overflow-hidden rounded-[2rem] border border-border/70 bg-black p-6 shadow-sm sm:p-8">
                        <Image
                            src="/images/one-third-logo.png"
                            alt="ONE-THIRD logo — wind, water, and growth"
                            width={640}
                            height={640}
                            priority
                            className="mx-auto h-auto w-full max-w-md object-contain"
                        />
                    </div>
                </div>
            </section>

            {/* Features */}
            <section className="mx-auto w-full max-w-6xl px-5 py-6">
                <div className="grid gap-5 md:grid-cols-3">
                    {features.map((f) => (
                        <article
                            key={f.title}
                            className="flex flex-col gap-3 rounded-3xl border border-border/70 bg-card p-6"
                        >
                            <span className="inline-flex size-11 items-center justify-center rounded-2xl bg-primary/12 text-primary">
                                <f.icon className="size-5" />
                            </span>
                            <h3 className="font-heading text-lg font-semibold">{f.title}</h3>
                            <p className="leading-relaxed text-muted-foreground text-pretty">{f.body}</p>
                        </article>
                    ))}
                </div>
            </section>

            {/* How it works */}
            <section className="mx-auto w-full max-w-6xl px-5 py-14 md:py-20">
                <div className="mb-10 flex flex-col items-center gap-3 text-center">
                    <span className="inline-flex items-center gap-2 rounded-full bg-primary/12 px-3.5 py-1.5 text-sm font-semibold text-primary">
                        <HeartHandshake className="size-4" />
                        How it works
                    </span>
                    <h2 className="font-heading text-3xl font-bold tracking-tight text-balance md:text-4xl">
                        Three gentle steps to a clear answer
                    </h2>
                </div>
                <ol className="grid gap-5 md:grid-cols-3">
                    {steps.map((s, i) => (
                        <li
                            key={s.title}
                            className="flex flex-col gap-3 rounded-3xl border border-border/70 bg-card p-6"
                        >
                            <span className="inline-flex size-9 items-center justify-center rounded-full bg-accent/30 font-heading text-sm font-bold text-accent-foreground">
                                {i + 1}
                            </span>
                            <h3 className="font-heading text-lg font-semibold">{s.title}</h3>
                            <p className="leading-relaxed text-muted-foreground text-pretty">{s.body}</p>
                        </li>
                    ))}
                </ol>
            </section>

            {/* CTA */}
            <section className="mx-auto w-full max-w-6xl px-5 pb-20">
                <div className="flex flex-col items-center gap-5 rounded-[2rem] border border-border/70 bg-primary/8 px-6 py-12 text-center">
                    <h2 className="font-heading text-3xl font-bold tracking-tight text-balance">
                        Have a health question on your mind?
                    </h2>
                    <p className="max-w-lg leading-relaxed text-muted-foreground text-pretty">
                        One Third is here to help you understand it — kindly, clearly, and
                        grounded in real medical knowledge.
                    </p>
                    <Button size="lg" className="h-11 rounded-xl px-6 text-base" render={<Link href="/chatbot" />}>
                        Talk to ONE-THIRD
                    </Button>
                </div>
            </section>
        </main>
    )
}
