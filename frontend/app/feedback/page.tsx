import type { Metadata } from 'next'
import { FeedbackForm } from '@/components/feedback-form'

export const metadata: Metadata = {
  title: 'Feedback - One Third',
  description:
    'Share your thoughts to help make One Third a kinder, clearer medical assistant.',
}

export default function FeedbackPage() {
  return (
    <main className="mx-auto w-full max-w-2xl px-5 py-12 md:py-16">
      <div className="mb-8 flex flex-col items-center gap-3 text-center">
        <h1 className="font-heading text-3xl font-bold tracking-tight text-balance md:text-4xl">
          We&apos;d love your feedback
        </h1>
        <p className="max-w-md leading-relaxed text-muted-foreground text-pretty">
          One Third gets better with every note you send. Tell us how the conversation felt and
          what we can improve.
        </p>
      </div>
      <FeedbackForm />
    </main>
  )
}
