'use client'

import { useState } from 'react'
import { CheckCircle2, Loader2, Star } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { sendFeedbackEmail } from '@/lib/send-feedback-email'
import { cn } from '@/lib/utils'

const CATEGORY_OPTIONS = [
    'Accuracy',
    'Helpfulness',
    'Tone & clarity',
    'Sources & citations',
    'Bug report',
    'Other',
]

function StarRating({
    value,
    onChange,
    label,
}: {
    value: number
    onChange: (n: number) => void
    label: string
}) {
    const [hover, setHover] = useState(0)
    return (
        <div className="flex items-center gap-1.5" role="radiogroup" aria-label={label}>
            {[1, 2, 3, 4, 5].map((n) => (
                <button
                    key={n}
                    type="button"
                    role="radio"
                    aria-checked={value === n}
                    aria-label={`${n} star${n > 1 ? 's' : ''}`}
                    onClick={() => onChange(n)}
                    onMouseEnter={() => setHover(n)}
                    onMouseLeave={() => setHover(0)}
                    className="rounded-md p-1 outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                    <Star
                        className={cn(
                            'size-7 transition-colors',
                            (hover || value) >= n
                                ? 'fill-primary text-primary'
                                : 'text-muted-foreground/40',
                        )}
                    />
                </button>
            ))}
        </div>
    )
}

export function FeedbackForm() {
    const [rating, setRating] = useState(0)
    const [accuracy, setAccuracy] = useState(0)
    const [categories, setCategories] = useState<string[]>([])
    const [comments, setComments] = useState('')
    const [status, setStatus] = useState<'idle' | 'submitting' | 'done' | 'error'>('idle')
    const [errorMsg, setErrorMsg] = useState('')

    function toggleCategory(c: string) {
        setCategories((prev) =>
            prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c],
        )
    }

    async function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        if (!comments.trim() || status === 'submitting') return

        setStatus('submitting')
        setErrorMsg('')
        const payload = {
            rating,
            accuracy,
            categories,
            comments: comments.trim(),
        }
        try {
            await sendFeedbackEmail(payload)
            void fetch('/api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            }).catch(() => {})
            setStatus('done')
        } catch (err) {
            setStatus('error')
            const text =
                err && typeof err === 'object' && 'text' in err
                    ? String((err as { text?: string }).text)
                    : ''
            setErrorMsg(
                text ||
                    (err instanceof Error ? err.message : 'Could not send feedback. Please try again.'),
            )
        }
    }

    if (status === 'done') {
        return (
            <div className="flex flex-col items-center gap-4 rounded-3xl border border-border/70 bg-card p-10 text-center">
                <span className="inline-flex size-14 items-center justify-center rounded-2xl bg-accent/30 text-accent-foreground">
                    <CheckCircle2 className="size-7" />
                </span>
                <h2 className="font-heading text-xl font-semibold">Thank you!</h2>
                <p className="max-w-sm leading-relaxed text-muted-foreground text-pretty">
                    Your feedback helps make One Third kinder, clearer, and more accurate
                    for everyone.
                </p>
                <Button
                    variant="outline"
                    onClick={() => {
                        setRating(0)
                        setAccuracy(0)
                        setCategories([])
                        setComments('')
                        setStatus('idle')
                    }}
                >
                    Send more feedback
                </Button>
            </div>
        )
    }

    return (
        <form
            onSubmit={handleSubmit}
            className="flex flex-col gap-6 rounded-3xl border border-border/70 bg-card p-6 md:p-8"
        >
            {/* Overall rating */}
            <fieldset className="flex flex-col gap-2">
                <legend className="mb-1 text-sm font-semibold text-foreground">
                    How was your overall experience?
                </legend>
                <StarRating value={rating} onChange={setRating} label="Overall rating" />
            </fieldset>

            {/* Accuracy rating */}
            <fieldset className="flex flex-col gap-2">
                <legend className="mb-1 text-sm font-semibold text-foreground">
                    How accurate were the answers?
                </legend>
                <StarRating value={accuracy} onChange={setAccuracy} label="Accuracy rating" />
            </fieldset>

            {/* Categories (multi-select) */}
            <div className="flex flex-col gap-2">
                <span className="text-sm font-semibold text-foreground">
                    What is it about?{' '}
                    <span className="font-normal text-muted-foreground">(select all that apply)</span>
                </span>
                <div className="flex flex-wrap gap-2">
                    {CATEGORY_OPTIONS.map((c) => {
                        const selected = categories.includes(c)
                        return (
                            <button
                                key={c}
                                type="button"
                                onClick={() => toggleCategory(c)}
                                aria-pressed={selected}
                                className={cn(
                                    'rounded-full border px-3.5 py-1.5 text-sm font-medium transition-colors',
                                    selected
                                        ? 'border-primary bg-primary/12 text-primary'
                                        : 'border-border/70 text-muted-foreground hover:bg-muted',
                                )}
                            >
                                {c}
                            </button>
                        )
                    })}
                </div>
            </div>

            {/* Comments */}
            <div className="flex flex-col gap-2">
                <label htmlFor="feedback-comments" className="text-sm font-semibold text-foreground">
                    Your feedback <span className="text-primary">*</span>
                </label>
                <textarea
                    id="feedback-comments"
                    required
                    rows={5}
                    value={comments}
                    onChange={(e) => setComments(e.target.value)}
                    placeholder="Tell us what worked well or what we could improve…"
                    className="resize-y rounded-2xl border border-input bg-background px-4 py-3 text-sm leading-relaxed outline-none transition-colors focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/40"
                />
            </div>

            {status === 'error' && (
                <p className="rounded-xl border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-foreground">
                    {errorMsg}
                </p>
            )}

            <Button
                type="submit"
                size="lg"
                disabled={!comments.trim() || status === 'submitting'}
                className="h-11 self-start px-6 text-base"
            >
                {status === 'submitting' ? (
                    <>
                        <Loader2 className="size-4 animate-spin" />
                        Sending…
                    </>
                ) : (
                    'Send feedback'
                )}
            </Button>
        </form>
    )
}
