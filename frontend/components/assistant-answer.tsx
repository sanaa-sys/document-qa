import type { ReactNode } from 'react'
import { Ban, BookText, Leaf, Pill } from 'lucide-react'
import {
  parseAnswerSections,
  sectionKind,
  type SectionKind,
} from '@/lib/parse-answer-sections'
import { cn } from '@/lib/utils'

const KIND_STYLE: Record<
  SectionKind,
  { wrap: string; iconWrap: string; icon: typeof Leaf }
> = {
  remedies: {
    wrap: 'border-emerald-200/80 bg-emerald-50/70 dark:border-emerald-800/70 dark:bg-emerald-950/40',
    iconWrap: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/80 dark:text-emerald-100',
    icon: Leaf,
  },
  otc: {
    wrap: 'border-sky-200/80 bg-sky-50/70 dark:border-sky-800/70 dark:bg-sky-950/40',
    iconWrap: 'bg-sky-100 text-sky-800 dark:bg-sky-900/80 dark:text-sky-100',
    icon: Pill,
  },
  avoid: {
    wrap: 'border-rose-200/80 bg-rose-50/70 dark:border-rose-800/70 dark:bg-rose-950/40',
    iconWrap: 'bg-rose-100 text-rose-800 dark:bg-rose-900/80 dark:text-rose-100',
    icon: Ban,
  },
  other: {
    wrap: 'border-border/70 bg-background/70',
    iconWrap: 'bg-muted text-foreground',
    icon: BookText,
  },
}

function applyInline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((part, i) => {
    const bold = part.match(/^\*\*([^*]+)\*\*$/)
    if (bold) {
      return (
        <strong key={i} className="font-semibold">
          {bold[1]}
        </strong>
      )
    }
    return part
  })
}

function RichBody({ text }: { text: string }) {
  const lines = text.split('\n')
  const nodes: ReactNode[] = []
  let list: { ordered: boolean; items: string[] } | null = null

  const flushList = () => {
    if (!list || list.items.length === 0) return
    const Tag = list.ordered ? 'ol' : 'ul'
    nodes.push(
      <Tag
        key={`list-${nodes.length}`}
        className={cn(
          'my-1.5 flex flex-col gap-1 pl-4 text-pretty',
          list.ordered ? 'list-decimal' : 'list-disc',
        )}
      >
        {list.items.map((item, i) => (
          <li key={i}>{applyInline(item)}</li>
        ))}
      </Tag>,
    )
    list = null
  }

  for (const line of lines) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/)
    if (bullet || numbered) {
      const ordered = Boolean(numbered)
      const item = (bullet?.[1] ?? numbered?.[1] ?? '').trim()
      if (!list || list.ordered !== ordered) {
        flushList()
        list = { ordered, items: [item] }
      } else {
        list.items.push(item)
      }
      continue
    }
    flushList()
    if (!line.trim()) continue
    nodes.push(
      <p key={`p-${nodes.length}`} className="text-pretty">
        {applyInline(line.trim())}
      </p>,
    )
  }
  flushList()

  if (nodes.length === 0) return null
  return <div className="flex flex-col gap-1.5 text-sm leading-relaxed">{nodes}</div>
}

export function AssistantAnswer({ content }: { content: string }) {
  const { intro, sections } = parseAnswerSections(content)

  if (sections.length === 0) {
    return (
      <div className="text-pretty">
        <RichBody text={content} />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {intro ? <RichBody text={intro} /> : null}
      {sections.map((section) => {
        const kind = sectionKind(section.title)
        const style = KIND_STYLE[kind]
        const Icon = style.icon
        return (
          <section
            key={section.title}
            className={cn('rounded-xl border px-3 py-2.5', style.wrap)}
          >
            <h3 className="mb-1.5 flex items-center gap-2 font-heading text-sm font-semibold">
              <span
                className={cn(
                  'flex size-6 items-center justify-center rounded-md',
                  style.iconWrap,
                )}
              >
                <Icon className="size-3.5" aria-hidden />
              </span>
              {section.title}
            </h3>
            {section.body ? <RichBody text={section.body} /> : null}
          </section>
        )
      })}
    </div>
  )
}
