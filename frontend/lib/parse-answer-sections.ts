export type AnswerSection = {
  title: string
  body: string
}

export type ParsedAnswer = {
  intro: string
  sections: AnswerSection[]
}

export type SectionKind = 'remedies' | 'otc' | 'avoid' | 'other'

export function parseAnswerSections(text: string): ParsedAnswer {
  const lines = (text || '').replace(/\r\n/g, '\n').split('\n')
  const sections: AnswerSection[] = []
  const introLines: string[] = []
  let current: AnswerSection | null = null

  for (const line of lines) {
    const heading = line.match(/^#{1,3}\s+(.+?)\s*$/)
    if (heading) {
      if (current) sections.push(current)
      current = { title: heading[1].trim(), body: '' }
      continue
    }
    if (current) {
      current.body += (current.body ? '\n' : '') + line
    } else {
      introLines.push(line)
    }
  }
  if (current) sections.push(current)

  return {
    intro: introLines.join('\n').trim(),
    sections: sections.map((s) => ({ ...s, body: s.body.trim() })),
  }
}

export function sectionKind(title: string): SectionKind {
  const t = title.toLowerCase()
  if (
    t.includes('home remed') ||
    t.includes('natural remed') ||
    t.includes('effective home')
  ) {
    return 'remedies'
  }
  if (
    t.includes('over-the-counter') ||
    t.includes('over the counter') ||
    /\botc\b/.test(t) ||
    t.includes('medication')
  ) {
    return 'otc'
  }
  if (t.includes('not to do') || t.includes('avoid') || t.includes('do not')) {
    return 'avoid'
  }
  return 'other'
}
