import Link from 'next/link'
import { OneThirdMark } from '@/components/one-third-mark'

export function SiteFooter() {
  return (
    <footer className="border-t border-border/70 bg-sidebar/70">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-5 py-8 sm:flex-row sm:items-center sm:justify-between">
        <Link href="/about" className="transition-opacity hover:opacity-90">
          <OneThirdMark withWordmark className="[&_img]:size-8" />
        </Link>
        <nav aria-label="Footer" className="flex items-center gap-5 text-sm text-muted-foreground">
          <Link href="/about" className="hover:text-foreground">
            About
          </Link>
          <Link href="/chatbot" className="hover:text-foreground">
            Chatbot
          </Link>
          <Link href="/feedback" className="hover:text-foreground">
            Feedback
          </Link>
        </nav>
        <p className="text-xs text-muted-foreground text-pretty">
          Not a substitute for professional medical advice.
        </p>
      </div>
    </footer>
  )
}
