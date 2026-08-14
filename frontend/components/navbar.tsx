'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { Info, MessageSquare, ClipboardList } from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { OneThirdMark } from '@/components/one-third-mark'

interface NavItem {
  href: string
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const navItems: NavItem[] = [
  { href: '/about', label: 'About', icon: Info },
  { href: '/chatbot', label: 'Chatbot', icon: MessageSquare },
  { href: '/feedback', label: 'Feedback', icon: ClipboardList },
]

export function Navbar() {
  const pathname = usePathname()

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-border/70 bg-background/85 backdrop-blur-md supports-[backdrop-filter]:bg-background/70">
      <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-4">
        <Link href="/about" className="shrink-0 transition-opacity hover:opacity-90">
          <OneThirdMark withWordmark priority className="[&_img]:size-10" />
        </Link>

        <div className="flex items-center gap-1.5 sm:gap-2">
          {navItems.map(({ href, label, icon: Icon }) => {
            const isActive = pathname === href
            return (
              <Link key={href} href={href}>
                <Button
                  variant={isActive ? 'default' : 'ghost'}
                  size="sm"
                  className={cn(
                    'gap-2 rounded-lg',
                    !isActive && 'text-foreground/80 hover:text-foreground',
                    isActive && 'shadow-sm',
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span className="hidden sm:inline">{label}</span>
                </Button>
              </Link>
            )
          })}
        </div>
      </div>
    </nav>
  )
}
