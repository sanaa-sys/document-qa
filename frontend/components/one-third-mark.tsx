import Image from 'next/image'
import { cn } from '@/lib/utils'

type OneThirdMarkProps = {
  className?: string
  /** Show wordmark text beside the mark (nav/footer). */
  withWordmark?: boolean
  priority?: boolean
}

export function OneThirdMark({
  className,
  withWordmark = false,
  priority = false,
}: OneThirdMarkProps) {
  if (withWordmark) {
    return (
      <span className={cn('inline-flex items-center gap-2.5', className)}>
        <Image
          src="/images/one-third-logo.png"
          alt="ONE-THIRD"
          width={160}
          height={160}
          priority={priority}
          className="size-9 rounded-md object-cover shadow-sm ring-1 ring-border/60"
        />
        <span className="font-heading text-base font-bold tracking-[0.08em] text-primary">
          ONE-THIRD
        </span>
      </span>
    )
  }

  return (
    <Image
      src="/images/one-third-logo.png"
      alt="ONE-THIRD"
      width={160}
      height={160}
      priority={priority}
      className={cn('rounded-md object-cover shadow-sm ring-1 ring-border/60', className)}
    />
  )
}
