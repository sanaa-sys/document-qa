/** Remove model reasoning tags, including unclosed <think> blocks. */
export function stripThinkTags(text: string): string {
  if (!text) return text
  return text
    .replace(/<think(?:ing)?>[\s\S]*?<\/think(?:ing)?>/gi, '')
    .replace(/<think(?:ing)?>[\s\S]*$/gi, '')
    .replace(/<\/?think(?:ing)?>/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}
