interface PlaceholderTabProps {
  title: string;
  description: string;
}

export function PlaceholderTab({ title, description }: PlaceholderTabProps) {
  return (
    <div className="flex flex-col items-center justify-center h-48 text-muted-foreground">
      <h3 className="text-sm font-medium text-foreground">{title}</h3>
      <p className="text-xs mt-1">{description}</p>
      <p className="text-3xs mt-3 opacity-50">Coming soon</p>
    </div>
  );
}
