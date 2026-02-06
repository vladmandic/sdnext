import { TooltipProvider } from "@/components/ui/tooltip";
import { AppShell } from "@/components/layout/AppShell";
import "./App.css";

function App() {
  return (
    <TooltipProvider delayDuration={100}>
      <AppShell />
    </TooltipProvider>
  );
}

export default App;
