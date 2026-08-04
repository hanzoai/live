import { GuiProvider } from "@hanzo/gui";
import { Toaster } from "@hanzo/ui";
import config from "@hanzo/ui/gui-config";
import { StreamPage } from "./pages/StreamPage";
import "./index.css";

function App() {
  return (
    <GuiProvider config={config} defaultTheme="dark">
      <StreamPage />
      <Toaster />
    </GuiProvider>
  );
}

export default App;
