import { BookOpenText } from "@hanzogui/lucide-icons-2";
import { HanzoLogo } from "@hanzo/logo/react";

export function Header() {
  return (
    <header className="app-header">
      <div className="app-header-group">
        <HanzoLogo variant="white" size={32} />
        <h1 className="app-title">Hanzo Live</h1>
      </div>
      <div className="app-header-group">
        <a
          href="https://github.com/hanzoai/live"
          target="_blank"
          rel="noopener noreferrer"
          className="app-link"
        >
          <img src="/assets/github-mark-white.svg" alt="GitHub" />
        </a>
        <a
          href="https://discord.gg/mnfGR4Fjhp"
          target="_blank"
          rel="noopener noreferrer"
          className="app-link"
        >
          <img src="/assets/discord-symbol-white.svg" alt="Discord" />
        </a>
        <a
          href="https://docs.daydream.live/knowledge-hub/research-references/about-video-and-world-models"
          target="_blank"
          rel="noopener noreferrer"
          className="app-link"
        >
          <BookOpenText size={20} />
        </a>
      </div>
    </header>
  );
}
