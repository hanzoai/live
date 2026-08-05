/**
 * Registers the @hanzo/ui gui config with the type system, so the shorthand
 * style props the components take (bg / px / py / items / justify / gap /
 * rounded / minH / maxW …) resolve to concrete types in this app.
 *
 * Same config object that `<GuiProvider config={…}>` is given at runtime — the
 * types and the runtime therefore cannot drift. Ambient + type-only.
 */
import type { Conf } from "@hanzo/ui/gui-config";

declare module "@hanzogui/web" {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type -- declaration merge, not a subtype
  interface GuiCustomConfig extends Conf {}
}

declare module "@hanzogui/core" {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type -- declaration merge, not a subtype
  interface GuiCustomConfig extends Conf {}
}
