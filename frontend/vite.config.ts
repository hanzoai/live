import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    // @hanzo/gui builds one tree for web, native and desktop and branches on
    // this at build time; on web it must be "web" so the native code is dropped.
    "process.env.GUI_TARGET": JSON.stringify("web"),
    "process.env.TAMAGUI_TARGET": JSON.stringify("web"),
  },
  resolve: {
    alias: {
      // The web bindings for the two native modules @hanzo/gui's components
      // import. Same names, web implementations.
      "react-native": "react-native-web",
      "react-native-svg": "@hanzogui/react-native-svg",
    },
  },
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/health": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
