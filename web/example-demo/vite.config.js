// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "vite";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

// Vite's dev server falls back to index.html (HTTP 200) for ANY unknown path.
// transformers.js probes for OPTIONAL files (generation_config.json,
// preprocessor_config.json, …) and expects a 404 when they're absent; an HTML
// 200 makes its JSON.parse throw. A real static host / the HF Hub returns 404, so
// this only bites local dev. This plugin returns a true 404 for missing files
// under the served model repo (publicDir/repo), matching production behavior.
// transformers.js src/utils/io.js (imported relatively by image.js/audio.js) pulls
// in node:fs/node:stream for blob saving — unused by the text-only Granite Switch
// path, but Rollup follows the static import and fails on the Node built-ins. This
// plugin redirects ANY resolution ending in `utils/io.js` to a browser stub.
// (A resolve.alias can't catch it: the relative `./io.js` specifier isn't the
// absolute path at alias time.)
function stubNodeIo(stubPath) {
  return {
    name: "stub-transformers-node-io",
    enforce: "pre",
    async resolveId(source, importer) {
      if (source.endsWith("/io.js") && importer && importer.includes("@huggingface/transformers")) {
        return stubPath;
      }
      return null;
    },
  };
}

// Emit the vendored coi-serviceworker.js to dist/ ROOT, unhashed. It can't go through
// public/ because the shipping (Space) build sets `publicDir: false` (HUB_MODE) to avoid
// copying the local dev repo — which would silently drop the SW from the only build that
// needs it. emitFile with a fixed fileName guarantees a stable root URL (the SW's own URL
// becomes its scope) regardless of publicDir. See coi-serviceworker.js for why it exists.
function emitCoiServiceWorker() {
  const swPath = fileURLToPath(new URL("./coi-serviceworker.js", import.meta.url));
  return {
    name: "emit-coi-serviceworker",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "coi-serviceworker.js",
        source: readFileSync(swPath, "utf8"),
      });
    },
  };
}

function repo404() {
  return {
    name: "repo-404-for-missing",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url && req.url.startsWith("/repo/")) {
          const rel = decodeURIComponent(req.url.split("?")[0]);
          const fp = join(server.config.publicDir, rel.slice(1));
          if (!existsSync(fp)) {
            res.statusCode = 404;
            res.end("Not found");
            return;
          }
        }
        next();
      });
    },
  };
}

// The Granite Switch shim deep-imports transformers.js internals
// (@huggingface/transformers/src/...) which the package `exports` map hides — so
// Vite/esbuild (like Node) refuse the bare deep specifier. We add a resolve alias
// that rewrites `@huggingface/transformers/src/*` to absolute file paths under the
// installed package, bypassing the exports gate. This mirrors the Node validators'
// test/hf-src-loader.mjs hook. Binding everything to `src/` is REQUIRED: the
// package's `dist/` build is a separate module instance and does not expose the
// internal maps the shim must mutate.
const require = createRequire(import.meta.url);
const pkgMain = require.resolve("@huggingface/transformers"); // .../dist/transformers.*.
const pkgRoot = pkgMain.slice(0, pkgMain.indexOf("/dist/") + 1);

// `base: "./"` makes the built asset URLs relative, so the static bundle works
// when served from any subpath (e.g. an HF Static Space).
//
// When VITE_MODEL_ID is set (Hub/Space build), the weights load from the HF Hub,
// so we DON'T copy the local dev repo (public/repo, ~1.4 GB) into dist/ — disable
// publicDir for that build. In dev / served-repo mode, publicDir stays on.
const HUB_MODE = !!process.env.VITE_MODEL_ID;
export default defineConfig({
  base: "./",
  publicDir: HUB_MODE ? false : "public",
  plugins: [
    stubNodeIo(fileURLToPath(new URL("./stubs/io.js", import.meta.url))),
    emitCoiServiceWorker(),
    repo404(),
  ],
  // transformers.js's src/ tree (and some transitive deps) reference Node globals
  // (`process`) at module scope. The package's own web build defines these away;
  // we provide a minimal browser shim so the bundled src/ runs in a tab.
  define: {
    "process.env": "{}",
    "process.platform": '""',
    "process.release": "{}",
    global: "globalThis",
  },
  resolve: {
    alias: [
      {
        find: /^@huggingface\/transformers\/src\/(.*)$/,
        replacement: pkgRoot + "src/$1",
      },
      // transformers.js's src/ statically imports onnxruntime-node (its own web
      // build strips it). In the browser only onnxruntime-web is used, so stub
      // the node package — it carries native .node binaries esbuild cannot bundle.
      {
        find: /^onnxruntime-node$/,
        replacement: fileURLToPath(new URL("./stubs/onnxruntime-node.js", import.meta.url)),
      },
      // `sharp` is transformers.js's Node image library — unused by the text-only
      // Granite Switch path and not browser-loadable. Its own web build excludes it.
      {
        find: /^sharp$/,
        replacement: fileURLToPath(new URL("./stubs/sharp.js", import.meta.url)),
      },
    ],
  },
  build: {
    target: "es2022",
    outDir: "dist",
    rollupOptions: {
      output: {
        // transformers.js's src/ tree has circular imports (modeling_utils.js <->
        // registry.js via models.js; the tokenizer auto-map references class
        // bindings hoisted out of order). Rollup's cross-chunk import hoisting can
        // reorder their init and trip a temporal-dead-zone error ("Cannot access
        // 'X' before initialization"). Keep one chunk AND disable transitive-import
        // hoisting so the original (working) init order is preserved. (esbuild's
        // dev prebundle flattens the cycle, which is why `npm run dev` was fine.)
        inlineDynamicImports: true,
        hoistTransitiveImports: false,
      },
    },
  },
  server: {
    // Cross-origin isolation enables the multi-threaded onnxruntime-web WASM
    // build (and is required for some large models). Harmless for the 350M fp32.
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
