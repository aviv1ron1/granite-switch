// SPDX-License-Identifier: Apache-2.0
// Node ESM resolver hook: allow bare deep specifiers into transformers.js's
// `src/` tree (e.g. `@huggingface/transformers/src/models/modeling_utils.js`).
//
// The package's `exports` map intentionally hides `./src/*`, so Node throws
// ERR_PACKAGE_PATH_NOT_EXPORTED for these specifiers. The shim
// (src/granite-switch-register.js) MUST reach those internal modules to register
// the granite_switch architecture, and must bind the whole app to the single
// `src/` module instance. A bundler (Vite) resolves these subpaths natively; in
// Node we install this hook to rewrite the specifier to an absolute file URL,
// which bypasses the exports gate.
//
// Usage: node --import ./test/hf-src-loader.mjs <script>

import { register } from "node:module";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const PREFIX = "@huggingface/transformers/src/";

// Resolve the package root once (via its exported main, then strip /dist/...).
const require = createRequire(import.meta.url);
const main = require.resolve("@huggingface/transformers");
const PKG_ROOT = main.slice(0, main.indexOf("/dist/") + 1);

// Register an in-thread resolver hook.
register(
  new URL("data:text/javascript," + encodeURIComponent(`
    const PREFIX = ${JSON.stringify(PREFIX)};
    const PKG_ROOT = ${JSON.stringify(PKG_ROOT)};
    export async function resolve(specifier, context, nextResolve) {
      if (specifier.startsWith(PREFIX)) {
        const rel = specifier.slice(PREFIX.length);
        const url = ${JSON.stringify(pathToFileURL(PKG_ROOT).href)} + "src/" + rel;
        return { url, shortCircuit: true };
      }
      return nextResolve(specifier, context);
    }
  `)),
  import.meta.url,
);
