// SPDX-License-Identifier: Apache-2.0
// JS tokenizer + chat-template wrapper for Granite Switch.
//
// Turns CTI text into the token-id sequence GraniteSwitch.generate consumes, and
// decodes generated ids back to text. We reuse @huggingface/transformers'
// AutoTokenizer (already a dependency) — for this model it loads as a plain
// PreTrainedTokenizer from the repo's tokenizer.json despite the non-standard
// `tokenizer_class: "TokenizersBackend"`.
//
// The chat template lives in a SEPARATE `chat_template.jinja` file (it is NOT set
// on the tokenizer), so it must be passed explicitly to apply_chat_template. The
// adapter's control token (e.g. `<|cti_technique_mapping|>`, id 100352) is
// injected by the template only when `adapter_name` is supplied — exactly
// mirroring the Python parity path (scratch/_350m_realprompt_parity.py).
//
// Engine-agnostic: the caller passes the chat-template TEXT (so no Node `fs` is
// pulled into a browser bundle). Read it from the repo's chat_template.jinja in
// Node, or fetch() it in the browser.

import { env as defaultEnv, AutoTokenizer as DefaultAutoTokenizer } from "@huggingface/transformers";

export class GraniteSwitchTokenizer {
  constructor(tokenizer, chatTemplate, meta) {
    this.tok = tokenizer;          // transformers.js PreTrainedTokenizer
    this.chatTemplate = chatTemplate;
    this.meta = meta || {};
  }

  /**
   * Load the tokenizer for a packaged Granite Switch repo.
   *
   * Local:  GraniteSwitchTokenizer.load({ localModelPath, modelName, chatTemplateText, meta })
   * Remote: GraniteSwitchTokenizer.load({ modelId, chatTemplateText, meta })
   *
   * @param {object} opts
   * @param {string} [opts.modelId]         HF Hub repo id (remote load)
   * @param {string} [opts.localModelPath]  base dir for local models
   * @param {string} [opts.modelName]       subdir under localModelPath (local load)
   * @param {string} opts.chatTemplateText  contents of chat_template.jinja
   * @param {object} [opts.meta]            gs_onnx.json metadata (for adapter names)
   * @param {string} [opts.revision]        commit SHA / branch / tag to pin the load to.
   *   When set, browser caching stays ON (loads are pinned, so a re-export under a new
   *   SHA gets fresh cache-key URLs); when omitted, the cache is wiped to avoid staleness.
   * @param {object} [opts.AutoTokenizer]  transformers.js AutoTokenizer to use
   *   (default: the package-root import). Pass the SAME instance the model uses
   *   when the app is bound to transformers.js's `src/` tree (e.g. via the native
   *   shim), so tokenizer + model share one `env`/module instance.
   * @param {object} [opts.env]            matching transformers.js `env` object.
   */
  static async load({
    modelId, localModelPath, modelName, chatTemplateText, meta, revision,
    AutoTokenizer = DefaultAutoTokenizer, env = defaultEnv,
  }) {
    // Stale-tokenizer hazard: transformers.js caches by URL, so a model re-exported
    // under the same name+branch would otherwise serve an old cached tokenizer (e.g.
    // an old control-token spelling) and the adapter control token would silently fail
    // to fire. We avoid that by PINNING the load to a commit SHA (`revision`) — a
    // re-export gets a new SHA, hence new cache-key URLs, so the cache never goes
    // stale. When a revision is given, browser caching is left ON (reloads reuse files);
    // without one (legacy callers), we keep the old belt-and-suspenders eviction.
    if (!revision) {
      env.useBrowserCache = false;
      if (typeof caches !== "undefined") {
        try { await caches.delete("transformers-cache"); } catch (_) { /* best effort */ }
      }
    }

    let tok;
    if (modelId) {
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      tok = await AutoTokenizer.from_pretrained(modelId, revision ? { revision } : {});
    } else {
      env.allowLocalModels = true;
      env.allowRemoteModels = false;
      env.localModelPath = localModelPath;
      tok = await AutoTokenizer.from_pretrained(modelName);
    }
    return new GraniteSwitchTokenizer(tok, chatTemplateText, meta);
  }

  /**
   * Render a user message through the chat template (injecting the adapter
   * control token) and tokenize it to ids.
   *
   * Each adapter was trained on a SPECIFIC prompt framing, not the raw input;
   * sending the wrong framing makes the model emit garbage even though the
   * control token fires. The framing is two parts: an optional `instruction`
   * line and an optional XML tag wrapping the input. Examples (all three live in
   * the demo model):
   *   - cti-technique-mapping: instruction + `<cti>…</cti>`  (the default tag)
   *   - genai-attack-vector:   instruction + `<incident>…</incident>`
   *   - text-to-json:          no instruction, no tag — query + schema preamble
   *     (use `buildContent` to express that shape)
   *
   * Backward compatibility: `wrapTag` defaults to `"cti"`, so an
   * `encode(text, { instruction })` call produces the exact legacy framing
   * `${instruction}\n\n<cti>\n${text}\n</cti>` (the goldens depend on this).
   *
   * @param {string} text  the input text (or raw user content if no instruction)
   * @param {object} [opts]
   * @param {string|null} [opts.adapterName]  adapter to activate; defaults to the
   *                                     first adapter in meta.adapter_names. Pass
   *                                     `null` EXPLICITLY to render WITHOUT firing
   *                                     any control token (the plain base model).
   * @param {string} [opts.instruction]  instruction line prepended to the content.
   * @param {string|null} [opts.wrapTag="cti"]  XML tag wrapping `text`. Default
   *                                     `"cti"` preserves legacy behavior; pass
   *                                     `"incident"` etc. for other adapters, or
   *                                     `null`/`""` to not wrap.
   * @param {(args: {text: string, instruction?: string, wrapTag?: string|null}) => string} [opts.buildContent]
   *                                     full override of the user-message content;
   *                                     when supplied, `instruction`/`wrapTag` are
   *                                     ignored (use for the text-to-json format).
   * @param {boolean} [opts.addGenerationPrompt=true]
   * @returns {number[]} token ids
   */
  /**
   * Render the user message through the chat template WITHOUT tokenizing, and
   * return the resulting prompt STRING — i.e. the exact "raw prompt after the
   * prompt template" the model is decoded against, including the injected adapter
   * control token (when an adapter is active) and the instruction/tag framing.
   *
   * `encode()` is `tokenize(renderPrompt(...))`; this method exists so callers
   * (e.g. the browser demo's "view raw prompt") can show the same string the
   * tokenizer feeds the model, with no extra template logic to keep in sync.
   *
   * Accepts the SAME options as {@link encode}.
   * @returns {string} the templated prompt
   */
  renderPrompt(text, { adapterName, instruction, wrapTag = "cti", buildContent, addGenerationPrompt = true } = {}) {
    // `adapterName: null` (explicit) => base model, no control token. `undefined`
    // => default to the first adapter. A truthy string => that adapter.
    const adapter = adapterName === null
      ? undefined
      : adapterName || (this.meta.adapter_names && this.meta.adapter_names[0]);
    // Content framing: a full override wins; otherwise an instruction line plus
    // an optional tag wrapper (default `<cti>` keeps the legacy path byte-identical);
    // with no instruction the raw text passes through.
    let content;
    if (typeof buildContent === "function") {
      content = buildContent({ text, instruction, wrapTag });
    } else if (instruction) {
      content = wrapTag
        ? `${instruction}\n\n<${wrapTag}>\n${text}\n</${wrapTag}>`
        : `${instruction}\n\n${text}`;
    } else {
      content = text;
    }
    return this.tok.apply_chat_template(
      [{ role: "user", content }],
      {
        tokenize: false,
        add_generation_prompt: addGenerationPrompt,
        chat_template: this.chatTemplate,
        adapter_name: adapter,
      },
    );
  }

  encode(text, opts = {}) {
    const rendered = this.renderPrompt(text, opts);
    const enc = this.tok(rendered, { add_special_tokens: false });
    // enc.input_ids is a transformers.js Tensor ([1, seq]); flatten to a JS array.
    return Array.from(enc.input_ids.data, Number);
  }

  /** Decode token ids back to text. */
  decode(ids, { skipSpecialTokens = true } = {}) {
    return this.tok.decode(ids, { skip_special_tokens: skipSpecialTokens });
  }

  /** EOS token id (for early-stopping a generation loop). */
  get eosTokenId() {
    return this.tok.eos_token_id ?? null;
  }
}
