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

import { env, AutoTokenizer } from "@huggingface/transformers";

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
   */
  static async load({ modelId, localModelPath, modelName, chatTemplateText, meta }) {
    // Avoid serving a STALE tokenizer from transformers.js' browser cache
    // (Cache Storage `transformers-cache`). When a model is re-exported under
    // the same name, a previously cached tokenizer is otherwise returned — e.g.
    // an old control-token spelling — so the adapter control token silently
    // fails to fire. `env.useBrowserCache = false` is not reliably honored, so
    // we also evict the cache outright. Guarded for Node, where `caches` is
    // absent (and there is no browser cache to worry about).
    env.useBrowserCache = false;
    if (typeof caches !== "undefined") {
      try { await caches.delete("transformers-cache"); } catch (_) { /* best effort */ }
    }

    let tok;
    if (modelId) {
      env.allowRemoteModels = true;
      env.allowLocalModels = false;
      tok = await AutoTokenizer.from_pretrained(modelId);
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
   * The adapter expects a specific prompt framing, not the raw input: an
   * instruction line followed by the input wrapped in `<cti>…</cti>` tags
   * (mirroring the barha/granite-switch-tiny Space). Pass `instruction` to apply
   * that framing — without it the model receives an unframed prompt and returns
   * garbage even though the control token fires. With it, the 350m CTI model
   * returns a clean ATT&CK technique id (e.g. "T1105").
   *
   * @param {string} text  the CTI sentence (or raw user content if no instruction)
   * @param {object} [opts]
   * @param {string} [opts.adapterName]  adapter to activate; defaults to the
   *                                     first adapter in meta.adapter_names
   * @param {string} [opts.instruction]  instruction line; when set, the user
   *                                     content becomes
   *                                     `${instruction}\n\n<cti>\n${text}\n</cti>`
   * @param {boolean} [opts.addGenerationPrompt=true]
   * @returns {number[]} token ids
   */
  encode(text, { adapterName, instruction, addGenerationPrompt = true } = {}) {
    const adapter = adapterName || (this.meta.adapter_names && this.meta.adapter_names[0]);
    const content = instruction ? `${instruction}\n\n<cti>\n${text}\n</cti>` : text;
    const rendered = this.tok.apply_chat_template(
      [{ role: "user", content }],
      {
        tokenize: false,
        add_generation_prompt: addGenerationPrompt,
        chat_template: this.chatTemplate,
        adapter_name: adapter,
      },
    );
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
