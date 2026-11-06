import type { FxCustomProviderModel } from "./libfx.cjs";

/** Settings of the `anthropic-messages` protocol, with the CLI's settings names. */
export interface FxAnthropicProviderSettings {
  /** The `anthropic-version` header. Defaults to `2023-06-01`. */
  version?: string;
  /** Extra `anthropic-beta` values. */
  betas?: string[];
  /** Automatic prompt caching. Defaults to true. */
  prompt_caching?: boolean;
  cache_ttl?: "5m" | "1h";
  /** Defaults to `omitted`, which keeps requests small; `summarized` also returns readable reasoning. */
  thinking_display?: "summarized" | "omitted";
  /** What the API does with a thinking block whose history changed. Defaults to `drop_block`. */
  prefix_mismatch_behavior?: "drop_block" | "error" | "unset";
  /**
   * The request's `context_management` object in the API's own terms, sent as it is: server-side
   * compaction (`compact_20260112`) and context editing (`clear_tool_uses_20250919`, ...). The
   * matching beta headers are added. Tool names in it use the API's names, with `__` for each dot.
   */
  context_management?: { edits: Array<Record<string, unknown>> };
  /** Tools sent with `strict: true`: fx's own tools, and others whose schema already sets `additionalProperties: false`. */
  strict_tools?: string[];
}

/** The Anthropic Messages API with the app's own key. */
export interface FxAnthropicProviderDefinition {
  /** Names the provider to `credentials()`: a letter, then letters, digits, `_` or `-`, at most 64 characters. Not `gateway`, `codex` or `grok`. */
  name: string;
  protocol: "anthropic-messages";
  /** Requests go to `{baseUrl}/v1/messages`. Defaults to `https://api.anthropic.com`. HTTPS, or HTTP on localhost. */
  baseUrl?: string;
  /** The current Claude models are known without it. */
  models?: Record<string, FxCustomProviderModel>;
  anthropic?: FxAnthropicProviderSettings;
}
