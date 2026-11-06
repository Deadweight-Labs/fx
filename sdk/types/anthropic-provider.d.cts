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
  /** Server-side compaction (beta). The trigger is at least 50,000 input tokens. */
  compaction?: { trigger_tokens?: number; instructions?: string };
  /** Server-side clearing of old tool results (beta). */
  context_editing?: { trigger_tokens?: number; keep_tool_uses?: number; clear_at_least_tokens?: number; clear_tool_inputs?: boolean; exclude_tools?: string[] };
  /** Tools sent with `strict: true`. */
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
