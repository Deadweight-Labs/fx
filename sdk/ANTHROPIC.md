# Anthropic Messages provider

A custom provider can speak the Anthropic Messages API, with your own Anthropic API key:

```js
const claude = { name: "claude", protocol: "anthropic-messages" };

const agent = createFxAgent({
  provider: claude,
  model: "claude-sonnet-5-5",
  credentials: async () => ({ apiKey: process.env.ANTHROPIC_API_KEY }),
});
```

Requests go to `{baseUrl}/v1/messages`, where `baseUrl` defaults to `https://api.anthropic.com`, and the key travels in the `x-api-key` header. libfx knows `claude-opus-5-5`, `claude-sonnet-5-5` and `claude-haiku-5-5` without `models`. Thinking blocks go back to the API unchanged, with their signatures.

`anthropic` takes the CLI's settings for this protocol, such as `{ cache_ttl: "1h" }` or `{ prompt_caching: false }`. `FxAnthropicProviderSettings` in `types/anthropic-provider.d.cts` lists them. Server-side compaction and context editing go in `context_management`, in the API's own terms, and libfx adds their beta headers:

```js
anthropic: { context_management: { edits: [{ type: "compact_20260112", trigger: { type: "input_tokens", value: 150000 } }] } }
```

In the fx CLI, the same provider is a `providers` entry in `~/.fx/settings.json`:

```json
{ "providers": { "claude": { "protocol": "anthropic-messages", "reviewer_model": "claude-haiku-5-5", "anthropic": { "cache_ttl": "1h" } } } }
```

Without `base_url` and `auth`, it uses `https://api.anthropic.com` and the key in `ANTHROPIC_API_KEY`.

## Installing this fork

npm's `libfx` is upstream's and has no Anthropic transport. Install from a [release of this fork](https://github.com/Deadweight-Labs/fx/releases) instead, which also brings the native addon for Linux (x64, arm64) or macOS from that release:

```sh
npm install https://github.com/Deadweight-Labs/fx/releases/download/v0.0.16-anthropic.1/libfx-0.0.16-anthropic.1.tgz
```

A release is made from the Actions tab ("Release (Anthropic)"), which tags main as `v<fx version>-anthropic.<n>`, or by pushing such a tag. The fx CLI archives of a release never update themselves: `fx upgrade` would install upstream fx.
