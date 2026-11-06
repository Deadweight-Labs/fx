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

`anthropic` takes the CLI's settings for this protocol, such as `{ cache_ttl: "1h" }`, `{ compaction: { trigger_tokens: 150000 } }` for server-side compaction, or `{ prompt_caching: false }`. `FxAnthropicProviderSettings` in `types/anthropic-provider.d.cts` lists them.

In the fx CLI, the same provider is a `providers` entry in `~/.fx/settings.json`:

```json
{ "providers": { "claude": { "protocol": "anthropic-messages", "reviewer_model": "claude-haiku-5-5", "anthropic": { "cache_ttl": "1h" } } } }
```

Without `base_url` and `auth`, it uses `https://api.anthropic.com` and the key in `ANTHROPIC_API_KEY`.
