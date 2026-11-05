// A custom provider for libfx agents: any server that speaks OpenAI Chat
// Completions, such as a local model server or a router, or the Anthropic
// Messages API with the app's own key. The engine sends its
// requests through the host's fetch without credentials; when the host has an
// API key, this module adds it at the fetch edge, to that provider's chat URL
// only. The key never enters the engine or a session's saved records.

import { tokenString } from "./subscription.js";

const encoder = new TextEncoder();
const reservedNames = new Set(["gateway", "codex", "grok"]);
const namePattern = /^[A-Za-z][A-Za-z0-9_-]{0,63}$/;
const loopbackHosts = new Set(["localhost", "127.0.0.1", "[::1]"]);
const providerFields = new Set(["name", "protocol", "baseUrl", "models", "anthropic"]);
const protocols = new Set(["openai-chat-completions", "anthropic-messages"]);
const anthropicBaseUrl = "https://api.anthropic.com";
// libfx option names to the CLI's settings names, which the engine parses.
const modelFields = {
  contextWindow: "context_window",
  maxOutputTokens: "max_output_tokens",
  supportsToolUse: "supports_tool_use",
  supportsVision: "supports_vision",
};

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function chatUrlOf(baseUrl, path = "/chat/completions") {
  if (typeof baseUrl !== "string") throw new TypeError("provider.baseUrl must be a string");
  let url;
  try {
    url = new URL(`${baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl}${path}`);
  } catch {
    throw new TypeError("provider.baseUrl must be an absolute URL");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new TypeError("provider.baseUrl must not carry credentials, a query or a fragment");
  }
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopbackHosts.has(url.hostname))) {
    throw new TypeError("provider.baseUrl must use https, or http on localhost");
  }
  // The same form the fetch edge compares requests in.
  return `${url.origin}${url.pathname}`;
}

// The engine's limits (configured_provider.zig), checked here so both engines
// report a mistake the same way.
const maxModels = 256;
const maxModelIdBytes = 1024;
const maxLimit = 2 ** 32 - 1;

function modelMetadata(models) {
  if (!isPlainObject(models)) throw new TypeError("provider.models must be an object keyed by model id");
  const entries = Object.entries(models);
  if (entries.length > maxModels) throw new RangeError(`provider.models lists more than ${maxModels} models`);
  const metadata = {};
  for (const [id, model] of entries) {
    if (id.length === 0 || id.trim() !== id || /[\u0000-\u001f\u007f]/.test(id) || encoder.encode(id).length > maxModelIdBytes) {
      throw new TypeError(`provider.models has an invalid model id "${id}"`);
    }
    if (!isPlainObject(model)) throw new TypeError(`provider.models["${id}"] must be an object`);
    const entry = {};
    for (const [field, value] of Object.entries(model)) {
      const name = modelFields[field];
      if (name === undefined) throw new TypeError(`provider.models["${id}"] has unknown field ${field}`);
      const limit = field === "contextWindow" || field === "maxOutputTokens";
      if (limit ? !(Number.isInteger(value) && value > 0 && value <= maxLimit) : typeof value !== "boolean") {
        throw new TypeError(`provider.models["${id}"].${field} must be ${limit ? "a positive integer" : "a boolean"}`);
      }
      entry[name] = value;
    }
    if (model.contextWindow !== undefined && model.maxOutputTokens !== undefined && model.maxOutputTokens >= model.contextWindow) {
      throw new RangeError(`provider.models["${id}"].maxOutputTokens must be less than contextWindow`);
    }
    metadata[id] = entry;
  }
  return metadata;
}

/**
 * Checks a custom provider `{ name, protocol?, baseUrl, models?, anthropic? }`
 * and returns its name, the URL its key may go to, and the settings-style JSON
 * the engine parses. The engine applies the CLI's complete rules; these checks
 * only catch mistakes early with a clearer message.
 */
export function customProviderDefinition(provider) {
  if (!isPlainObject(provider)) throw new TypeError("a custom provider must be an object { name, baseUrl, models? }");
  for (const field of Object.keys(provider)) {
    if (!providerFields.has(field)) throw new TypeError(`a custom provider has unknown field ${field}`);
  }
  const { name, protocol = "openai-chat-completions", models, anthropic } = provider;
  if (typeof name !== "string" || !namePattern.test(name) || reservedNames.has(name.toLowerCase())) {
    throw new TypeError("provider.name must be a letter followed by letters, digits, _ or -, at most 64 characters, and not gateway, codex or grok");
  }
  if (!protocols.has(protocol)) throw new TypeError(`provider.protocol must be one of ${[...protocols].join(", ")}`);
  const isAnthropic = protocol === "anthropic-messages";
  if (anthropic !== undefined && (!isAnthropic || !isPlainObject(anthropic))) {
    throw new TypeError("provider.anthropic is an object of settings for the anthropic-messages protocol only");
  }
  const baseUrl = provider.baseUrl ?? (isAnthropic ? anthropicBaseUrl : undefined);
  const chatUrl = chatUrlOf(baseUrl, isAnthropic ? "/v1/messages" : "/chat/completions");
  const definition = {
    protocol,
    base_url: baseUrl,
    // The host adds the key, so the engine's copy has none.
    auth: { type: "none" },
    ...(models === undefined ? {} : { model_metadata: modelMetadata(models) }),
    ...(anthropic === undefined ? {} : { anthropic }),
  };
  return { name, chatUrl, apiKeyHeader: isAnthropic ? "x-api-key" : "authorization", json: JSON.stringify({ [name]: definition }) };
}

/**
 * The custom provider's spec for `createSubscriptionAuthorizer()`: the host's
 * API key as a bearer token, on the provider's chat URL only. `sessionId()`
 * names the agent's session. A key has no refresh token, so it is never
 * refreshed, and a rejected request returns as it is.
 */
export function customProviderSpec({ name, chatUrl, apiKeyHeader = "authorization" }, sessionId) {
  return {
    provider: name,
    matches: (endpoint) => endpoint === chatUrl,
    sessionId: () => sessionId(),
    normalize(value) {
      if (!isPlainObject(value)) throw new TypeError("credentials() must return { apiKey } for a custom provider");
      return { accessToken: tokenString(value.apiKey, "apiKey", true, `provider "${name}"`) };
    },
    authorize(headers, _endpoint, credentials) {
      if (apiKeyHeader === "x-api-key") headers.set("x-api-key", credentials.accessToken);
      else headers.set("authorization", `Bearer ${credentials.accessToken}`);
    },
  };
}
