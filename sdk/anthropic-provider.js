// A custom provider with `protocol: "anthropic-messages"`: the Anthropic
// Messages API with the app's own key. custom-provider.js hands such a
// provider to this module; the engine builds the requests, and libfx adds the
// key in the x-api-key header at the fetch edge, to the provider's messages URL
// only.

import { tokenString } from "./subscription.js";

const providerFields = new Set(["name", "protocol", "baseUrl", "models", "anthropic"]);
const loopbackHosts = new Set(["localhost", "127.0.0.1", "[::1]"]);
const defaultBaseUrl = "https://api.anthropic.com";

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function messagesUrlOf(baseUrl) {
  if (typeof baseUrl !== "string") throw new TypeError("provider.baseUrl must be a string");
  let url;
  try {
    url = new URL(`${baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl}/v1/messages`);
  } catch {
    throw new TypeError("provider.baseUrl must be an absolute URL");
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new TypeError("provider.baseUrl must not carry credentials, a query or a fragment");
  }
  if (url.protocol !== "https:" && !(url.protocol === "http:" && loopbackHosts.has(url.hostname))) {
    throw new TypeError("provider.baseUrl must use https, or http on localhost");
  }
  return `${url.origin}${url.pathname}`;
}

/**
 * The definition of an Anthropic provider, in the shape customProviderDefinition
 * returns. `checks` are custom-provider.js's own name and model checks.
 */
export function anthropicProviderDefinition(provider, { checkName, modelMetadata }) {
  for (const field of Object.keys(provider)) {
    if (!providerFields.has(field)) throw new TypeError(`a custom provider has unknown field ${field}`);
  }
  const { name, models, anthropic } = provider;
  checkName(name);
  if (anthropic !== undefined && !isPlainObject(anthropic)) throw new TypeError("provider.anthropic must be an object of settings");
  const baseUrl = provider.baseUrl ?? defaultBaseUrl;
  const definition = {
    protocol: "anthropic-messages",
    base_url: baseUrl,
    // The host adds the key, so the engine's copy has none.
    auth: { type: "none" },
    ...(models === undefined ? {} : { model_metadata: modelMetadata(models) }),
    ...(anthropic === undefined ? {} : { anthropic }),
  };
  return { name, chatUrl: messagesUrlOf(baseUrl), apiKeyHeader: "x-api-key", json: JSON.stringify({ [name]: definition }) };
}

/** customProviderSpec for an Anthropic provider: the key goes in x-api-key. */
export function anthropicProviderSpec({ name, chatUrl }, sessionId) {
  return {
    provider: name,
    matches: (endpoint) => endpoint === chatUrl,
    sessionId: () => sessionId(),
    normalize(value) {
      if (!isPlainObject(value)) throw new TypeError("credentials() must return { apiKey } for a custom provider");
      return { accessToken: tokenString(value.apiKey, "apiKey", true, `provider "${name}"`) };
    },
    authorize(headers, _endpoint, credentials) {
      headers.set("x-api-key", credentials.accessToken);
    },
  };
}
