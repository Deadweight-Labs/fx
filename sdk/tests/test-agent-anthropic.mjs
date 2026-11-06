#!/usr/bin/env node
// A custom provider that speaks the Anthropic Messages API: the engine's
// requests reach {baseUrl}/v1/messages through the host fetch, with the host's
// key in x-api-key, and a reply's thinking block goes back unchanged with its
// signature. No network: every request is answered here.
import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { customProviderDefinition, customProviderSpec } from "../custom-provider.js";
import { createFxEngine, supportsJspi } from "../node.js";

const backend = process.argv[2] || "native";
if (!new Set(["native", "wasm"]).has(backend)) throw new Error("usage: test-agent-anthropic.mjs [native|wasm]");
if (backend === "wasm" && !supportsJspi()) {
  console.error("Node JSPI is disabled. Run with --experimental-wasm-jspi");
  process.exit(2);
}

const scriptDir = fileURLToPath(new URL(".", import.meta.url));
const backendOptions = {
  backend,
  nativeAddon: resolve(scriptDir, "../../zig-out/lib/libfx.node"),
  ...(backend === "wasm" ? { wasm: await readFile(resolve(scriptDir, "../../zig-out/bin/fx-core.wasm")) } : {}),
};

const messagesUrl = "https://api.anthropic.com/v1/messages";
const provider = { name: "claude", protocol: "anthropic-messages", anthropic: { prefix_mismatch_behavior: "error" } };

const sse = (...events) => events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join("");
const start = { type: "message_start", message: { id: "msg_1", usage: { input_tokens: 5, cache_read_input_tokens: 100, cache_creation_input_tokens: 20, output_tokens: 1 } } };
const stop = (reason) => [{ type: "message_delta", delta: { stop_reason: reason }, usage: { output_tokens: 9 } }, { type: "message_stop" }];
const toolCall = sse(
  start,
  { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Look it up." } },
  { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "c2lnbmVk" } },
  { type: "content_block_stop", index: 0 },
  { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_1", name: "lookup", input: {} } },
  { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "{\"key\":\"alpha\"}" } },
  { type: "content_block_stop", index: 1 },
  ...stop("tool_use"),
);
const answer = sse(
  start,
  { type: "content_block_start", index: 0, content_block: { type: "text", text: "alpha is 41" } },
  { type: "content_block_stop", index: 0 },
  ...stop("end_turn"),
);

// 1. The definition the engine parses, and the key's header.
{
  const definition = customProviderDefinition(provider);
  assert.equal(definition.chatUrl, messagesUrl);
  assert.equal(definition.apiKeyHeader, "x-api-key");
  assert.deepEqual(JSON.parse(definition.json).claude.auth, { type: "none" });
  const headers = new Headers();
  customProviderSpec(definition, () => "s").authorize(headers, messagesUrl, { accessToken: "sk-ant" });
  assert.equal(headers.get("x-api-key"), "sk-ant");
  assert.equal(headers.get("authorization"), null);
  assert.throws(() => customProviderDefinition({ name: "x", baseUrl: "https://a.example", anthropic: {} }), /unknown field anthropic/);
  assert.throws(() => customProviderDefinition({ name: "x", baseUrl: "https://a.example", protocol: "responses" }), /unknown field protocol/);
  assert.throws(() => customProviderDefinition({ name: "gateway", protocol: "anthropic-messages" }), /provider.name/);
}

// 2. A tool round trip on the host's key replays the thinking block.
{
  const log = [];
  const fetch = async (input, init = {}) => {
    const url = String(input?.url ?? input);
    const body = typeof init.body === "string" ? init.body : new TextDecoder().decode(init.body);
    log.push({ url, headers: new Headers(init.headers), body });
    if (url !== messagesUrl) return new Response(`unexpected ${url}`, { status: 599 });
    const reply = log.filter((entry) => entry.url === messagesUrl).length === 1 ? toolCall : answer;
    return new Response(reply, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
  const lookup = {
    name: "lookup",
    description: "Look up a value.",
    inputSchema: { type: "object", properties: { key: { type: "string" } }, required: ["key"] },
    execute: ({ key }) => (key === "alpha" ? "41" : "unknown"),
  };
  const engine = await createFxEngine({ ...backendOptions, provider, model: "claude-haiku-5-5", effort: "low", fetch, credentials: () => ({ apiKey: "sk-ant" }), tools: [lookup] });
  try {
    const turn = engine.prompt("what is alpha?");
    let text = "";
    for await (const event of turn) if (event.type === "text_delta") text += event.delta;
    const result = await turn.result;
    assert.equal(result.stopReason, "end_turn", JSON.stringify(result));
    assert.equal(text, "alpha is 41");
  } finally {
    await engine.close();
  }
  const requests = log.filter((entry) => entry.url === messagesUrl);
  assert.equal(requests.length, 2);
  for (const request of requests) {
    assert.equal(request.headers.get("x-api-key"), "sk-ant");
    assert.equal(request.headers.get("anthropic-version"), "2023-06-01");
    assert.match(request.headers.get("anthropic-beta"), /thinking-binding-controls-2026-08-01/);
  }
  const first = JSON.parse(requests[0].body);
  assert.equal(first.model, "claude-haiku-5-5");
  assert.deepEqual(first.output_config, { effort: "low" });
  assert.equal(first.thinking.type, "adaptive");
  const second = JSON.parse(requests[1].body);
  const assistant = second.messages.find((message) => message.role === "assistant");
  assert.deepEqual(assistant.content[0], { type: "thinking", thinking: "Look it up.", signature: "c2lnbmVk" });
  assert.deepEqual(assistant.content[1], { type: "tool_use", id: "toolu_1", name: "lookup", input: { key: "alpha" } });
  const results = second.messages.at(-1).content;
  assert.equal(results[0].type, "tool_result");
  assert.equal(results[0].tool_use_id, "toolu_1");
}

console.log(`test-agent-anthropic ${backend}: ok`);
