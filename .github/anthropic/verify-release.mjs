#!/usr/bin/env node
// Checks a libfx installed from one of this fork's GitHub releases: the
// version is the release's, the backend loads, and a tool round trip on an
// anthropic-messages provider replays the thinking block with its signature.
// No network: every request is answered here.
//
//   node verify-release.mjs <version> native|wasm
// Run it from the directory whose node_modules holds the installed libfx.
import { strict as assert } from "node:assert";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

const [version, backend] = process.argv.slice(2);
if (!version || !["native", "wasm"].includes(backend)) throw new Error("usage: verify-release.mjs <version> native|wasm");

const { createFxEngine, getBackendInfo } = await import("libfx");
const installed = JSON.parse(await readFile(resolve("node_modules/libfx/package.json"), "utf8"));
assert.equal(installed.version, version);

const info = await getBackendInfo({ backend: backend === "native" ? "native" : "wasm" });
assert.equal(info.backend, backend === "native" ? "native" : "wasm-jspi", JSON.stringify(info));

const messagesUrl = "https://api.anthropic.com/v1/messages";
const sse = (...events) => events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join("");
const start = { type: "message_start", message: { id: "msg_1", usage: { input_tokens: 5, output_tokens: 1 } } };
const stop = (reason) => [{ type: "message_delta", delta: { stop_reason: reason }, usage: { output_tokens: 9 } }, { type: "message_stop" }];
const replies = [
  sse(
    start,
    { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } },
    { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Look it up." } },
    { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "c2lnbmVk" } },
    { type: "content_block_stop", index: 0 },
    { type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "toolu_1", name: "lookup", input: {} } },
    { type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: "{\"key\":\"alpha\"}" } },
    { type: "content_block_stop", index: 1 },
    ...stop("tool_use"),
  ),
  sse(
    start,
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "alpha is 41" } },
    { type: "content_block_stop", index: 0 },
    ...stop("end_turn"),
  ),
];

const requests = [];
const fetch = async (input, init = {}) => {
  const url = String(input?.url ?? input);
  if (url !== messagesUrl) return new Response(`unexpected ${url}`, { status: 599 });
  const body = typeof init.body === "string" ? init.body : new TextDecoder().decode(init.body);
  requests.push({ headers: new Headers(init.headers), body: JSON.parse(body) });
  return new Response(replies[requests.length - 1] ?? "", { status: 200, headers: { "content-type": "text/event-stream" } });
};

const engine = await createFxEngine({
  backend,
  provider: { name: "claude", protocol: "anthropic-messages" },
  model: "claude-haiku-5-5",
  effort: "low",
  fetch,
  credentials: () => ({ apiKey: "sk-ant-test" }),
  tools: [{
    name: "lookup",
    description: "Look up a value.",
    inputSchema: { type: "object", properties: { key: { type: "string" } }, required: ["key"] },
    execute: ({ key }) => (key === "alpha" ? "41" : "unknown"),
  }],
});
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

assert.equal(requests.length, 2);
for (const { headers } of requests) {
  assert.equal(headers.get("x-api-key"), "sk-ant-test");
  assert.equal(headers.get("anthropic-version"), "2023-06-01");
}
const assistant = requests[1].body.messages.find((message) => message.role === "assistant");
assert.deepEqual(assistant.content[0], { type: "thinking", thinking: "Look it up.", signature: "c2lnbmVk" });
assert.deepEqual(assistant.content[1], { type: "tool_use", id: "toolu_1", name: "lookup", input: { key: "alpha" } });

console.log(`libfx@${version} from the release: ${info.backend} backend, Anthropic round trip ok`);
