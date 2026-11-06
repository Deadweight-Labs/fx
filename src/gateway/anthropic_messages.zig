//! A configured provider that speaks the Anthropic Messages API directly,
//! with the user's own API key. The codec is `anthropic_messages_protocol.zig`.
const std = @import("std");
const codec = @import("anthropic_messages_protocol.zig");
const chat_codec = @import("chat_completions_protocol.zig");
const client_mod = @import("client.zig");
const definitions = @import("../core/config/configured_provider.zig");
const streams = @import("../core/agent/stream_provider.zig");
const provider_set = @import("../core/gateway/provider_set.zig");
const catalog = @import("../core/gateway/model_catalog.zig");
const gateway_provider = @import("../core/gateway/gateway_provider.zig");
const model_capabilities = @import("../core/config/model_capabilities.zig");
const model_catalog_metadata = @import("../core/gateway/model_catalog_metadata.zig");
const classifier = @import("../core/permissions/auto_classifier.zig");
const gateway_step = @import("../core/agent/runtime/gateway_step.zig");
const review_messages = @import("vercel_protocol.zig");
const io = @import("../core/shared/io.zig");
const secret = @import("../core/auth/secret.zig");
const types = @import("../core/shared/types.zig");
const model_provider = @import("../core/config/model_provider.zig");
const debug_trace = @import("../core/shared/debug_trace.zig");
const sse = @import("sse.zig");
const Allocator = std.mem.Allocator;

/// The cheapest current model, for session titles.
const title_model = "claude-haiku-5-5";

/// The bundle for any configured provider, by its protocol.
pub fn configuredBundle(definition: *const definitions.Definition) provider_set.Bundle {
    return switch (definition.protocol) {
        .@"openai-chat-completions" => @import("chat_completions.zig").bundle(definition),
        .@"anthropic-messages" => bundle(definition),
    };
}

/// Caller owns the returned URL. base_url has no `/v1`.
pub fn messagesUrl(definition: *const definitions.Definition, alloc: Allocator) Allocator.Error![]u8 {
    return std.mem.concat(alloc, u8, &.{ definition.base_url, "/v1/messages" });
}

test {
    _ = @import("anthropic_messages_protocol.zig");
    _ = @import("host_anthropic_messages.zig");
    _ = @import("../core/config/anthropic_provider_options.zig");
}

/// Every callback borrows the immutable definition from the owning profile runtime.
pub fn bundle(definition: *const definitions.Definition) provider_set.Bundle {
    const context: *anyopaque = @ptrCast(@constCast(definition));
    return .{
        .title_model = title_model,
        .agent_stream = agentStream(definition),
        .model_catalog = model_catalog_provider(definition),
        .cli_model_catalog = .{ .context = context, .fetch_fn = fetch_cli_catalog },
        .permission_reviewer = .{ .context = context, .review_fn = review },
    };
}

pub fn agentStream(definition: *const definitions.Definition) streams.Provider {
    const context: *anyopaque = @ptrCast(@constCast(definition));
    return .{ .context = context, .stream_fn = stream, .build_request_fn = build, .project_replay_fn = project_replay };
}

/// The built-in Claude models and the ones the definition declares; there is
/// no network request. Borrows the definition, which must outlive the provider.
pub fn model_catalog_provider(definition: *const definitions.Definition) catalog.Provider {
    const context: *anyopaque = @ptrCast(@constCast(definition));
    return .{ .context = context, .fetch_fn = fetch_catalog, .lookup_capabilities_fn = lookup_capabilities, .provider_id = bound_identity(definition) };
}

fn definition_at(raw: ?*anyopaque) *const definitions.Definition {
    return @ptrCast(@alignCast(raw.?));
}

fn bound_identity(definition: *const definitions.Definition) model_provider.ProviderId {
    var identity = model_provider.parse(definition.id).?;
    identity.configured.binding = definition.binding_identity();
    return identity;
}

fn build(raw: ?*anyopaque, alloc: Allocator, request: streams.RequestData) ![]u8 {
    return build_for(definition_at(raw), alloc, request);
}

/// The request body for `definition`. Caller owns the result.
pub fn build_for(definition: *const definitions.Definition, alloc: Allocator, request: streams.RequestData) ![]u8 {
    const identity = bound_identity(definition);
    return codec.build_request(alloc, request, .{ .options = &definition.anthropic, .provider = &identity });
}

pub fn project_replay(alloc: Allocator, replay: ?types.ProviderReplay, calls: []const types.ToolCall, text: bool, reasoning: bool) !?types.ProviderReplay {
    const selected = try codec.project_replay(alloc, replay, calls, text, reasoning);
    if (replay != null and selected == null) debug_trace.logf("gateway", "provider_replay_omitted reason={s}", .{if (reasoning) "associated_calls_removed" else "reasoning_removed"});
    return selected;
}

/// Request headers besides content type and the key. `beta` is owned by the
/// caller and may be null.
pub fn headers(definition: *const definitions.Definition, beta: ?[]const u8, buffer: *[3]std.http.Header) []const std.http.Header {
    buffer[0] = .{ .name = "anthropic-version", .value = definition.anthropic.version };
    buffer[1] = .{ .name = "accept", .value = "text/event-stream" };
    if (beta) |value| {
        buffer[2] = .{ .name = "anthropic-beta", .value = value };
        return buffer[0..3];
    }
    return buffer[0..2];
}

fn stream(raw: ?*anyopaque, alloc: Allocator, request: streams.ModelRequest) !streams.Result {
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    const definition = definition_at(raw);
    if (request.credential.credentialSource() != .configured) return error.ConfiguredProviderCredentialRequired;
    const token = request.credential.secret();
    if (token) |value| {
        if (value.len > 16 * 1024) return error.InvalidConfiguredProviderCredential;
        for (value) |byte| if (byte <= 0x20 or byte >= 0x7f) return error.InvalidConfiguredProviderCredential;
    }
    switch (definition.auth) {
        .none => if (token != null) return error.UnexpectedConfiguredProviderCredential,
        .bearer => if (token == null) return error.MissingConfiguredProviderCredential,
    }
    const payload = request.prepared_request_body orelse try build(raw, alloc, request.data());
    defer if (request.prepared_request_body == null) alloc.free(payload);
    return post(alloc, definition, request, token, payload) catch |err| {
        request.attempt_evidence.network_failure = client_mod.networkFailureEvidence(err, request.delivery.load());
        if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
        if (request.deadline) |deadline| if (expired(deadline)) return error.Timeout;
        return err;
    };
}

fn expired(deadline: std.Io.Clock.Timestamp) bool {
    return !std.Io.Clock.Timestamp.compare(std.Io.Clock.Timestamp.now(io.getIo(), .awake), .lt, deadline);
}

fn phase_deadline(milliseconds: i64, caller: ?std.Io.Clock.Timestamp) std.Io.Clock.Timestamp {
    const phase = std.Io.Clock.Timestamp.fromNow(io.getIo(), .{ .clock = .awake, .raw = .fromMilliseconds(milliseconds) });
    if (caller) |deadline| if (std.Io.Clock.Timestamp.compare(deadline, .lt, phase)) return deadline;
    return phase;
}

fn post(alloc: Allocator, definition: *const definitions.Definition, request: streams.ModelRequest, token: ?[]const u8, payload: []const u8) !streams.Result {
    const url = try messagesUrl(definition, alloc);
    defer alloc.free(url);
    var uri = try std.Uri.parse(url);
    uri.scheme = if (std.ascii.eqlIgnoreCase(uri.scheme, "https")) "https" else if (std.ascii.eqlIgnoreCase(uri.scheme, "http")) "http" else return error.UnsupportedUriScheme;
    const beta = try codec.betaHeader(alloc, &definition.anthropic, request.model);
    defer if (beta) |value| alloc.free(value);
    var header_buffer: [4]std.http.Header = undefined;
    var base: [3]std.http.Header = undefined;
    const fixed = headers(definition, beta, &base);
    @memcpy(header_buffer[0..fixed.len], fixed);
    var count = fixed.len;
    const api_key = if (token) |value| try alloc.dupe(u8, value) else null;
    defer if (api_key) |value| secret.zeroAndFree(alloc, value);
    if (api_key) |value| {
        header_buffer[count] = .{ .name = "x-api-key", .value = value };
        count += 1;
    }
    try request.admission.admit();
    var opened = try client_mod.openBoundedPost(request.cancel_flag, phase_deadline(30_000, request.deadline), .{
        .uri = uri,
        .authorization = null,
        .extra_headers = header_buffer[0..count],
    });
    defer opened.deinit();
    const http = &opened.request;
    var watch: client_mod.ConnectionWatch = .init(.{
        .timeout_ms = 120_000,
        .stall_timeout_ms = client_mod.ConnectionWatch.bodyStallTimeoutMs(request.deadline),
    });
    defer watch.stop();
    if (http.connection) |connection| try watch.start(request.cancel_flag, request.deadline, connection.stream_writer.stream);
    if (watch.armHead()) |err| return err;
    http.transfer_encoding = .{ .content_length = payload.len };
    var buffer: [8192]u8 = undefined;
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    request.delivery.markPossiblySent();
    var body = try http.sendBodyUnflushed(&buffer);
    try body.writer.writeAll(payload);
    try body.end();
    if (http.connection) |connection| try connection.flush();
    var response = http.receiveHead(&.{}) catch |err| return watch.failure(err);
    if (watch.commitHead()) |err| return err;
    if (response.head.status != .ok) {
        watch.stop();
        watch = .init(.{ .patient = true, .stall_timeout_ms = std.math.maxInt(i64) });
        if (http.connection) |connection| try watch.start(request.cancel_flag, phase_deadline(30_000, request.deadline), connection.stream_writer.stream);
    }
    var retry_after: ?u64 = null;
    var headers_iterator = response.head.iterateHeaders();
    while (headers_iterator.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "retry-after")) {
            retry_after = std.fmt.parseUnsigned(u64, std.mem.trim(u8, header.value, " \t"), 10) catch null;
        } else if (std.ascii.eqlIgnoreCase(header.name, "request-id")) {
            debug_trace.logf("gateway", "anthropic_request_id {s} status={d}", .{ header.value[0..@min(header.value.len, 128)], @backingInt(response.head.status) });
        }
    }
    var transfer: [64 * 1024]u8 = undefined;
    const reader = response.reader(&transfer);
    if (response.head.status != .ok) {
        var detail = reader.allocRemaining(alloc, .limited(64 * 1024)) catch |err| switch (err) {
            error.StreamTooLong => try alloc.dupe(u8, "Provider error response exceeded the local limit"),
            else => return err,
        };
        errdefer alloc.free(detail);
        if (token) |value| {
            const redacted = try chat_codec.redact_error_detail(alloc, detail, value);
            alloc.free(detail);
            detail = redacted;
        }
        return .{ .failed = .{ .kind = failureKind(alloc, response.head.status, detail), .http_status = response.head.status, .detail = detail, .retry_after_seconds = retry_after, .ownership = .owned } };
    }
    var result = consume(alloc, definition, reader, request, watch.progress()) catch |err| return watch.failure(err);
    if (watch.finish()) |err| {
        result.deinit(alloc);
        return err;
    }
    return result;
}

/// 529 is Anthropic's "overloaded", which fx retries like a 503.
pub fn failureKind(alloc: Allocator, status: std.http.Status, detail: []const u8) streams.FailureKind {
    if (@backingInt(status) == 529) return .unavailable;
    return client_mod.httpFailureKind(alloc, .fromHttpStatus(status), detail);
}

/// Parses a successful streamed reply from `reader`, telling `progress` about
/// each event when it is set.
pub fn consume(alloc: Allocator, definition: *const definitions.Definition, reader: *std.Io.Reader, request: streams.ModelRequest, progress: ?sse.Progress) !streams.Result {
    var limits: codec.Limits = .{};
    if (request.content_capture_limit) |limit| limits.content_bytes = @min(limit, limits.content_bytes);
    const identity = bound_identity(definition);
    return codec.consume_stream(alloc, reader, request.data(), .{ .options = &definition.anthropic, .provider = &identity }, limits, request.events, request.cancel_flag, progress);
}

// ---------------------------------------------------------------------------
// Models

const effort_values = blk: {
    var values: [codec.efforts.len]types.ReasoningEffort = undefined;
    for (codec.efforts, 0..) |name, index| values[index] = types.ReasoningEffort.literal(name);
    break :blk values;
};

/// The returned entry borrows `id` and the static effort list; `appendEntry`
/// makes owned copies.
fn metadata_entry(definition: *const definitions.Definition, id: []const u8) ?catalog.ModelCatalogEntry {
    const declared = definition.model(id);
    const builtin = codec.builtinModel(id);
    if (declared == null and builtin == null) return null;
    const known = builtin != null;
    const vision = if (declared) |metadata| metadata.supports_vision orelse known else true;
    return .{
        .id = @constCast(id),
        .model_type = @constCast("language"),
        .has_tool_use = if (declared) |metadata| metadata.supports_tool_use orelse known else true,
        .has_reasoning = known,
        .reasoning_efforts = if (known) .fromOwnedSlice(@constCast(effort_values[0..])) else .empty,
        .has_vision = vision,
        .has_file_input = vision,
        .has_explicit_caching = true,
        .context_window = if (declared) |metadata| metadata.context_window orelse (if (builtin) |info| info.context_window else 0) else builtin.?.context_window,
        .max_tokens = if (declared) |metadata| metadata.max_output_tokens orelse (if (builtin) |info| info.max_output_tokens else 0) else builtin.?.max_output_tokens,
    };
}

fn lookup_capabilities(raw: ?*anyopaque, model: []const u8) model_capabilities.Capabilities {
    const entry = metadata_entry(definition_at(raw), model) orelse return .{};
    var capabilities = model_capabilities.mergeCapabilities(.{}, model_catalog_metadata.fromCatalogEntry(entry));
    capabilities.parallel_tool_calls = true;
    return capabilities;
}

fn fetch_catalog(raw: ?*anyopaque, alloc: Allocator, input: catalog.FetchInput) Allocator.Error!catalog.ProviderResult {
    if (input.cancel_flag) |flag| if (flag.load(.seq_cst)) return .{ .failure = .{ .category = .cancellation } };
    const definition = definition_at(raw);
    var entries: std.ArrayList(catalog.ModelCatalogEntry) = .empty;
    errdefer catalog.freeModelCatalog(alloc, &entries);
    for (codec.builtin_models) |info| try appendEntry(alloc, &entries, metadata_entry(definition, info.id).?);
    for (definition.model_metadata) |metadata| {
        if (codec.builtinModel(metadata.id) != null) continue;
        try appendEntry(alloc, &entries, metadata_entry(definition, metadata.id).?);
    }
    return .{ .catalog = entries };
}

fn appendEntry(alloc: Allocator, entries: *std.ArrayList(catalog.ModelCatalogEntry), source: catalog.ModelCatalogEntry) Allocator.Error!void {
    var entry = source;
    entry.id = try alloc.dupe(u8, entry.id);
    errdefer alloc.free(entry.id);
    entry.model_type = try alloc.dupe(u8, entry.model_type);
    errdefer alloc.free(entry.model_type);
    entry.reasoning_efforts = .empty;
    try entry.reasoning_efforts.appendSlice(alloc, source.reasoning_efforts.items);
    errdefer entry.reasoning_efforts.deinit(alloc);
    try entries.append(alloc, entry);
}

fn fetch_cli_catalog(raw: ?*anyopaque, alloc: Allocator, input: gateway_provider.CliModelCatalogInput) gateway_provider.CliModelCatalogResult {
    const provenance = catalog.Provenance{ .access = catalog.AccessMetadata.init(input.access) };
    const result = fetch_catalog(raw, alloc, .{ .access = input.access, .endpoint = input.endpoint, .cancel_flag = input.cancel_flag }) catch
        return .{ .failure = .{ .access = provenance.access, .anonymous_fallback_used = false, .failure = .{ .category = .resource_exhausted } } };
    switch (result) {
        .failure => |failure| return .{ .failure = .{ .access = provenance.access, .anonymous_fallback_used = false, .failure = failure } },
        .catalog => |value| {
            var entries = value;
            defer catalog.freeModelCatalog(alloc, &entries);
            const ids = catalog.projectModelIds(alloc, entries.items) catch return .{ .failure = .{ .access = provenance.access, .anonymous_fallback_used = false, .failure = .{ .category = .resource_exhausted } } };
            return .{ .loaded = .{ .ids = ids, .provenance = provenance } };
        },
    }
}

// ---------------------------------------------------------------------------
// Permission reviewer

const Review = struct { definition: *const definitions.Definition, input: classifier.ProviderInput };
fn review(raw: ?*anyopaque, alloc: Allocator, input: classifier.ProviderInput, request: classifier.ReviewRequest) !classifier.ParseOutcome {
    var state = Review{ .definition = definition_at(raw), .input = input };
    return classifier.Reviewer.withTransportModel(.{ .context = &state, .build_fn = build_review, .send_fn = send_review }, input.cancel_flag, classifier.Reviewer.default_timeout_ms, state.definition.reviewer_model orelse request.review_turn.model).review(alloc, request);
}
fn build_review(raw: *anyopaque, alloc: Allocator, model: []const u8, _: []const u8, instructions: []const types.ChatMessage, messages: []const types.ChatMessage, target_id: []const u8, deadline: std.Io.Clock.Timestamp, cancel: *std.atomic.Value(bool)) ![]u8 {
    const state: *Review = @ptrCast(@alignCast(raw));
    const expanded = try review_messages.expandPendingToolReviewMessages(alloc, messages, target_id, deadline, cancel);
    defer alloc.free(expanded);
    // Each review's prompt carries the reviewed action inside its one system
    // block, so no later request shares its prefix; caching would only pay
    // the write surcharge.
    var options = state.definition.anthropic;
    options.prompt_caching = false;
    const identity = bound_identity(state.definition);
    return codec.build_request(alloc, .{ .model = model, .instructions = instructions, .messages = expanded, .tools = .{ .additional_functions = &.{classifier.function_schema} }, .tool_choice = .required, .provider_options = .{ .reasoning = types.ReasoningEffort.literal("low") }, .max_output_tokens = 4096 }, .{ .options = &options, .provider = &identity });
}
fn ignore_event(_: *anyopaque, _: streams.Event) void {}
fn free_result(raw: *anyopaque, alloc: Allocator) void {
    const result: *streams.Result = @ptrCast(@alignCast(raw));
    result.deinit(alloc);
    alloc.destroy(result);
}
fn send_review(raw: *anyopaque, alloc: Allocator, model: []const u8, payload: []const u8, deadline: std.Io.Clock.Timestamp, cancel: *std.atomic.Value(bool)) !classifier.TransportOutcome {
    const state: *Review = @ptrCast(@alignCast(raw));
    var delivery: streams.DeliveryCertainty = .init();
    var evidence: streams.AttemptEvidence = .{};
    var event_context: u8 = 0;
    var result = gateway_step.streamModelCompletion(agentStream(state.definition), alloc, .{
        .credential = .{ .direct = .{ .secret_bytes = state.input.credential, .source = state.input.credential_source } },
        .model = model,
        .retry_count = 1,
        .messages = &.{},
        .tools = .{ .additional_functions = &.{classifier.function_schema} },
        .tool_choice = .required,
        .provider_options = .{ .reasoning = types.ReasoningEffort.literal("low") },
        .prepared_request_body = payload,
        .trace_ctx = .{},
        .content_capture_limit = 16 * 1024,
        .deadline = deadline,
        .delivery = &delivery,
        .attempt_evidence = &evidence,
        .events = .{ .context = &event_context, .emit_fn = ignore_event },
        .cancel_flag = cancel,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Cancelled => return .cancelled,
        error.Timeout => return .timed_out,
        error.RequiredToolMissing => return .{ .completion = .{ .completion = .{} } },
        else => return .permanent_failure,
    };
    errdefer result.deinit(alloc);
    const cut_off = result == .completed and result.completed.completion.finish_reason == .length;
    if (result == .failed or cut_off) {
        result.deinit(alloc);
        return .permanent_failure;
    }
    const owned = try alloc.create(streams.Result);
    owned.* = result;
    return .{ .completion = .{ .completion = owned.completed.completion, .context = owned, .deinit_fn = free_result } };
}

test "anthropic catalog lists built-in models with effort levels and keeps declared limits" {
    const alloc = std.testing.allocator;
    var registry = try definitions.Registry.parse_json(alloc,
        \\{"claude":{"protocol":"anthropic-messages","model_metadata":{"claude-haiku-5-5":{"context_window":200000},"claude-custom":{"max_output_tokens":1000,"supports_vision":false}}}}
    );
    defer registry.deinit(alloc);
    const definition = registry.get("claude").?;
    try std.testing.expectEqualStrings("https://api.anthropic.com", definition.base_url);
    try std.testing.expectEqualStrings("ANTHROPIC_API_KEY", definition.auth.bearer);
    const provider = bundle(definition).model_catalog.?;
    var fetched = try provider.fetch(alloc, .{ .endpoint = "unused" });
    defer catalog.freeModelCatalog(alloc, &fetched.catalog);
    try std.testing.expectEqual(@as(usize, 4), fetched.catalog.items.len);
    try std.testing.expectEqual(codec.efforts.len, fetched.catalog.items[0].reasoning_efforts.items.len);
    for (fetched.catalog.items) |entry| try model_capabilities.expectCapabilitiesEqualForTest(blk: {
        var expected = model_capabilities.mergeCapabilities(.{}, model_catalog_metadata.fromCatalogEntry(entry));
        expected.parallel_tool_calls = true;
        break :blk expected;
    }, provider.lookupCapabilities(entry.id).?);
    const haiku = provider.lookupCapabilities("claude-haiku-5-5").?;
    try std.testing.expectEqual(@as(?u32, 200000), haiku.context_window);
    try std.testing.expectEqual(@as(?u32, 128000), haiku.max_output_tokens);
    try std.testing.expect(haiku.supports_reasoning and haiku.supports_vision and haiku.supports_tool_use);
    try std.testing.expectEqual(codec.efforts.len, haiku.reasoning_efforts.len);
    const custom = provider.lookupCapabilities("claude-custom").?;
    try std.testing.expect(!custom.supports_reasoning and !custom.supports_vision);
    try std.testing.expectEqual(@as(?u32, 1000), custom.max_output_tokens);
    try model_capabilities.expectCapabilitiesEqualForTest(.{}, provider.lookupCapabilities("unknown-model").?);
}

test "anthropic headers carry the version and only the betas the settings need" {
    const alloc = std.testing.allocator;
    var registry = try definitions.Registry.parse_json(alloc,
        \\{"claude":{"protocol":"anthropic-messages","anthropic":{"version":"2026-10-21","betas":["extra-1"],"compaction":{},"context_editing":{}}}}
    );
    defer registry.deinit(alloc);
    const definition = registry.get("claude").?;
    const beta = (try codec.betaHeader(alloc, &definition.anthropic, "claude-haiku-5-5")).?;
    defer alloc.free(beta);
    try std.testing.expectEqualStrings("extra-1," ++ codec.thinking_binding_beta ++ "," ++ codec.compaction_beta ++ "," ++ codec.context_management_beta, beta);
    var buffer: [3]std.http.Header = undefined;
    const sent = headers(definition, beta, &buffer);
    try std.testing.expectEqualStrings("2026-10-21", sent[0].value);
    try std.testing.expectEqualStrings(beta, sent[2].value);
    const plain = try codec.betaHeader(alloc, &(definitions.Definition{ .id = "x", .protocol = .@"anthropic-messages", .base_url = "", .auth = .none }).anthropic, "other-model");
    try std.testing.expect(plain == null);
    try std.testing.expectEqual(streams.FailureKind.unavailable, failureKind(alloc, @fromBackingInt(@intCast(529)), "{}"));
}
