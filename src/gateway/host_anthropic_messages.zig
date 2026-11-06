//! A configured Anthropic Messages provider sent through a host transport,
//! for libfx, whose network access belongs to the host. The request, models
//! and reply parsing are the CLI's own (`anthropic_messages.zig`); the host
//! adds the API key.
const std = @import("std");
const stream_provider = @import("../core/agent/stream_provider.zig");
const provider_set = @import("../core/gateway/provider_set.zig");
const definitions = @import("../core/config/configured_provider.zig");
const anthropic = @import("anthropic_messages.zig");
const codec = @import("anthropic_messages_protocol.zig");
const host_chat_completions = @import("host_chat_completions.zig");
const host_stream = @import("host_stream_provider.zig");

const Allocator = std.mem.Allocator;
const Context = host_chat_completions.Context;

/// The bundle for a context whose one provider speaks `anthropic-messages`.
pub fn bundle(context: *Context) provider_set.Bundle {
    const own = context.definition();
    return .{
        .title_model = "claude-haiku-5-5",
        .agent_stream = .{
            .context = context,
            .stream_fn = stream,
            .build_request_fn = buildRequest,
            .project_replay_fn = anthropic.project_replay,
        },
        .model_catalog = anthropic.model_catalog_provider(own),
    };
}

fn buildRequest(raw: ?*anyopaque, alloc: Allocator, request: stream_provider.RequestData) anyerror![]u8 {
    const context: *Context = @ptrCast(@alignCast(raw.?));
    return anthropic.build_for(context.definition(), alloc, request);
}

fn stream(raw: ?*anyopaque, alloc: Allocator, request: stream_provider.ModelRequest) anyerror!stream_provider.Result {
    if (host_stream.deadlineExpired(request.deadline)) return error.Timeout;
    if (request.cancel_flag.load(.seq_cst)) return stream_provider.failResult(error.Cancelled);
    // The engine never holds this provider's key; the host adds it.
    if (request.credential.credentialSource() != .host_managed) {
        return stream_provider.failResult(error.ConfiguredProviderCredentialRequired);
    }
    const context: *Context = @ptrCast(@alignCast(raw.?));
    const transport = context.transport;
    const definition = context.definition();
    const payload = request.prepared_request_body orelse
        try anthropic.build_for(definition, alloc, request.data());
    defer if (request.prepared_request_body == null) alloc.free(payload);
    const endpoint = try anthropic.messagesUrl(definition, alloc);
    defer alloc.free(endpoint);
    const beta = try codec.betaHeader(alloc, &definition.anthropic, request.model);
    defer if (beta) |value| alloc.free(value);
    var fixed: [3]std.http.Header = undefined;
    const extra = anthropic.headers(definition, beta, &fixed);
    var headers: [4]std.http.Header = undefined;
    headers[0] = .{ .name = "content-type", .value = "application/json" };
    @memcpy(headers[1..][0..extra.len], extra);

    const opened = try host_stream.openPost(alloc, transport, request, endpoint, headers[0 .. extra.len + 1], payload);
    defer transport.close(opened.handle);
    if (opened.status != .ok) {
        const body = try host_stream.readErrorBody(alloc, transport, opened.handle, request);
        return .{ .failed = .{
            .kind = anthropic.failureKind(alloc, opened.status, body),
            .http_status = opened.status,
            .detail = body,
            .retry_after_seconds = opened.retry_after_seconds,
            .ownership = .owned,
        } };
    }

    var reader: host_stream.HostStreamReader = undefined;
    reader.init(transport, opened.handle, request.cancel_flag, request.deadline, request.cooperative_pulse);
    const result = anthropic.consume(alloc, payload, &reader.interface, request, null) catch |err| switch (err) {
        error.ReadFailed => return host_stream.readFailure(&reader, request.cancel_flag),
        else => return err,
    };
    switch (result) {
        .completed => |completed| host_stream.signalConsumed(transport, opened.handle, request, completed.completion),
        else => {},
    }
    return result;
}

const FakeTransport = @import("host_transport_testing.zig").FakeTransport;

const test_provider =
    \\{"claude":{"protocol":"anthropic-messages","auth":{"type":"none"},"anthropic":{"compaction":{}}}}
;

const reply_sse =
    "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_h\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n" ++
    "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"libfx on Claude\"}}\n\n" ++
    "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" ++
    "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n" ++
    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

test "host anthropic messages posts to /v1/messages with version and beta headers and no key" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .model = "claude-haiku-5-5", .responses = &.{.{ .status = 200, .body = reply_sse }} };
    defer fake.deinit();
    var context = try Context.init(alloc, fake.transport(), test_provider);
    defer context.deinit(alloc);
    const served = host_chat_completions.bundle(&context, context.definition());
    var result = try served.agent_stream.?.stream(alloc, fake.request(.host_managed));
    defer result.deinit(alloc);

    try std.testing.expectEqualStrings("https://api.anthropic.com/v1/messages", fake.urls[0]);
    try std.testing.expectEqualStrings(
        "[{\"name\":\"content-type\",\"value\":\"application/json\"},{\"name\":\"anthropic-version\",\"value\":\"2023-06-01\"},{\"name\":\"accept\",\"value\":\"text/event-stream\"},{\"name\":\"anthropic-beta\",\"value\":\"" ++ codec.thinking_binding_beta ++ "," ++ codec.compaction_beta ++ "\"}]",
        fake.headers[0],
    );
    try std.testing.expect(std.mem.find(u8, fake.headers[0], "x-api-key") == null);
    try std.testing.expectEqualStrings("libfx on Claude", result.completed.completion.content.?);
    try std.testing.expectEqual(@as(usize, 1), fake.consumed);
    try std.testing.expectEqual(fake.calls, fake.closed);
}

test "host anthropic messages maps a rejected request and refuses an engine-held key" {
    const alloc = std.testing.allocator;
    var fake = FakeTransport{ .model = "claude-haiku-5-5", .responses = &.{.{ .status = 529, .body = "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}" }} };
    defer fake.deinit();
    var context = try Context.init(alloc, fake.transport(), test_provider);
    defer context.deinit(alloc);
    const served = host_chat_completions.bundle(&context, context.definition());
    var result = try served.agent_stream.?.stream(alloc, fake.request(.host_managed));
    defer result.deinit(alloc);
    try std.testing.expectEqual(stream_provider.FailureKind.unavailable, result.failed.kind);
    const lease: stream_provider.CredentialLease = .{ .direct = .{ .secret_bytes = "key", .source = .configured } };
    try std.testing.expectError(error.ConfiguredProviderCredentialRequired, served.agent_stream.?.stream(alloc, fake.request(lease)));
}
