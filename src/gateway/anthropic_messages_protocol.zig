//! Anthropic Messages API codec: the request body for `POST /v1/messages`
//! and the reducer for its streamed reply. No I/O; the transport in
//! `anthropic_messages.zig` owns the connection.
const std = @import("std");
const testing_allocator = @import("../core/shared/testing_allocator.zig");
const stream_provider = @import("../core/agent/stream_provider.zig");
const types = @import("../core/shared/types.zig");
const model_tool_schema = @import("../core/tooling/model_tool_schema.zig");
const tool_result_errors = @import("../core/tooling/tool_result_errors.zig");
const image_attachments = @import("../core/images/image_attachments.zig");
const model_provider = @import("../core/config/model_provider.zig");
const configured_provider = @import("../core/config/configured_provider.zig");
const anthropic_options = @import("../core/config/anthropic_provider_options.zig");
const debug_trace = @import("../core/shared/debug_trace.zig");
const tool_call_ids = @import("tool_call_ids.zig");
const json_depth = @import("json_depth.zig");
const sse = @import("sse.zig");

const Allocator = std.mem.Allocator;

/// Tells whether the history before a reply changed: two 64-bit Wyhash
/// lanes, fast enough to run over every body. Not a security boundary; the API
/// checks the signatures itself.
const PrefixHash = struct {
    a: std.hash.Wyhash = .init(0x66786132),
    b: std.hash.Wyhash = .init(0x70726566),

    fn update(self: *PrefixHash, bytes: []const u8) void {
        self.a.update(bytes);
        self.b.update(bytes);
    }

    fn final(self: PrefixHash) Digest {
        var copy = self;
        var digest: Digest = undefined;
        std.mem.writeInt(u64, digest[0..8], copy.a.final(), .little);
        std.mem.writeInt(u64, digest[8..16], copy.b.final(), .little);
        return digest;
    }
};
const Digest = [16]u8;
const digest_hex_len = 2 * @sizeOf(Digest);

pub const Options = anthropic_options.Options;

pub const Error = error{
    OutOfMemory,
    InvalidProviderPrompt,
    InvalidModel,
    UnsupportedProviderOption,
    UnsupportedVision,
    UnsupportedResponseFormat,
    UnsupportedReplay,
    InvalidProviderState,
    ReplayTooLarge,
    UnsupportedToolProvenance,
    InvalidOutputLimit,
    InvalidToolSelection,
    InvalidToolSchema,
    InvalidToolCallId,
    InvalidToolName,
    InvalidToolHistory,
    ImageUnavailable,
    RequiredToolMissing,
    UnexpectedToolCall,
    InvalidChunk,
    MalformedAnthropicEvent,
    ConflictingIdentity,
    InvalidFinishReason,
    InconsistentFinishReason,
    IncompleteStream,
    StreamClosed,
    ProviderError,
    EventTooLarge,
    StreamTooLarge,
    TooManyEvents,
    TooManyTools,
    IdentityTooLarge,
    ArgumentsTooLarge,
    ContentTooLarge,
    JsonTooDeep,
    Cancelled,
    ReadFailed,
};

/// What fx knows about the current Claude models without asking the Models
/// API. Other model IDs work too; they get no thinking or effort settings
/// unless the provider definition says more about them.
pub const ModelInfo = struct {
    id: []const u8,
    context_window: u32,
    max_output_tokens: u32,
    /// A forced `tool_choice` (`any` or `tool`) is a 400 on this model.
    rejects_forced_tool_choice: bool,
};

pub const efforts = [_][]const u8{ "low", "medium", "high", "xhigh", "max" };

pub const builtin_models = [_]ModelInfo{
    .{ .id = "claude-opus-5-5", .context_window = 1_000_000, .max_output_tokens = 128_000, .rejects_forced_tool_choice = true },
    .{ .id = "claude-sonnet-5-5", .context_window = 1_000_000, .max_output_tokens = 128_000, .rejects_forced_tool_choice = true },
    .{ .id = "claude-haiku-5-5", .context_window = 1_000_000, .max_output_tokens = 128_000, .rejects_forced_tool_choice = false },
};

pub fn builtinModel(id: []const u8) ?*const ModelInfo {
    for (&builtin_models) |*info| if (std.mem.eql(u8, info.id, id)) return info;
    return null;
}

/// `max_tokens` is required; fx leaves it unset for some side calls.
const default_max_tokens: u32 = 32_000;
const max_selected_tools = 256;
const max_name_bytes = 128;
const max_history_arguments_bytes = 1024 * 1024;

pub const thinking_binding_beta = "thinking-binding-controls-2026-08-01";
pub const compaction_beta = "compact-2026-01-12";
pub const context_management_beta = "context-management-2025-06-27";

pub const Settings = struct {
    options: *const Options,
    /// Borrowed during serialization; includes the configured authority binding.
    provider: ?*const model_provider.ProviderId = null,
};

/// The `anthropic-beta` header value, or null when the request needs none.
/// Caller owns the result.
pub fn betaHeader(alloc: Allocator, options: *const Options, model: []const u8) Allocator.Error!?[]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var listed = std.mem.splitScalar(u8, options.betas, ',');
    while (listed.next()) |beta| if (beta.len != 0) try appendBeta(alloc, &out, beta);
    if (sendsBlockBinding(options, model)) try appendBeta(alloc, &out, thinking_binding_beta);
    // The edits in `context_management` each need their beta.
    if (options.context_management) |edits| {
        if (std.mem.find(u8, edits, "\"compact_") != null) try appendBeta(alloc, &out, compaction_beta);
        if (std.mem.find(u8, edits, "\"clear_") != null) try appendBeta(alloc, &out, context_management_beta);
    }
    if (out.items.len == 0) return null;
    return try out.toOwnedSlice(alloc);
}

fn appendBeta(alloc: Allocator, out: *std.ArrayList(u8), beta: []const u8) Allocator.Error!void {
    var iterator = std.mem.splitScalar(u8, out.items, ',');
    while (iterator.next()) |existing| if (std.mem.eql(u8, existing, beta)) return;
    if (out.items.len != 0) try out.append(alloc, ',');
    try out.appendSlice(alloc, beta);
}

fn sendsBlockBinding(options: *const Options, model: []const u8) bool {
    return options.prefix_mismatch_behavior != .unset and builtinModel(model) != null;
}

// ---------------------------------------------------------------------------
// Tool names. Anthropic accepts [A-Za-z0-9_-]{1,128}; fx also advertises
// dotted names, which travel with each dot written as "__".

fn validateName(name: []const u8) Error!void {
    if (name.len == 0 or name.len > max_name_bytes) return error.InvalidToolName;
    for (name) |byte| if (!std.ascii.isAlphanumeric(byte) and byte != '_' and byte != '-' and byte != '.') return error.InvalidToolName;
}

fn wireName(buffer: *[2 * max_name_bytes]u8, name: []const u8) Error![]const u8 {
    var len: usize = 0;
    for (name) |byte| {
        if (byte == '.') {
            buffer[len] = '_';
            buffer[len + 1] = '_';
            len += 2;
        } else {
            buffer[len] = byte;
            len += 1;
        }
    }
    if (len > max_name_bytes) return error.InvalidToolName;
    return buffer[0..len];
}

fn writeWireName(writer: *std.Io.Writer, name: []const u8) !void {
    var buffer: [2 * max_name_bytes]u8 = undefined;
    try std.json.Stringify.value(try wireName(&buffer, name), .{}, writer);
}

const Function = struct {
    name: []const u8,
    description: []const u8,
    schema: union(enum) {
        builtin: model_tool_schema.ObjectSchema,
        dynamic: std.json.Value,
    },
};

fn containsName(names: []const []const u8, name: []const u8) bool {
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}

fn appendFunction(alloc: Allocator, functions: *std.ArrayList(Function), function: Function) Error!void {
    try validateName(function.name);
    if (functions.items.len == max_selected_tools) return error.TooManyTools;
    for (functions.items) |prior| if (std.mem.eql(u8, prior.name, function.name)) return error.InvalidToolSelection;
    try functions.append(alloc, function);
}

/// Every advertised tool, whatever `tool_choice` says: the tool set is part of
/// the prefix a thinking block is bound to, so it must not change when one
/// request turns tools off.
fn selectFunctions(alloc: Allocator, tools: stream_provider.ToolSelection) Error!std.ArrayList(Function) {
    var functions: std.ArrayList(Function) = .empty;
    errdefer functions.deinit(alloc);
    if (tools.advertised_names.len > max_selected_tools or tools.additional_functions.len > max_selected_tools or tools.selected_dynamic.len > max_selected_tools) return error.TooManyTools;
    for (tools.advertised_names) |name| {
        const function = tools.advertisedFunction(name) orelse {
            const registered = tools.registry.lookup(name) orelse return error.InvalidToolSelection;
            if (registered.provider_executed) continue;
            return error.InvalidToolSelection;
        };
        try appendFunction(alloc, &functions, .{ .name = name, .description = function.description, .schema = .{ .builtin = function.input_schema } });
    }
    for (tools.additional_functions) |function| {
        if (containsName(tools.advertised_names, function.name)) continue;
        try appendFunction(alloc, &functions, .{ .name = function.name, .description = function.description, .schema = .{ .builtin = function.input_schema } });
    }
    for (tools.selected_dynamic) |function| {
        if (containsName(tools.advertised_names, function.name)) continue;
        if (function.input_schema != .object) return error.InvalidToolSchema;
        try checkSchemaDepth(function.input_schema, 0);
        try appendFunction(alloc, &functions, .{ .name = function.name, .description = function.description, .schema = .{ .dynamic = function.input_schema } });
    }
    return functions;
}

fn checkSchemaDepth(value: std.json.Value, depth: usize) Error!void {
    if (depth > json_depth.max_depth) return error.JsonTooDeep;
    switch (value) {
        .object => |fields| for (fields.values()) |child| try checkSchemaDepth(child, depth + 1),
        .array => |items| for (items.items) |child| try checkSchemaDepth(child, depth + 1),
        .float => |number| if (!std.math.isFinite(number)) return error.InvalidToolSchema,
        else => {},
    }
}

/// Maps a tool name from the reply back to fx's name.
fn originalName(functions: []const Function, wire: []const u8) []const u8 {
    for (functions) |function| {
        var buffer: [2 * max_name_bytes]u8 = undefined;
        const encoded = wireName(&buffer, function.name) catch continue;
        if (std.mem.eql(u8, encoded, wire)) return function.name;
    }
    return wire;
}

// ---------------------------------------------------------------------------
// Validation

fn validateRequest(request: stream_provider.RequestData) Error!void {
    try request.validatePrompt();
    configured_provider.validate_model_id(request.model) catch return error.InvalidModel;
    const options = request.provider_options;
    if (options.fast or options.ultrafast) return error.UnsupportedProviderOption;
    if (options.provider_order.len != 0) return error.UnsupportedProviderOption;
    if (request.response_format != null) return error.UnsupportedResponseFormat;
    if (request.vision_mode != .unavailable) return error.UnsupportedVision;
    if (request.verified_images != null and request.verified_images.?.len != 0) return error.UnsupportedVision;
    if (request.max_output_tokens == 0) return error.InvalidOutputLimit;
    for (request.messages) |message| {
        if (message.images.len != 0 and message.role != .user) return error.InvalidProviderPrompt;
        if (message.provider_replay != null and message.role != .assistant) return error.InvalidProviderState;
        if (message.role != .assistant and message.tool_calls.len != 0) return error.InvalidToolHistory;
        if (message.role != .tool and message.tool_call_id != null) return error.InvalidToolHistory;
        if (message.role != .assistant and message.content == null) return error.InvalidProviderPrompt;
    }
}

fn validateHistory(alloc: Allocator, messages: []const types.ChatMessage) Error!void {
    var pending: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer pending.deinit(alloc);
    for (messages) |message| {
        if (message.role == .tool) {
            const id = message.tool_call_id orelse return error.InvalidToolCallId;
            const name = pending.get(id) orelse return error.InvalidToolHistory;
            if (message.tool_name) |tool_name| if (!std.mem.eql(u8, tool_name, name)) return error.InvalidToolHistory;
            _ = pending.remove(id);
            continue;
        }
        if (pending.count() != 0) return error.InvalidToolHistory;
        if (message.tool_calls.len > max_selected_tools) return error.TooManyTools;
        for (message.tool_calls) |call| {
            if (call.provenance != .fx_local or call.provider_result != null) return error.UnsupportedToolProvenance;
            if (call.id.len == 0 or call.final_identity != .valid) return error.InvalidToolCallId;
            try validateName(call.name);
            if (call.arguments_json.len > max_history_arguments_bytes) return error.ArgumentsTooLarge;
            if (json_depth.exceeds(call.arguments_json)) return error.JsonTooDeep;
            const entry = try pending.getOrPut(alloc, call.id);
            if (entry.found_existing) return error.InvalidToolHistory;
            entry.value_ptr.* = call.name;
        }
    }
    if (pending.count() != 0) return error.InvalidToolHistory;
}

// ---------------------------------------------------------------------------
// Replay. A reply keeps only what fx cannot rebuild from its own copy of the
// message: its thinking, redacted_thinking and compaction blocks, serialized
// once so every later request copies them unchanged, and where its text and
// tool_use blocks sat between them. The text needs no JSON parser to read:
//   fxa1 <prefix>\n<layout>\n<blocks>
// `layout` lists the reply's blocks in order: `t<bytes>` a text block of fx's
// content, `u<id>` a tool call, `k<bytes>` a thinking or redacted_thinking
// block and `c<bytes>` a compaction block in `blocks` (`C` when it holds a
// summary). `prefix` is the SHA-256 of the body that produced the reply, up to
// the end of its last message, in hex. Every body starts with system, tools
// and messages, so those bytes are the history the reply's thinking is bound
// to. `compaction` marks a reply with a summary, whose history starts at that
// block. When the bytes before a reply differ on a later request, its thinking
// would fail the API's preserved-thinking check, so the request leaves it out.

const replay_magic = "fxa1 ";
const compaction_prefix = "compaction";
/// Ends the messages array in every body this codec writes.
const messages_end = "],\"model\":";

const Replay = struct {
    prefix: []const u8,
    layout: []const u8,
    blocks: []const u8,

    /// Null for text this codec did not write, such as an earlier format.
    fn parse(raw: []const u8) ?Replay {
        if (!std.mem.startsWith(u8, raw, replay_magic)) return null;
        const head_end = std.mem.findScalar(u8, raw, '\n') orelse return null;
        const layout_end = head_end + 1 + (std.mem.findScalar(u8, raw[head_end + 1 ..], '\n') orelse return null);
        return .{ .prefix = raw[replay_magic.len..head_end], .layout = raw[head_end + 1 .. layout_end], .blocks = raw[layout_end + 1 ..] };
    }

    fn entries(self: Replay) std.mem.TokenIterator(u8, .scalar) {
        return std.mem.tokenizeScalar(u8, self.layout, ' ');
    }

    /// Whether the tool calls the layout names are `calls`, in order.
    fn matchesCalls(self: Replay, calls: []const types.ToolCall) bool {
        var it = self.entries();
        var index: usize = 0;
        while (it.next()) |entry| {
            if (entry[0] != 'u') continue;
            if (index == calls.len or !std.mem.eql(u8, entry[1..], calls[index].id)) return false;
            index += 1;
        }
        return index == calls.len;
    }

    fn hasThinking(self: Replay) bool {
        var it = self.entries();
        while (it.next()) |entry| if (entry[0] == 'k') return true;
        return false;
    }

    fn summarizes(self: Replay) bool {
        if (!std.mem.eql(u8, self.prefix, compaction_prefix)) return false;
        var it = self.entries();
        while (it.next()) |entry| if (entry[0] == 'C') return true;
        return false;
    }
};

fn entrySize(entry: []const u8) ?usize {
    return std.fmt.parseUnsigned(usize, entry[1..], 10) catch null;
}

/// Keeps the reply's thinking with the calls fx keeps. `text` false drops the
/// text blocks, `reasoning` false drops the whole replay. Returns `replay`
/// itself when nothing changes; a projected copy lives in `alloc`.
pub fn project_replay(alloc: Allocator, replay: ?types.ProviderReplay, calls: []const types.ToolCall, text: bool, reasoning: bool) Error!?types.ProviderReplay {
    const value = replay orelse return null;
    if (!reasoning) return null;
    const parsed = Replay.parse(value.parts_json) orelse return null;
    var layout: std.Io.Writer.Allocating = .init(alloc);
    defer layout.deinit();
    var blocks: std.Io.Writer.Allocating = .init(alloc);
    defer blocks.deinit();
    var changed = false;
    var kept_calls: usize = 0;
    var offset: usize = 0;
    var it = parsed.entries();
    while (it.next()) |entry| {
        var keep = true;
        switch (entry[0]) {
            't' => keep = text,
            'u' => {
                keep = for (calls) |call| {
                    if (std.mem.eql(u8, call.id, entry[1..])) break true;
                } else false;
                if (keep) kept_calls += 1;
            },
            'k', 'c', 'C' => {
                const size = entrySize(entry) orelse return error.InvalidProviderState;
                if (size > parsed.blocks.len - offset) return error.InvalidProviderState;
                blocks.writer.writeAll(parsed.blocks[offset..][0..size]) catch return error.OutOfMemory;
                offset += size;
            },
            else => return error.InvalidProviderState,
        }
        if (!keep) {
            changed = true;
            continue;
        }
        if (layout.written().len != 0) layout.writer.writeByte(' ') catch return error.OutOfMemory;
        layout.writer.writeAll(entry) catch return error.OutOfMemory;
    }
    if (kept_calls != calls.len) return null;
    if (!changed) return value;
    const parts = std.mem.concat(alloc, u8, &.{ replay_magic, parsed.prefix, "\n", layout.written(), "\n", blocks.written() }) catch return error.OutOfMemory;
    return .{ .source = value.source, .parts_json = parts };
}

/// The digest a reply to `body` records as its prefix, or null for a body
/// this codec did not write.
pub fn bodyPrefix(body: []const u8) ?Digest {
    const end = std.mem.findLast(u8, body, messages_end) orelse return null;
    var hasher: PrefixHash = .{};
    hasher.update(body[0..end]);
    return hasher.final();
}

// ---------------------------------------------------------------------------
// Request serialization

/// Pure Messages serialization. Caller owns the returned bytes.
pub fn build_request(alloc: Allocator, input: stream_provider.RequestData, settings: Settings) Error![]u8 {
    try validateRequest(input);
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var request = input;
    if (settings.provider) |provider| {
        if (try types.projectProviderReplay(arena, input.messages, .{ .provider = provider.*, .model = input.model })) |projected| request.messages = projected;
    } else {
        for (input.messages) |message| if (message.provider_replay != null) return error.UnsupportedReplay;
    }
    try validateHistory(arena, request.messages);
    const functions = try selectFunctions(arena, request.tools);
    if (request.tool_choice == .required and functions.items.len == 0) return error.RequiredToolMissing;
    var projection = tool_call_ids.Projection.init(arena, request.messages) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidToolCallId,
    };
    _ = &projection;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    var renderer: Renderer = .{
        .arena = arena,
        .request = request,
        .settings = settings,
        .functions = functions.items,
        .projection = &projection,
        .out = &out,
    };
    renderer.write() catch |err| switch (err) {
        error.WriteFailed, error.OutOfMemory => return error.OutOfMemory,
        error.InvalidProviderState, error.ReplayTooLarge, error.JsonTooDeep, error.ImageUnavailable, error.InvalidProviderPrompt, error.InvalidToolName => |failure| return failure,
        else => return error.InvalidToolSchema,
    };
    return out.toOwnedSlice();
}

fn toolResultDenied(message: types.ChatMessage) bool {
    const failed = if (message.tool_result_status) |status| status == .failure else false;
    return failed and tool_result_errors.toolPermissionDenialReason(message.content orelse "") != null;
}

const Side = enum {
    user,
    assistant,

    fn open(side: Side) []const u8 {
        return switch (side) {
            .user => "{\"role\":\"user\",\"content\":[",
            .assistant => "{\"role\":\"assistant\",\"content\":[",
        };
    }
};

const Renderer = struct {
    arena: Allocator,
    request: stream_provider.RequestData,
    settings: Settings,
    functions: []const Function,
    projection: *const tool_call_ids.Projection,
    /// The body. Prefix digests hash its bytes as they are written.
    out: *std.Io.Writer.Allocating,
    hasher: PrefixHash = .{},
    hashed: usize = 0,
    wire_count: usize = 0,

    /// The hasher over every byte written so far.
    fn synced(self: *Renderer) PrefixHash {
        const written = self.out.written();
        self.hasher.update(written[self.hashed..]);
        self.hashed = written.len;
        return self.hasher;
    }

    /// The prefix digest after the messages written so far, the message
    /// still open, and a user message of `first` and `second` blocks.
    fn pendingDigest(self: *Renderer, open: ?Side, body: []const u8, first: ?[]const u8, second: []const u8) Digest {
        var hasher = self.synced();
        var count = self.wire_count;
        if (open) |side| {
            if (count != 0) hasher.update(",");
            hasher.update(side.open());
            hasher.update(body);
            hasher.update("]}");
            count += 1;
        }
        if (first) |blocks| {
            if (count != 0) hasher.update(",");
            hasher.update(Side.user.open());
            hasher.update(blocks);
            if (blocks.len != 0 and second.len != 0) hasher.update(",");
            hasher.update(second);
            hasher.update("]}");
        }
        return hasher.final();
    }

    fn write(self: *Renderer) !void {
        const w = &self.out.writer;
        const request = self.request;
        const options = self.settings.options;
        try w.writeByte('{');
        try self.writeSystem(w);
        try self.writeTools(w);
        try w.writeAll("\"messages\":[");
        try self.writeMessages(w);
        // Everything after the messages may change without touching the
        // history a reply's thinking is bound to.
        try w.writeAll(messages_end);
        try std.json.Stringify.value(request.model, .{}, w);
        const info = builtinModel(request.model);
        const limit = request.max_output_tokens orelse if (info) |model| @min(model.max_output_tokens, default_max_tokens) else default_max_tokens;
        try w.print(",\"max_tokens\":{d},\"stream\":true", .{limit});
        if (info != null) {
            try w.writeAll(",\"thinking\":{\"type\":\"adaptive\",\"display\":\"");
            try w.writeAll(@tagName(options.thinking_display));
            try w.writeByte('"');
            if (sendsBlockBinding(options, request.model)) {
                try w.writeAll(",\"block_binding\":{\"prefix_mismatch_behavior\":\"");
                try w.writeAll(@tagName(options.prefix_mismatch_behavior));
                try w.writeAll("\"}");
            }
            try w.writeByte('}');
        }
        if (request.provider_options.reasoning) |*effort| {
            const label = effort.label();
            if (containsName(&efforts, label)) {
                try w.writeAll(",\"output_config\":{\"effort\":\"");
                try w.writeAll(label);
                try w.writeAll("\"}");
            }
        }
        if (options.prompt_caching) {
            try w.writeAll(",\"cache_control\":");
            try self.writeCacheControl(w);
        }
        try self.writeContextManagement(w);
        try self.writeToolChoice(w);
        try w.writeByte('}');
    }

    fn writeCacheControl(self: *const Renderer, w: *std.Io.Writer) !void {
        try w.writeAll(if (self.settings.options.cache_ttl == .@"1h") "{\"type\":\"ephemeral\",\"ttl\":\"1h\"}" else "{\"type\":\"ephemeral\"}");
    }

    fn writeContextManagement(self: *const Renderer, w: *std.Io.Writer) !void {
        const edits = self.settings.options.context_management orelse return;
        try w.writeAll(",\"context_management\":");
        try w.writeAll(edits);
    }

    fn writeSystem(self: *Renderer, w: *std.Io.Writer) !void {
        var last_nonempty: ?usize = null;
        var flagged = false;
        for (self.request.instructions, 0..) |instruction, index| {
            if (instruction.content.?.len == 0) continue;
            last_nonempty = index;
            flagged = flagged or instruction.cache_breakpoint;
        }
        const last = last_nonempty orelse return;
        try w.writeAll("\"system\":[");
        var count: usize = 0;
        var markers: usize = 0;
        for (self.request.instructions, 0..) |instruction, index| {
            const text = instruction.content.?;
            if (text.len == 0) continue;
            if (count != 0) try w.writeByte(',');
            count += 1;
            try w.writeAll("{\"type\":\"text\",\"text\":");
            try std.json.Stringify.value(text, .{}, w);
            const marked = self.settings.options.prompt_caching and markers < 2 and
                (if (flagged) instruction.cache_breakpoint else index == last);
            if (marked) {
                markers += 1;
                try w.writeAll(",\"cache_control\":");
                try self.writeCacheControl(w);
            }
            try w.writeByte('}');
        }
        try w.writeAll("],");
    }

    fn writeTools(self: *Renderer, w: *std.Io.Writer) !void {
        if (self.functions.len == 0) return;
        try w.writeAll("\"tools\":[");
        for (self.functions, 0..) |function, index| {
            if (index != 0) try w.writeByte(',');
            try w.writeAll("{\"name\":");
            try writeWireName(w, function.name);
            try w.writeAll(",\"description\":");
            try model_tool_schema.writeCappedDescriptionJsonString(self.arena, w, function.description);
            try w.writeAll(",\"input_schema\":");
            // A strict tool needs its object schema closed. fx closes its own
            // tools' schemas; another tool is strict only when it already is.
            var strict = Options.listed(self.settings.options.strict_tools, function.name);
            switch (function.schema) {
                .builtin => |schema| {
                    var sent = schema;
                    if (strict) sent.additional_properties = false;
                    try model_tool_schema.writeObjectSchema(self.arena, w, sent);
                },
                .dynamic => |schema| {
                    const closed = schema.object.get("additionalProperties");
                    strict = strict and closed != null and closed.? == .bool and !closed.?.bool;
                    try std.json.Stringify.value(schema, .{}, w);
                },
            }
            if (strict) try w.writeAll(",\"strict\":true");
            try w.writeByte('}');
        }
        try w.writeAll("],");
    }

    fn writeToolChoice(self: *const Renderer, w: *std.Io.Writer) !void {
        if (self.functions.len == 0) return;
        const choice: []const u8 = switch (self.request.tool_choice) {
            .auto => "auto",
            .none => "none",
            .required => if (builtinModel(self.request.model)) |info| (if (info.rejects_forced_tool_choice) "auto" else "any") else "any",
        };
        try w.writeAll(",\"tool_choice\":{\"type\":\"");
        try w.writeAll(choice);
        try w.writeByte('"');
        if (self.request.provider_options.parallel_tool_calls) |parallel| {
            if (!parallel and !std.mem.eql(u8, choice, "none")) try w.writeAll(",\"disable_parallel_tool_use\":true");
        }
        try w.writeByte('}');
    }

    /// Index of the newest assistant reply that carries a server compaction
    /// summary. The API ignores everything before that block, so the request
    /// starts there and fx's own size estimate sees only what is left.
    fn compactionStart(self: *const Renderer) Error!usize {
        var index = self.request.messages.len;
        while (index > 0) {
            index -= 1;
            const message = self.request.messages[index];
            if (message.role != .assistant) continue;
            const replay = message.provider_replay orelse continue;
            const parsed = Replay.parse(replay.parts_json) orelse continue;
            if (parsed.summarizes()) return index;
        }
        return 0;
    }

    fn writeMessages(self: *Renderer, w: *std.Io.Writer) !void {
        const messages = self.request.messages;
        const start = try self.compactionStart();
        if (start != 0) debug_trace.logf("gateway", "anthropic_compaction_history_trimmed omitted_messages={d}", .{start});
        var current: ?Side = null;
        // The open wire message's blocks, one source message's blocks, and
        // the parts of a user run. Each is reused for every message.
        var body: std.Io.Writer.Allocating = .init(self.arena);
        var blocks: std.Io.Writer.Allocating = .init(self.arena);
        var kept: std.Io.Writer.Allocating = .init(self.arena);
        var tail: std.Io.Writer.Allocating = .init(self.arena);
        // User blocks moved behind the reply that follows them; see deferTail.
        var deferred: std.Io.Writer.Allocating = .init(self.arena);
        var index = start;
        while (index < messages.len) {
            const message = messages[index];
            const side: Side = if (message.role == .assistant) .assistant else .user;
            blocks.clearRetainingCapacity();
            const first_wire = self.wire_count == 0 and (current == null or current == side);
            var consumed: usize = 1;
            switch (side) {
                .assistant => {
                    // A reply's prefix ends with the message before its own
                    // wire message: an open user message counts, an open
                    // assistant message this one joins does not.
                    const open: ?Side = if (current == .assistant) null else current;
                    try self.writeAssistant(&blocks.writer, message, open, body.written(), first_wire);
                },
                .user => {
                    kept.clearRetainingCapacity();
                    tail.clearRetainingCapacity();
                    consumed = try self.writeUserRun(&kept.writer, &tail.writer, messages[index..]);
                    try appendBlocks(&kept, deferred.written());
                    deferred.clearRetainingCapacity();
                    const next = index + consumed;
                    if (next < messages.len and try self.deferTail(current, body.written(), kept.written(), tail.written(), messages[next])) {
                        try deferred.writer.writeAll(tail.written());
                    } else try appendBlocks(&kept, tail.written());
                    try blocks.writer.writeAll(kept.written());
                },
            }
            index += consumed;
            if (blocks.written().len == 0) continue;
            if (current != side) {
                if (current) |open| try self.flush(w, &body, open);
                current = side;
            } else try body.writer.writeByte(',');
            try body.writer.writeAll(blocks.written());
        }
        if (deferred.written().len != 0) {
            if (current == .user) try body.writer.writeByte(',') else if (current) |open| try self.flush(w, &body, open);
            try body.writer.writeAll(deferred.written());
            current = .user;
        }
        if (current) |open| try self.flush(w, &body, open);
    }

    fn appendBlocks(out: *std.Io.Writer.Allocating, more: []const u8) !void {
        if (more.len == 0) return;
        if (out.written().len != 0) try out.writer.writeByte(',');
        try out.writer.writeAll(more);
    }

    /// fx adds notes, such as the session's file evidence, to the end of a
    /// turn's last tool results after the turn's final reply exists. Sent in
    /// place, they change the history that reply's thinking is bound to.
    /// Returns true when the reply `next` was produced without `tail`, so
    /// `tail` goes behind the reply instead, where the model first sees it.
    fn deferTail(self: *Renderer, open: ?Side, body: []const u8, kept: []const u8, tail: []const u8, next: types.ChatMessage) !bool {
        if (tail.len == 0 or kept.len == 0) return false;
        if (next.role != .assistant or next.tool_calls.len != 0) return false;
        const replay = next.provider_replay orelse return false;
        const parsed = Replay.parse(replay.parts_json) orelse return false;
        if (parsed.prefix.len != digest_hex_len) return false;
        var hex = std.fmt.bytesToHex(self.pendingDigest(open, body, kept, tail), .lower);
        if (std.mem.eql(u8, parsed.prefix, &hex)) return false;
        hex = std.fmt.bytesToHex(self.pendingDigest(open, body, kept, ""), .lower);
        const bound = std.mem.eql(u8, parsed.prefix, &hex);
        if (bound) debug_trace.logf("gateway", "anthropic_user_tail_deferred bytes={d}", .{tail.len});
        return bound;
    }

    fn flush(self: *Renderer, w: *std.Io.Writer, body: *std.Io.Writer.Allocating, side: Side) !void {
        if (self.wire_count != 0) try w.writeByte(',');
        try w.writeAll(side.open());
        try w.writeAll(body.written());
        try w.writeAll("]}");
        self.wire_count += 1;
        body.clearRetainingCapacity();
    }

    /// Writes a run of tool results and user messages as the blocks of one
    /// user message: tool results to `results`, which the API requires first,
    /// and the rest to `rest`. Returns how many source messages it used.
    fn writeUserRun(self: *Renderer, results: *std.Io.Writer, rest: *std.Io.Writer, messages: []const types.ChatMessage) !usize {
        var count: usize = 0;
        var wrote_results = false;
        var wrote_rest = false;
        for (messages) |message| {
            if (message.role == .assistant) break;
            count += 1;
            switch (message.role) {
                .tool => try self.writeToolResult(results, message, &wrote_results),
                else => try self.writeUserBlocks(rest, message, &wrote_rest),
            }
        }
        return count;
    }

    fn separator(w: *std.Io.Writer, wrote: *bool) !void {
        if (wrote.*) try w.writeByte(',');
        wrote.* = true;
    }

    fn writeText(w: *std.Io.Writer, text: []const u8) !void {
        try w.writeAll("{\"type\":\"text\",\"text\":");
        try std.json.Stringify.value(text, .{}, w);
        try w.writeByte('}');
    }

    fn writeUserBlocks(self: *Renderer, w: *std.Io.Writer, message: types.ChatMessage, wrote: *bool) !void {
        for (message.images) |image| {
            var snapshot = image_attachments.loadVerifiedSnapshot(self.arena, image, .{}) catch return error.ImageUnavailable;
            defer snapshot.deinit(self.arena);
            try separator(w, wrote);
            try writeImageRaw(w, snapshot.media_type, snapshot.bytes);
        }
        const text = message.content.?;
        if (text.len != 0) {
            try separator(w, wrote);
            try writeText(w, text);
        }
    }

    fn writeToolResult(self: *Renderer, w: *std.Io.Writer, message: types.ChatMessage, wrote: *bool) !void {
        try separator(w, wrote);
        try w.writeAll("{\"type\":\"tool_result\",\"tool_use_id\":");
        try std.json.Stringify.value(self.projection.resolve(message.tool_call_id.?), .{}, w);
        const text = message.content.?;
        const images = if (message.tool_result_memory) |memory| (if (toolResultDenied(message)) &.{} else memory.tool_images) else &.{};
        if (text.len != 0 or images.len != 0) {
            try w.writeAll(",\"content\":[");
            var inner = false;
            if (text.len != 0) {
                try separator(w, &inner);
                try writeText(w, text);
            }
            for (images) |image| {
                if (image.data.len == 0) continue;
                try separator(w, &inner);
                try writeImageEncoded(w, image.mime_type, image.data);
            }
            try w.writeByte(']');
        }
        if (message.tool_result_status) |status| if (status == .failure) try w.writeAll(",\"is_error\":true");
        try w.writeByte('}');
    }

    fn writeToolUse(self: *Renderer, w: *std.Io.Writer, call: types.ToolCall) !void {
        try w.writeAll("{\"type\":\"tool_use\",\"id\":");
        try std.json.Stringify.value(self.projection.resolve(call.id), .{}, w);
        try w.writeAll(",\"name\":");
        try writeWireName(w, call.name);
        try w.writeAll(",\"input\":");
        try writeToolInput(self.arena, w, call.arguments_json);
        try w.writeByte('}');
    }

    fn writeAssistant(self: *Renderer, w: *std.Io.Writer, message: types.ChatMessage, open: ?Side, body: []const u8, first_wire: bool) !void {
        if (message.provider_replay) |replay| {
            if (try self.writeReplayed(w, message, replay, open, body, first_wire)) return;
        }
        var wrote = false;
        if (message.content) |text| if (text.len != 0) {
            try separator(w, &wrote);
            try writeText(w, text);
        };
        for (message.tool_calls) |call| {
            try separator(w, &wrote);
            try self.writeToolUse(w, call);
        }
    }

    /// Writes the reply's blocks in their order, with text and tool calls
    /// from fx's message. Returns false when that message no longer matches
    /// the reply, so the canonical form must be written instead.
    fn writeReplayed(self: *Renderer, w: *std.Io.Writer, message: types.ChatMessage, replay: types.ProviderReplay, open: ?Side, body: []const u8, first_wire: bool) !bool {
        const parsed = Replay.parse(replay.parts_json) orelse return false;
        if (!parsed.matchesCalls(message.tool_calls)) return false;
        const content = message.content orelse "";
        var covered: usize = 0;
        var stored: usize = 0;
        var it = parsed.entries();
        while (it.next()) |entry| switch (entry[0]) {
            't' => {
                if (covered != 0) covered += text_block_separator.len;
                covered += entrySize(entry) orelse return false;
            },
            'k', 'c', 'C' => stored += entrySize(entry) orelse return false,
            'u' => {},
            else => return false,
        };
        if (covered != content.len or stored != parsed.blocks.len) return false;
        // Hashing the history is needed only for a reply with thinking.
        const bound = if (std.mem.eql(u8, parsed.prefix, compaction_prefix))
            first_wire
        else if (!parsed.hasThinking())
            true
        else
            std.mem.eql(u8, parsed.prefix, &std.fmt.bytesToHex(self.pendingDigest(open, body, null, ""), .lower));
        if (!bound) debug_trace.logf("gateway", "anthropic_replay_prefix_changed thinking_omitted=true", .{});
        var wrote = false;
        var text_offset: usize = 0;
        var block_offset: usize = 0;
        it.reset();
        while (it.next()) |entry| switch (entry[0]) {
            't' => {
                const size = entrySize(entry).?;
                if (text_offset != 0) text_offset += text_block_separator.len;
                try separator(w, &wrote);
                try writeText(w, content[text_offset..][0..size]);
                text_offset += size;
            },
            'u' => for (message.tool_calls) |call| {
                if (!std.mem.eql(u8, call.id, entry[1..])) continue;
                try separator(w, &wrote);
                try self.writeToolUse(w, call);
                break;
            },
            else => {
                const block = parsed.blocks[block_offset..][0..entrySize(entry).?];
                block_offset += block.len;
                if (entry[0] == 'k' and !bound) continue;
                try separator(w, &wrote);
                try w.writeAll(block);
            },
        };
        return true;
    }
};

/// Text blocks of one reply join with this in fx's assistant content.
pub const text_block_separator = "\n\n";

/// Writes the call's arguments as they are when they are one JSON object,
/// which is what the model sent; anything else becomes an empty object.
fn writeToolInput(arena: Allocator, w: *std.Io.Writer, arguments: []const u8) !void {
    const trimmed = std.mem.trim(u8, arguments, " \t\r\n");
    if (trimmed.len < 2 or trimmed[0] != '{' or !(std.json.validate(arena, trimmed) catch false)) return w.writeAll("{}");
    try w.writeAll(trimmed);
}

fn writeImageEncoded(w: *std.Io.Writer, media_type: []const u8, base64_data: []const u8) !void {
    try w.writeAll("{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":");
    try std.json.Stringify.value(media_type, .{}, w);
    try w.writeAll(",\"data\":\"");
    try w.writeAll(base64_data);
    try w.writeAll("\"}}");
}

fn writeImageRaw(w: *std.Io.Writer, media_type: []const u8, bytes: []const u8) !void {
    try w.writeAll("{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":");
    try std.json.Stringify.value(media_type, .{}, w);
    try w.writeAll(",\"data\":\"");
    var offset: usize = 0;
    while (offset < bytes.len) {
        const end = @min(offset + 3 * 1024, bytes.len);
        try std.base64.standard.Encoder.encodeWriter(w, bytes[offset..end]);
        offset = end;
    }
    try w.writeAll("\"}}");
}

// ---------------------------------------------------------------------------
// Streamed reply

pub const Limits = struct {
    event_bytes: usize = 1024 * 1024,
    total_wire_bytes: usize = 32 * 1024 * 1024,
    events: usize = 200_000,
    blocks: usize = 512,
    tool_calls: usize = 128,
    identity_bytes: usize = 256,
    arguments_bytes: usize = 1024 * 1024,
    content_bytes: usize = 8 * 1024 * 1024,
    reasoning_bytes: usize = types.ProviderReplay.max_bytes,
};

const BlockKind = enum { text, thinking, redacted_thinking, tool_use, compaction };

const Block = struct {
    kind: BlockKind,
    /// text, thinking, partial tool input JSON, or compaction summary.
    text: std.ArrayList(u8) = .empty,
    signature: std.ArrayList(u8) = .empty,
    /// redacted_thinking `data`.
    data: []const u8 = "",
    id: []const u8 = "",
    name: []const u8 = "",
    summary_present: bool = false,
    stopped: bool = false,
};

/// What the reducer hands to presentation for one event. Slices borrow the
/// reducer until its next call.
pub const Deltas = struct {
    /// Set when `content` opens a later text block, which fx joins to the
    /// earlier text with `text_block_separator`.
    separated: bool = false,
    content: ?[]const u8 = null,
    reasoning: ?[]const u8 = null,
    tool_started: ?struct { id: []const u8, name: []const u8 } = null,
    tool_input: ?[]const u8 = null,
};

/// Single-owner, request-local state machine over decoded SSE data events.
/// Owns everything it keeps in its arena; `finish` copies the result into
/// `alloc`.
pub const Reducer = struct {
    alloc: Allocator,
    arena_state: std.heap.ArenaAllocator,
    limits: Limits,
    choice: types.ToolChoice,
    functions: []const Function = &.{},
    prefix: [digest_hex_len]u8,
    blocks: std.ArrayList(Block) = .empty,
    message_id: ?[]const u8 = null,
    usage: struct {
        input: ?u64 = null,
        output: ?u64 = null,
        cache_read: ?u64 = null,
        cache_write: ?u64 = null,
        thinking: ?u64 = null,
    } = .{},
    stop_reason: ?[]const u8 = null,
    stop_details: ?[]const u8 = null,
    text_bytes: usize = 0,
    /// The text block whose text presentation saw last.
    shown_text_block: ?usize = null,
    reasoning_bytes: usize = 0,
    tool_count: usize = 0,
    event_count: usize = 0,
    wire_bytes: usize = 0,
    phase: enum { waiting, receiving, stopped, closed } = .waiting,
    failure: ?struct { kind: stream_provider.FailureKind, detail: []const u8 } = null,

    /// `prefix` is the digest of the request this reply answers, from bodyPrefix.
    pub fn init(alloc: Allocator, request: stream_provider.RequestData, prefix: ?Digest, limits: Limits) Error!Reducer {
        var self: Reducer = .{
            .alloc = alloc,
            .arena_state = std.heap.ArenaAllocator.init(alloc),
            .limits = limits,
            .choice = request.tool_choice,
            .prefix = if (prefix) |digest| std.fmt.bytesToHex(digest, .lower) else @splat('0'),
        };
        errdefer self.arena_state.deinit();
        var functions = try selectFunctions(self.arena_state.allocator(), request.tools);
        self.functions = functions.items;
        _ = &functions;
        return self;
    }

    pub fn deinit(self: *Reducer) void {
        self.arena_state.deinit();
        self.* = undefined;
    }

    fn arena(self: *Reducer) Allocator {
        return self.arena_state.allocator();
    }

    pub fn accept(self: *Reducer, data: []const u8, cancelled: bool) Error!Deltas {
        errdefer self.phase = .closed;
        if (cancelled) return error.Cancelled;
        if (self.phase == .closed or self.phase == .stopped) return error.StreamClosed;
        if (data.len > self.limits.event_bytes) return error.EventTooLarge;
        if (data.len > self.limits.total_wire_bytes - self.wire_bytes) return error.StreamTooLarge;
        self.wire_bytes += data.len;
        if (self.event_count == self.limits.events) return error.TooManyEvents;
        self.event_count += 1;
        if (json_depth.exceeds(data)) return error.JsonTooDeep;
        var parsed = std.json.parseFromSlice(std.json.Value, self.alloc, data, .{ .duplicate_field_behavior = .@"error", .parse_numbers = false }) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.DuplicateField => return error.InvalidChunk,
            else => return error.MalformedAnthropicEvent,
        };
        defer parsed.deinit();
        const root = try object(parsed.value);
        const kind = try string(root.get("type") orelse return error.InvalidChunk);
        if (std.mem.eql(u8, kind, "ping")) return .{};
        if (std.mem.eql(u8, kind, "error")) {
            try self.acceptError(root, data);
            self.phase = .stopped;
            return .{};
        }
        if (std.mem.eql(u8, kind, "message_start")) {
            if (self.phase != .waiting) return error.InvalidChunk;
            const message = try object(root.get("message") orelse return error.InvalidChunk);
            if (nonNull(message, "id")) |id| {
                const text = try string(id);
                if (text.len == 0 or text.len > self.limits.identity_bytes) return error.IdentityTooLarge;
                self.message_id = try self.arena().dupe(u8, text);
            }
            if (nonNull(message, "usage")) |usage| try self.acceptUsage(usage);
            self.phase = .receiving;
            return .{};
        }
        if (self.phase != .receiving) return error.InvalidChunk;
        if (std.mem.eql(u8, kind, "content_block_start")) return self.acceptBlockStart(root);
        if (std.mem.eql(u8, kind, "content_block_delta")) return self.acceptBlockDelta(root);
        if (std.mem.eql(u8, kind, "content_block_stop")) {
            const block = try self.blockAt(root);
            if (block.stopped) return error.InvalidChunk;
            block.stopped = true;
            return .{};
        }
        if (std.mem.eql(u8, kind, "message_delta")) {
            if (nonNull(root, "delta")) |delta_value| {
                const delta = try object(delta_value);
                if (nonNull(delta, "stop_reason")) |reason| {
                    const text = try string(reason);
                    if (text.len == 0 or text.len > 64) return error.InvalidFinishReason;
                    self.stop_reason = try self.arena().dupe(u8, text);
                }
                if (nonNull(delta, "stop_details")) |details| self.stop_details = try std.json.Stringify.valueAlloc(self.arena(), details, .{});
            }
            if (nonNull(root, "usage")) |usage| try self.acceptUsage(usage);
            return .{};
        }
        if (std.mem.eql(u8, kind, "message_stop")) {
            self.phase = .stopped;
            return .{};
        }
        // The stream may grow new event types; ignore them as the API asks.
        return .{};
    }

    /// `data` is the event as it arrived, which becomes the failure's detail.
    fn acceptError(self: *Reducer, root: std.json.ObjectMap, data: []const u8) Error!void {
        const detail = try self.arena().dupe(u8, data);
        var kind: stream_provider.FailureKind = .provider_error;
        if (nonNull(root, "error")) |value| if (value == .object) if (nonNull(value.object, "type")) |name| if (name == .string) {
            kind = errorKind(name.string);
        };
        self.failure = .{ .kind = kind, .detail = detail };
    }

    fn blockAt(self: *Reducer, root: std.json.ObjectMap) Error!*Block {
        const index = try indexValue(root.get("index") orelse return error.InvalidChunk);
        if (index >= self.blocks.items.len) return error.InvalidChunk;
        return &self.blocks.items[index];
    }

    fn acceptBlockStart(self: *Reducer, root: std.json.ObjectMap) Error!Deltas {
        const index = try indexValue(root.get("index") orelse return error.InvalidChunk);
        if (index != self.blocks.items.len) return error.InvalidChunk;
        if (index == self.limits.blocks) return error.StreamTooLarge;
        const start = try object(root.get("content_block") orelse return error.InvalidChunk);
        const kind_name = try string(start.get("type") orelse return error.InvalidChunk);
        const kind = std.meta.stringToEnum(BlockKind, kind_name) orelse return error.InvalidChunk;
        var block: Block = .{ .kind = kind };
        var deltas: Deltas = .{};
        switch (kind) {
            .text => if (nonNull(start, "text")) |text| try self.appendText(&block, try string(text)),
            .thinking => {
                if (nonNull(start, "thinking")) |text| try self.appendReasoning(&block.text, try string(text));
                if (nonNull(start, "signature")) |text| try self.appendReasoning(&block.signature, try string(text));
            },
            .redacted_thinking => {
                const data = try string(start.get("data") orelse return error.InvalidChunk);
                try self.countReasoning(data.len);
                block.data = try self.arena().dupe(u8, data);
            },
            .tool_use => {
                if (self.choice == .none) return error.UnexpectedToolCall;
                if (self.tool_count == self.limits.tool_calls) return error.TooManyTools;
                self.tool_count += 1;
                const id = try string(start.get("id") orelse return error.InvalidChunk);
                const name = try string(start.get("name") orelse return error.InvalidChunk);
                if (id.len == 0 or id.len > self.limits.identity_bytes) return error.IdentityTooLarge;
                if (name.len == 0 or name.len > max_name_bytes) return error.InvalidToolName;
                for (self.blocks.items) |prior| if (prior.kind == .tool_use and std.mem.eql(u8, prior.id, id)) return error.ConflictingIdentity;
                block.id = try self.arena().dupe(u8, id);
                block.name = try self.arena().dupe(u8, name);
                if (nonNull(start, "input")) |input| {
                    if (input != .object) return error.InvalidChunk;
                    if (input.object.count() != 0) {
                        const encoded = try std.json.Stringify.valueAlloc(self.arena(), input, .{});
                        try block.text.appendSlice(self.arena(), encoded);
                    }
                }
                deltas.tool_started = .{ .id = block.id, .name = originalName(self.functions, block.name) };
            },
            .compaction => if (nonNull(start, "content")) |summary| {
                try self.appendCompaction(&block, try string(summary));
            },
        }
        try self.blocks.append(self.arena(), block);
        const added = &self.blocks.items[index];
        if (kind == .text and added.text.items.len != 0) {
            deltas.content = added.text.items;
            deltas.separated = self.showText(index);
        }
        if (kind == .thinking and added.text.items.len != 0) deltas.reasoning = added.text.items;
        return deltas;
    }

    fn acceptBlockDelta(self: *Reducer, root: std.json.ObjectMap) Error!Deltas {
        const block = try self.blockAt(root);
        if (block.stopped) return error.InvalidChunk;
        const delta = try object(root.get("delta") orelse return error.InvalidChunk);
        const kind = try string(delta.get("type") orelse return error.InvalidChunk);
        if (std.mem.eql(u8, kind, "text_delta")) {
            if (block.kind != .text) return error.InvalidChunk;
            const text = try string(delta.get("text") orelse return error.InvalidChunk);
            const start = block.text.items.len;
            try self.appendText(block, text);
            if (block.text.items.len == start) return .{};
            const index = (@intFromPtr(block) - @intFromPtr(self.blocks.items.ptr)) / @sizeOf(Block);
            return .{ .content = block.text.items[start..], .separated = self.showText(index) };
        }
        if (std.mem.eql(u8, kind, "thinking_delta")) {
            if (block.kind != .thinking) return error.InvalidChunk;
            const text = try string(delta.get("thinking") orelse return error.InvalidChunk);
            const start = block.text.items.len;
            try self.appendReasoning(&block.text, text);
            return .{ .reasoning = if (block.text.items.len > start) block.text.items[start..] else null };
        }
        if (std.mem.eql(u8, kind, "signature_delta")) {
            if (block.kind != .thinking) return error.InvalidChunk;
            try self.appendReasoning(&block.signature, try string(delta.get("signature") orelse return error.InvalidChunk));
            return .{};
        }
        if (std.mem.eql(u8, kind, "input_json_delta")) {
            if (block.kind != .tool_use) return error.InvalidChunk;
            const text = try string(delta.get("partial_json") orelse return error.InvalidChunk);
            if (text.len > self.limits.arguments_bytes - block.text.items.len) return error.ArgumentsTooLarge;
            const start = block.text.items.len;
            try block.text.appendSlice(self.arena(), text);
            return .{ .tool_input = if (text.len != 0) block.text.items[start..] else null };
        }
        if (std.mem.eql(u8, kind, "compaction_delta")) {
            if (block.kind != .compaction) return error.InvalidChunk;
            if (nonNull(delta, "content")) |summary| try self.appendCompaction(block, try string(summary));
            return .{};
        }
        // citations_delta and future delta types carry nothing fx keeps.
        return .{};
    }

    fn showText(self: *Reducer, index: usize) bool {
        const separated = if (self.shown_text_block) |shown| shown != index else false;
        self.shown_text_block = index;
        return separated;
    }

    fn appendText(self: *Reducer, block: *Block, text: []const u8) Error!void {
        if (text.len > self.limits.content_bytes - self.text_bytes) return error.ContentTooLarge;
        self.text_bytes += text.len;
        try block.text.appendSlice(self.arena(), text);
    }

    fn appendCompaction(self: *Reducer, block: *Block, text: []const u8) Error!void {
        try self.countReasoning(text.len);
        block.summary_present = true;
        try block.text.appendSlice(self.arena(), text);
    }

    fn countReasoning(self: *Reducer, bytes: usize) Error!void {
        const limit = @min(self.limits.reasoning_bytes, types.ProviderReplay.max_bytes);
        if (bytes > limit - self.reasoning_bytes) return error.ReplayTooLarge;
        self.reasoning_bytes += bytes;
    }

    fn appendReasoning(self: *Reducer, destination: *std.ArrayList(u8), text: []const u8) Error!void {
        try self.countReasoning(text.len);
        try destination.appendSlice(self.arena(), text);
    }

    fn acceptUsage(self: *Reducer, value: std.json.Value) Error!void {
        const fields = try object(value);
        if (try tokenCount(fields, "input_tokens")) |count| self.usage.input = count;
        if (try tokenCount(fields, "output_tokens")) |count| self.usage.output = count;
        if (try tokenCount(fields, "cache_read_input_tokens")) |count| self.usage.cache_read = count;
        if (try tokenCount(fields, "cache_creation_input_tokens")) |count| self.usage.cache_write = count;
        if (nonNull(fields, "output_tokens_details")) |details| {
            if (try tokenCount(try object(details), "thinking_tokens")) |count| self.usage.thinking = count;
        }
    }

    fn finalUsage(self: *const Reducer) types.Usage {
        const usage = self.usage;
        // fx counts every prompt token as input; Anthropic reports cached
        // reads and writes separately from `input_tokens`.
        const input: ?u64 = if (usage.input == null and usage.cache_read == null and usage.cache_write == null)
            null
        else
            (usage.input orelse 0) +| (usage.cache_read orelse 0) +| (usage.cache_write orelse 0);
        return .{ .input_tokens = input, .output_tokens = usage.output, .cache_read_tokens = usage.cache_read, .cache_write_tokens = usage.cache_write, .reasoning_tokens = usage.thinking };
    }

    /// The reply's replay text, in the reducer's arena; null when its tool
    /// call ids cannot be stored.
    fn replayState(self: *Reducer) (Error || std.Io.Writer.Error)!?[]u8 {
        const arena_alloc = self.arena();
        var layout: std.Io.Writer.Allocating = .init(arena_alloc);
        var stored: std.Io.Writer.Allocating = .init(arena_alloc);
        var compaction = false;
        var replayable = true;
        // fx's canonical form is at most one text block, before every tool call.
        var texts: usize = 0;
        var seen_tool = false;
        for (self.blocks.items) |block| {
            const before = stored.written().len;
            const kind: u8 = switch (block.kind) {
                .text => {
                    if (block.text.items.len == 0) continue;
                    // Text after a tool call is out of canonical order too.
                    texts += if (seen_tool) 2 else 1;
                    try separatedEntry(&layout, 't');
                    try layout.writer.print("{d}", .{block.text.items.len});
                    continue;
                },
                .tool_use => {
                    seen_tool = true;
                    for (block.id) |byte| if (byte <= ' ' or byte >= 0x7f) {
                        replayable = false;
                    };
                    try separatedEntry(&layout, 'u');
                    try layout.writer.writeAll(block.id);
                    continue;
                },
                .thinking => kind: {
                    try stored.writer.writeAll("{\"type\":\"thinking\",\"thinking\":");
                    try std.json.Stringify.value(block.text.items, .{}, &stored.writer);
                    try stored.writer.writeAll(",\"signature\":");
                    try std.json.Stringify.value(block.signature.items, .{}, &stored.writer);
                    break :kind 'k';
                },
                .redacted_thinking => kind: {
                    try stored.writer.writeAll("{\"type\":\"redacted_thinking\",\"data\":");
                    try std.json.Stringify.value(block.data, .{}, &stored.writer);
                    break :kind 'k';
                },
                .compaction => kind: {
                    try stored.writer.writeAll("{\"type\":\"compaction\",\"content\":");
                    try std.json.Stringify.value(if (block.summary_present) @as(?[]const u8, block.text.items) else null, .{}, &stored.writer);
                    compaction = compaction or block.summary_present;
                    break :kind if (block.summary_present) 'C' else 'c';
                },
            };
            try stored.writer.writeByte('}');
            try separatedEntry(&layout, kind);
            try layout.writer.print("{d}", .{stored.written().len - before});
        }
        if (compaction) debug_trace.logf("gateway", "anthropic_server_compaction summary_bytes={d}", .{self.reasoning_bytes});
        const prefix: []const u8 = if (compaction) compaction_prefix else &self.prefix;
        const state_len = replay_magic.len + prefix.len + 1 + layout.written().len + 1 + stored.written().len;
        if (state_len > types.ProviderReplay.max_bytes) return error.ReplayTooLarge;
        // Without thinking or compaction, and in the canonical block order,
        // fx's own message rebuilds the reply exactly: no replay is kept.
        if (!replayable or (stored.written().len == 0 and texts <= 1)) return null;
        return try std.mem.concat(arena_alloc, u8, &.{ replay_magic, prefix, "\n", layout.written(), "\n", stored.written() });
    }

    /// The reply as fx keeps it. A stream that ended in an `error` event is a
    /// failed result; one cut off before `message_stop` is an error.
    pub fn finish(self: *Reducer, cancelled: bool) Error!stream_provider.Result {
        errdefer self.phase = .closed;
        if (cancelled) return error.Cancelled;
        if (self.phase != .stopped) return error.IncompleteStream;
        if (self.failure) |failure| {
            self.phase = .closed;
            return .{ .failed = .{ .kind = failure.kind, .detail = try self.alloc.dupe(u8, failure.detail), .ownership = .owned } };
        }
        const raw = self.stop_reason orelse return error.InvalidFinishReason;
        for (self.blocks.items) |block| if (!block.stopped) return error.IncompleteStream;
        const reason = finishReason(raw);
        if (reason == .tool_calls and self.tool_count == 0) return error.InconsistentFinishReason;
        if (reason == .stop and self.tool_count != 0) return error.InconsistentFinishReason;
        if (reason != .length and reason != .content_filter and self.choice == .required and self.tool_count == 0) return error.RequiredToolMissing;

        const arena_alloc = self.arena();
        var text: std.ArrayList(u8) = .empty;
        for (self.blocks.items) |block| {
            if (block.kind != .text or block.text.items.len == 0) continue;
            if (text.items.len != 0) try text.appendSlice(arena_alloc, text_block_separator);
            try text.appendSlice(arena_alloc, block.text.items);
        }
        const provider_state = self.replayState() catch |err| switch (err) {
            error.WriteFailed => return error.OutOfMemory,
            else => |failure| return failure,
        };
        var calls: std.ArrayList(types.ToolCall) = .empty;
        for (self.blocks.items) |block| {
            if (block.kind != .tool_use) continue;
            try calls.append(arena_alloc, .{ .id = block.id, .name = originalName(self.functions, block.name), .arguments_json = if (block.text.items.len == 0) "{}" else block.text.items });
        }
        self.phase = .closed;
        // Everything above lives in the reducer's arena; the result owns copies.
        const borrowed: stream_provider.Result = .{ .completed = .{ .completion = .{
            .content = if (text.items.len != 0) text.items else null,
            .tool_calls = calls.items,
            .generation_id = self.message_id,
            .provider_state_json = provider_state,
            .provider_failure_detail = self.stop_details,
            .finish_reason = reason,
            .finish_reason_raw = .init(raw),
            .usage = self.finalUsage(),
        } } };
        return borrowed.dupe(self.alloc);
    }
};

fn separatedEntry(layout: *std.Io.Writer.Allocating, kind: u8) !void {
    if (layout.written().len != 0) try layout.writer.writeByte(' ');
    try layout.writer.writeByte(kind);
}

fn finishReason(raw: []const u8) types.ProviderFinishReason {
    if (std.mem.eql(u8, raw, "end_turn") or std.mem.eql(u8, raw, "stop_sequence")) return .stop;
    if (std.mem.eql(u8, raw, "tool_use")) return .tool_calls;
    if (std.mem.eql(u8, raw, "max_tokens") or std.mem.eql(u8, raw, "model_context_window_exceeded")) return .length;
    if (std.mem.eql(u8, raw, "refusal")) return .content_filter;
    // pause_turn and compaction ask the caller to send the reply back as is;
    // fx keeps the reply and the user continues the turn.
    return .other;
}

/// Maps an Anthropic error type to the failure kind fx's recovery reads.
pub fn errorKind(name: []const u8) stream_provider.FailureKind {
    const table = [_]struct { []const u8, stream_provider.FailureKind }{
        .{ "invalid_request_error", .invalid_request },
        .{ "authentication_error", .unauthorized },
        .{ "permission_error", .forbidden },
        .{ "not_found_error", .invalid_request },
        .{ "request_too_large", .request_too_large },
        .{ "rate_limit_error", .rate_limited },
        .{ "api_error", .server_error },
        .{ "overloaded_error", .unavailable },
        .{ "timeout_error", .gateway_timeout },
    };
    for (table) |entry| if (std.mem.eql(u8, entry[0], name)) return entry[1];
    return .provider_error;
}

fn object(value: std.json.Value) Error!std.json.ObjectMap {
    return if (value == .object) value.object else error.InvalidChunk;
}

fn string(value: std.json.Value) Error![]const u8 {
    return if (value == .string) value.string else error.InvalidChunk;
}

fn nonNull(fields: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const value = fields.get(key) orelse return null;
    return if (value == .null) null else value;
}

fn integer(value: std.json.Value) Error!i64 {
    return switch (value) {
        .integer => value.integer,
        .number_string => std.fmt.parseInt(i64, value.number_string, 10) catch error.InvalidChunk,
        else => error.InvalidChunk,
    };
}

fn indexValue(value: std.json.Value) Error!usize {
    return std.math.cast(usize, try integer(value)) orelse error.InvalidChunk;
}

fn tokenCount(fields: std.json.ObjectMap, key: []const u8) Error!?u64 {
    const value = nonNull(fields, key) orelse return null;
    return std.math.cast(u64, try integer(value)) orelse error.InvalidChunk;
}

/// Thin stream consumer, not a network transport. Stops at `message_stop`
/// (or an `error` event) without waiting for EOF.
pub fn consume_stream(
    alloc: Allocator,
    source: *std.Io.Reader,
    request: stream_provider.RequestData,
    /// The body that was sent, from which the reply's prefix comes.
    body: []const u8,
    limits: Limits,
    events: ?stream_provider.EventSink,
    cancel_flag: *const std.atomic.Value(bool),
    progress: ?sse.Progress,
) Error!stream_provider.Result {
    if (cancel_flag.load(.seq_cst)) return error.Cancelled;
    var reducer = try Reducer.init(alloc, request, bodyPrefix(body), limits);
    defer reducer.deinit();
    var framing = sse.Reader{ .max_event_bytes = limits.event_bytes, .progress = progress };
    defer framing.deinit(alloc);
    while (true) {
        const remaining = limits.total_wire_bytes - framing.total_bytes;
        const event_limited = limits.event_bytes < remaining;
        framing.max_total_bytes = framing.total_bytes + @min(limits.event_bytes, remaining);
        const data = framing.next(alloc, source, cancel_flag) catch |err| switch (err) {
            error.StreamTooLarge => return if (event_limited) error.EventTooLarge else error.StreamTooLarge,
            else => return err,
        };
        const chunk = data orelse return error.IncompleteStream;
        const deltas = try reducer.accept(chunk, cancel_flag.load(.seq_cst));
        if (events) |sink| {
            if (cancel_flag.load(.seq_cst)) return error.Cancelled;
            if (deltas.reasoning) |text| sink.emit(.{ .reasoning_delta = text });
            if (deltas.content) |text| {
                if (deltas.separated) sink.emit(.{ .content_delta = text_block_separator });
                sink.emit(.{ .content_delta = text });
            }
            if (deltas.tool_started) |tool| sink.emit(.{ .tool_started = .{ .id = tool.id, .name = tool.name } });
            if (deltas.tool_input) |text| sink.emit(.{ .tool_input_delta = text });
        }
        if (reducer.phase == .stopped) return reducer.finish(cancel_flag.load(.seq_cst));
    }
}

// ---------------------------------------------------------------------------
// Tests

const test_functions = [_]model_tool_schema.FunctionSchema{
    .{ .name = "read_file", .description = "Read a file", .input_schema = .{ .properties = &.{.{ .name = "path", .json_type = .string }}, .required = &.{"path"} } },
    .{ .name = "skill.run", .description = "Run a skill" },
};
const test_provider: model_provider.ProviderId = .gateway;
const test_options: Options = .{};

fn testSettings(options: *const Options) Settings {
    return .{ .options = options, .provider = &test_provider };
}

fn testRequest(model: []const u8, messages: []const types.ChatMessage) stream_provider.RequestData {
    return .{
        .model = model,
        .instructions = &.{ .{ .role = .system, .content = "You are fx." }, .{ .role = .system, .content = "Workspace rules." } },
        .messages = messages,
        .tools = .{ .additional_functions = &test_functions },
        .tool_choice = .auto,
        .provider_options = .{},
        .max_output_tokens = 1024,
    };
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.find(u8, haystack, needle) == null) {
        std.debug.print("\nmissing: {s}\nin: {s}\n", .{ needle, haystack });
        return error.TestExpectedContains;
    }
}

fn parseBody(alloc: Allocator, body: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, alloc, body, .{});
}

test "anthropic request maps system, tools, effort, thinking and caching" {
    const alloc = std.testing.allocator;
    var request = testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "hello" }});
    request.provider_options.reasoning = types.ReasoningEffort.literal("high");
    request.provider_options.parallel_tool_calls = false;
    var options: Options = .{ .strict_tools = "read_file" };
    const body = try build_request(alloc, request, testSettings(&options));
    defer alloc.free(body);
    var parsed = try parseBody(alloc, body);
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("claude-haiku-5-5", root.get("model").?.string);
    try std.testing.expectEqual(@as(i64, 1024), root.get("max_tokens").?.integer);
    try std.testing.expect(root.get("stream").?.bool);
    try std.testing.expect(root.get("temperature") == null and root.get("top_p") == null and root.get("top_k") == null);
    try expectContains(body, "\"thinking\":{\"type\":\"adaptive\",\"display\":\"omitted\",\"block_binding\":{\"prefix_mismatch_behavior\":\"drop_block\"}}");
    try expectContains(body, "\"output_config\":{\"effort\":\"high\"}");
    try expectContains(body, ",\"cache_control\":{\"type\":\"ephemeral\"},");
    const system = root.get("system").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), system.len);
    try std.testing.expect(system[0].object.get("cache_control") == null);
    try std.testing.expect(system[1].object.get("cache_control") != null);
    const tools = root.get("tools").?.array.items;
    try std.testing.expectEqualStrings("read_file", tools[0].object.get("name").?.string);
    try std.testing.expect(tools[0].object.get("strict").?.bool);
    try std.testing.expect(!tools[0].object.get("input_schema").?.object.get("additionalProperties").?.bool);
    try std.testing.expectEqualStrings("object", tools[0].object.get("input_schema").?.object.get("type").?.string);
    // Dots are not allowed in Anthropic tool names.
    try std.testing.expectEqualStrings("skill__run", tools[1].object.get("name").?.string);
    try std.testing.expect(tools[1].object.get("strict") == null);
    try expectContains(body, "\"tool_choice\":{\"type\":\"auto\",\"disable_parallel_tool_use\":true}");
    try expectContains(body, "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hello\"}]}]");

    for ([_][]const u8{ "low", "medium", "max" }) |level| {
        var leveled = testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "x" }});
        leveled.provider_options.reasoning = .{ .named = .{ .bytes = undefined, .len = 0 } };
        leveled.provider_options.reasoning = if (std.mem.eql(u8, level, "low")) types.ReasoningEffort.literal("low") else if (std.mem.eql(u8, level, "medium")) types.ReasoningEffort.literal("medium") else types.ReasoningEffort.literal("max");
        const leveled_body = try build_request(alloc, leveled, testSettings(&test_options));
        defer alloc.free(leveled_body);
        const expected = try std.mem.concat(alloc, u8, &.{ "\"output_config\":{\"effort\":\"", level, "\"}" });
        defer alloc.free(expected);
        try expectContains(leveled_body, expected);
    }

    var plain_options: Options = .{ .prompt_caching = false, .prefix_mismatch_behavior = .unset, .thinking_display = .omitted };
    const unknown = try build_request(alloc, testRequest("claude-custom", &.{.{ .role = .user, .content = "x" }}), testSettings(&plain_options));
    defer alloc.free(unknown);
    try std.testing.expect(std.mem.find(u8, unknown, "thinking") == null);
    try std.testing.expect(std.mem.find(u8, unknown, "cache_control") == null);
}

test "anthropic forced tool choice follows each model's rule and none keeps the tool set" {
    const alloc = std.testing.allocator;
    const cases = [_]struct { model: []const u8, choice: types.ToolChoice, wire: []const u8 }{
        .{ .model = "claude-sonnet-5-5", .choice = .required, .wire = "\"tool_choice\":{\"type\":\"auto\"}" },
        .{ .model = "claude-opus-5-5", .choice = .required, .wire = "\"tool_choice\":{\"type\":\"auto\"}" },
        .{ .model = "claude-haiku-5-5", .choice = .required, .wire = "\"tool_choice\":{\"type\":\"any\"}" },
        .{ .model = "claude-haiku-5-5", .choice = .none, .wire = "\"tool_choice\":{\"type\":\"none\"}" },
    };
    for (cases) |case| {
        var request = testRequest(case.model, &.{.{ .role = .user, .content = "x" }});
        request.tool_choice = case.choice;
        const body = try build_request(alloc, request, testSettings(&test_options));
        defer alloc.free(body);
        try expectContains(body, case.wire);
        try expectContains(body, "\"name\":\"read_file\"");
    }
}

test "anthropic parallel tool results travel in one user message after the calls" {
    const alloc = std.testing.allocator;
    const calls = [_]types.ToolCall{
        .{ .id = "toolu_1", .name = "read_file", .arguments_json = "{\"path\":\"a\"}" },
        .{ .id = "toolu_2", .name = "skill.run", .arguments_json = "" },
    };
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "read both" },
        .{ .role = .assistant, .content = "Reading.", .tool_calls = &calls },
        .{ .role = .tool, .tool_call_id = "toolu_1", .tool_name = "read_file", .content = "A" },
        .{ .role = .tool, .tool_call_id = "toolu_2", .tool_name = "skill.run", .content = "boom", .tool_result_status = .failure },
        .{ .role = .user, .content = "steer: also check c" },
    };
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &messages), testSettings(&test_options));
    defer alloc.free(body);
    var parsed = try parseBody(alloc, body);
    defer parsed.deinit();
    const wire = parsed.value.object.get("messages").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), wire.len);
    try std.testing.expectEqualStrings("assistant", wire[1].object.get("role").?.string);
    const assistant = wire[1].object.get("content").?.array.items;
    try std.testing.expectEqualStrings("tool_use", assistant[1].object.get("type").?.string);
    try std.testing.expectEqualStrings("skill__run", assistant[2].object.get("name").?.string);
    try std.testing.expectEqual(@as(usize, 0), assistant[2].object.get("input").?.object.count());
    const results = wire[2].object.get("content").?.array.items;
    try std.testing.expectEqual(@as(usize, 3), results.len);
    try std.testing.expectEqualStrings("tool_result", results[0].object.get("type").?.string);
    try std.testing.expectEqualStrings("toolu_1", results[0].object.get("tool_use_id").?.string);
    try std.testing.expectEqualStrings("tool_result", results[1].object.get("type").?.string);
    try std.testing.expect(results[1].object.get("is_error").?.bool);
    try std.testing.expectEqualStrings("text", results[2].object.get("type").?.string);
}

test "anthropic user images become base64 image blocks before the text" {
    const alloc = std.testing.allocator;
    const png = "\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x00\x01\xff\x89\x99=\x1d\x00\x00\x00\x00IEND\xaeB`\x82";
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(png, &digest, .{});
    var hex = std.fmt.bytesToHex(digest, .lower);
    const image: types.ImageAttachment = .{ .path = @constCast("pixel.png"), .media_type = @constCast("image/png"), .inline_data = @constCast(png), .snapshot_sha256 = &hex };
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "what color?", .images = &.{image} }};
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &messages), testSettings(&test_options));
    defer alloc.free(body);
    try expectContains(body, "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"iVBORw0K");
    try expectContains(body, "\"}},{\"type\":\"text\",\"text\":\"what color?\"}");
}

test "anthropic context management goes out as configured, with its betas" {
    const alloc = std.testing.allocator;
    const edits = "{\"edits\":[{\"type\":\"clear_tool_uses_20250919\",\"keep\":{\"type\":\"tool_uses\",\"value\":1}},{\"type\":\"compact_20260112\",\"instructions\":\"Keep open tasks.\"}]}";
    var options: Options = .{ .context_management = edits };
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "x" }}), testSettings(&options));
    defer alloc.free(body);
    try expectContains(body, ",\"context_management\":" ++ edits);
    const beta = (try betaHeader(alloc, &options, "claude-haiku-5-5")).?;
    defer alloc.free(beta);
    try std.testing.expectEqualStrings(thinking_binding_beta ++ "," ++ compaction_beta ++ "," ++ context_management_beta, beta);
}

fn testStream(alloc: Allocator, request: stream_provider.RequestData, wire: []const u8, sink: ?stream_provider.EventSink) !stream_provider.Result {
    const body = try build_request(alloc, request, testSettings(&test_options));
    defer alloc.free(body);
    var source = std.Io.Reader.fixed(wire);
    var cancel = std.atomic.Value(bool).init(false);
    return consume_stream(alloc, &source, request, body, .{}, sink, &cancel, null);
}

inline fn sseEvent(comptime name: []const u8, comptime data: []const u8) []const u8 {
    return "event: " ++ name ++ "\ndata: " ++ data ++ "\n\n";
}

const test_thinking_tool_stream =
    sseEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-5-5\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"cache_creation_input_tokens\":100,\"cache_read_input_tokens\":2000,\"output_tokens\":1}}}") ++
    sseEvent("ping", "{\"type\":\"ping\"}") ++
    sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}") ++
    sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"Need the file.\"}}") ++
    sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"c2lnbmVk\"}}") ++
    sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}") ++
    sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"redacted_thinking\",\"data\":\"b3BhcXVl\"}}") ++
    sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":1}") ++
    sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}") ++
    sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"Let me \"}}") ++
    sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":2,\"delta\":{\"type\":\"text_delta\",\"text\":\"look.\"}}") ++
    sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":2}") ++
    sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":3,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_A\",\"name\":\"read_file\",\"input\":{}}}") ++
    sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"path\\\":\"}}") ++
    sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":3,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\" \\\"a.txt\\\"}\"}}") ++
    sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":3}") ++
    sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":4,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_B\",\"name\":\"skill__run\",\"input\":{}}}") ++
    sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":4}") ++
    sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":42,\"output_tokens_details\":{\"thinking_tokens\":30}}}") ++
    sseEvent("message_stop", "{\"type\":\"message_stop\"}");

const EventLog = struct {
    alloc: Allocator,
    content: std.ArrayList(u8) = .empty,
    reasoning: std.ArrayList(u8) = .empty,
    tools: usize = 0,
    tool_input: std.ArrayList(u8) = .empty,

    fn emit(raw: *anyopaque, event: stream_provider.Event) void {
        const self: *EventLog = @ptrCast(@alignCast(raw));
        switch (event) {
            .content_delta => |text| self.content.appendSlice(self.alloc, text) catch unreachable,
            .reasoning_delta => |text| self.reasoning.appendSlice(self.alloc, text) catch unreachable,
            .tool_started => self.tools += 1,
            .tool_input_delta => |text| self.tool_input.appendSlice(self.alloc, text) catch unreachable,
        }
    }

    fn deinit(self: *EventLog) void {
        self.content.deinit(self.alloc);
        self.reasoning.deinit(self.alloc);
        self.tool_input.deinit(self.alloc);
    }
};

test "anthropic stream parses thinking, signature, text, tools, usage and keeps the blocks for replay" {
    const alloc = std.testing.allocator;
    const request = testRequest("claude-sonnet-5-5", &.{.{ .role = .user, .content = "read a.txt" }});
    var log: EventLog = .{ .alloc = alloc };
    defer log.deinit();
    var result = try testStream(alloc, request, test_thinking_tool_stream, .{ .context = &log, .emit_fn = EventLog.emit });
    defer result.deinit(alloc);
    const completion = result.completed.completion;
    try std.testing.expectEqualStrings("Let me look.", completion.content.?);
    try std.testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
    try std.testing.expectEqualStrings("tool_use", completion.finish_reason_raw.slice());
    try std.testing.expectEqualStrings("msg_1", completion.generation_id.?);
    try std.testing.expectEqual(@as(usize, 2), completion.tool_calls.len);
    try std.testing.expectEqualStrings("toolu_A", completion.tool_calls[0].id);
    try std.testing.expectEqualStrings("{\"path\": \"a.txt\"}", completion.tool_calls[0].arguments_json);
    try std.testing.expectEqualStrings("skill.run", completion.tool_calls[1].name);
    try std.testing.expectEqualStrings("{}", completion.tool_calls[1].arguments_json);
    try std.testing.expectEqual(@as(?u64, 2110), completion.usage.input_tokens);
    try std.testing.expectEqual(@as(?u64, 2000), completion.usage.cache_read_tokens);
    try std.testing.expectEqual(@as(?u64, 100), completion.usage.cache_write_tokens);
    try std.testing.expectEqual(@as(?u64, 42), completion.usage.output_tokens);
    try std.testing.expectEqual(@as(?u64, 30), completion.usage.reasoning_tokens);
    try std.testing.expectEqualStrings("Let me look.", log.content.items);
    try std.testing.expectEqualStrings("Need the file.", log.reasoning.items);
    try std.testing.expectEqual(@as(usize, 2), log.tools);
    const state = completion.provider_state_json.?;
    try expectContains(state, "{\"type\":\"thinking\",\"thinking\":\"Need the file.\",\"signature\":\"c2lnbmVk\"}");
    try expectContains(state, "{\"type\":\"redacted_thinking\",\"data\":\"b3BhcXVl\"}");
    // Text and tool calls come back from fx's own message; the replay keeps
    // only where they sat.
    try expectContains(state, " t12 utoolu_A utoolu_B\n");
}

test "anthropic tool roundtrip replays thinking unchanged and drops it only when the prefix changed" {
    const alloc = std.testing.allocator;
    const first_messages = [_]types.ChatMessage{.{ .role = .user, .content = "read a.txt" }};
    const first_request = testRequest("claude-sonnet-5-5", &first_messages);
    var result = try testStream(alloc, first_request, test_thinking_tool_stream, null);
    defer result.deinit(alloc);
    const completion = result.completed.completion;
    const replay: types.ProviderReplay = .{ .source = .{ .provider = test_provider, .model = "claude-sonnet-5-5" }, .parts_json = completion.provider_state_json.? };
    const second_messages = [_]types.ChatMessage{
        first_messages[0],
        .{ .role = .assistant, .content = completion.content, .tool_calls = completion.tool_calls, .provider_replay = replay },
        .{ .role = .tool, .tool_call_id = "toolu_A", .tool_name = "read_file", .content = "file a" },
        .{ .role = .tool, .tool_call_id = "toolu_B", .tool_name = "skill.run", .content = "ran" },
    };
    const second = try build_request(alloc, testRequest("claude-sonnet-5-5", &second_messages), testSettings(&test_options));
    defer alloc.free(second);
    try expectContains(second, "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"Need the file.\",\"signature\":\"c2lnbmVk\"},{\"type\":\"redacted_thinking\",\"data\":\"b3BhcXVl\"},{\"type\":\"text\",\"text\":\"Let me look.\"},{\"type\":\"tool_use\",\"id\":\"toolu_A\",\"name\":\"read_file\",\"input\":{\"path\": \"a.txt\"}},{\"type\":\"tool_use\",\"id\":\"toolu_B\",\"name\":\"skill__run\",\"input\":{}}]}");
    try expectContains(second, "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_A\",\"content\":[{\"type\":\"text\",\"text\":\"file a\"}]},{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_B\"");
    // The same history serializes to the same bytes every time.
    const again = try build_request(alloc, testRequest("claude-sonnet-5-5", &second_messages), testSettings(&test_options));
    defer alloc.free(again);
    try std.testing.expectEqualStrings(second, again);

    // An earlier turn that changed (a keep-tail compaction rewrote it) leaves
    // the reply's thinking out, but keeps its text and calls.
    var edited = second_messages;
    edited[0] = .{ .role = .user, .content = "Summary of earlier work. read a.txt" };
    const stripped = try build_request(alloc, testRequest("claude-sonnet-5-5", &edited), testSettings(&test_options));
    defer alloc.free(stripped);
    try std.testing.expect(std.mem.find(u8, stripped, "\"type\":\"thinking\"") == null);
    try std.testing.expect(std.mem.find(u8, stripped, "redacted_thinking") == null);
    try expectContains(stripped, "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Let me look.\"},{\"type\":\"tool_use\",\"id\":\"toolu_A\"");

    // Another model's or route's replay is ignored.
    var other = second_messages;
    other[1].provider_replay = .{ .source = .{ .provider = .codex, .model = "claude-sonnet-5-5" }, .parts_json = replay.parts_json };
    const canonical = try build_request(alloc, testRequest("claude-sonnet-5-5", &other), testSettings(&test_options));
    defer alloc.free(canonical);
    try std.testing.expect(std.mem.find(u8, canonical, "signature") == null);

    // fx changed the reply's text: the canonical form is sent.
    var rewritten = second_messages;
    rewritten[1].content = "Different text.";
    const rewritten_body = try build_request(alloc, testRequest("claude-sonnet-5-5", &rewritten), testSettings(&test_options));
    defer alloc.free(rewritten_body);
    try std.testing.expect(std.mem.find(u8, rewritten_body, "signature") == null);
    try expectContains(rewritten_body, "Different text.");
}

test "anthropic replay projection keeps thinking with the kept calls" {
    const alloc = std.testing.allocator;
    var result = try testStream(alloc, testRequest("claude-sonnet-5-5", &.{.{ .role = .user, .content = "x" }}), test_thinking_tool_stream, null);
    defer result.deinit(alloc);
    const completion = result.completed.completion;
    const replay: types.ProviderReplay = .{ .source = .{ .provider = test_provider, .model = "claude-sonnet-5-5" }, .parts_json = completion.provider_state_json.? };
    const same = (try project_replay(alloc, replay, completion.tool_calls, true, true)).?;
    try std.testing.expect(same.parts_json.ptr == replay.parts_json.ptr);
    try std.testing.expect(try project_replay(alloc, replay, completion.tool_calls, true, false) == null);
    const one = (try project_replay(alloc, replay, completion.tool_calls[0..1], false, true)).?;
    defer alloc.free(one.parts_json);
    try expectContains(one.parts_json, "\"signature\":\"c2lnbmVk\"");
    try std.testing.expect(std.mem.find(u8, one.parts_json, "toolu_B") == null);
    try std.testing.expect(std.mem.find(u8, one.parts_json, "t12") == null);
    try std.testing.expect(try project_replay(alloc, replay, &.{.{ .id = "other", .name = "read_file", .arguments_json = "{}" }}, true, true) == null);
}

test "anthropic stop reasons map to fx finish reasons" {
    const alloc = std.testing.allocator;
    const request = testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "x" }});
    const start = sseEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_2\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}");
    const text = sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"partial\"}}") ++
        sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}");
    const stop = sseEvent("message_stop", "{\"type\":\"message_stop\"}");
    const cases = [_]struct { wire: []const u8, reason: types.ProviderFinishReason, raw: []const u8 }{
        .{ .wire = start ++ text ++ sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":16}}") ++ stop, .reason = .length, .raw = "max_tokens" },
        .{ .wire = start ++ text ++ sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":3}}") ++ stop, .reason = .stop, .raw = "end_turn" },
        .{ .wire = start ++ sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"refusal\",\"stop_details\":{\"type\":\"refusal\",\"category\":\"cyber\",\"explanation\":\"declined\"}},\"usage\":{\"output_tokens\":0}}") ++ stop, .reason = .content_filter, .raw = "refusal" },
        .{ .wire = start ++ text ++ sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"pause_turn\"},\"usage\":{\"output_tokens\":3}}") ++ stop, .reason = .other, .raw = "pause_turn" },
    };
    for (cases) |case| {
        var result = try testStream(alloc, request, case.wire, null);
        defer result.deinit(alloc);
        const completion = result.completed.completion;
        try std.testing.expectEqual(case.reason, completion.finish_reason.?);
        try std.testing.expectEqualStrings(case.raw, completion.finish_reason_raw.slice());
        if (case.reason == .content_filter) {
            try expectContains(completion.provider_failure_detail.?, "\"category\":\"cyber\"");
            try std.testing.expectEqual(types.ProviderCompletionDisposition.provider_failure, types.classifyProviderCompletion(completion));
        }
        if (case.reason == .length) try std.testing.expectEqual(types.ProviderCompletionDisposition.length_limited, types.classifyProviderCompletion(completion));
    }
}

test "anthropic stream errors fail clearly" {
    const alloc = std.testing.allocator;
    const request = testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "x" }});
    const start = sseEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_3\",\"usage\":{\"input_tokens\":5}}}");
    var overloaded = try testStream(alloc, request, start ++ sseEvent("error", "{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}"), null);
    defer overloaded.deinit(alloc);
    try std.testing.expectEqual(stream_provider.FailureKind.unavailable, overloaded.failed.kind);
    try expectContains(overloaded.failed.detail.?, "Overloaded");

    const failures = [_]struct { wire: []const u8, err: anyerror }{
        .{ .wire = start, .err = error.IncompleteStream },
        .{ .wire = start ++ sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}"), .err = error.InvalidChunk },
        .{ .wire = sseEvent("message_start", "{not json"), .err = error.MalformedAnthropicEvent },
        .{ .wire = start ++ sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}") ++ sseEvent("message_stop", "{\"type\":\"message_stop\"}"), .err = error.InconsistentFinishReason },
        .{ .wire = start ++ sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}") ++ sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}") ++ sseEvent("message_stop", "{\"type\":\"message_stop\"}"), .err = error.IncompleteStream },
        .{ .wire = start ++ sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}") ++ sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"x\"}}"), .err = error.InvalidChunk },
    };
    for (failures) |case| try std.testing.expectError(case.err, testStream(alloc, request, case.wire, null));
    try std.testing.expectEqual(stream_provider.FailureKind.invalid_request, errorKind("not_found_error"));
    try std.testing.expectEqual(stream_provider.FailureKind.rate_limited, errorKind("rate_limit_error"));
}

test "anthropic server compaction block is kept and later requests start at it" {
    const alloc = std.testing.allocator;
    const wire = sseEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_c\",\"usage\":{\"input_tokens\":900,\"output_tokens\":0,\"iterations\":[]}}}") ++
        sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"compaction\",\"content\":null}}") ++
        sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"compaction_delta\",\"content\":\"Summary: task T open.\"}}") ++
        sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}") ++
        sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"\"}}") ++
        sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"c2ln\"}}") ++
        sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":1}") ++
        sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"Done.\"}}") ++
        sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":2}") ++
        sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}") ++
        sseEvent("message_stop", "{\"type\":\"message_stop\"}");
    const history = [_]types.ChatMessage{
        .{ .role = .user, .content = "old turn" },
        .{ .role = .assistant, .content = "old answer" },
        .{ .role = .user, .content = "continue" },
    };
    var result = try testStream(alloc, testRequest("claude-haiku-5-5", &history), wire, null);
    defer result.deinit(alloc);
    const completion = result.completed.completion;
    try std.testing.expect(std.mem.startsWith(u8, completion.provider_state_json.?, "fxa1 compaction\n"));
    try expectContains(completion.provider_state_json.?, "{\"type\":\"compaction\",\"content\":\"Summary: task T open.\"}");
    const next = history ++ [_]types.ChatMessage{
        .{ .role = .assistant, .content = "Done.", .provider_replay = .{ .source = .{ .provider = test_provider, .model = "claude-haiku-5-5" }, .parts_json = completion.provider_state_json.? } },
        .{ .role = .user, .content = "next" },
    };
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &next), testSettings(&test_options));
    defer alloc.free(body);
    try std.testing.expect(std.mem.find(u8, body, "old turn") == null);
    try expectContains(body, "\"messages\":[{\"role\":\"assistant\",\"content\":[{\"type\":\"compaction\",\"content\":\"Summary: task T open.\"},{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"c2ln\"},{\"type\":\"text\",\"text\":\"Done.\"}]},{\"role\":\"user\"");
}

test "anthropic codec releases every allocation on failure" {
    try std.testing.checkAllAllocationFailures(testing_allocator.no_resize, struct {
        fn run(alloc: Allocator) !void {
            const request = testRequest("claude-sonnet-5-5", &.{.{ .role = .user, .content = "x" }});
            var result = try testStream(alloc, request, test_thinking_tool_stream, null);
            defer result.deinit(alloc);
            const completion = result.completed.completion;
            const messages = [_]types.ChatMessage{
                .{ .role = .user, .content = "x" },
                .{ .role = .assistant, .content = completion.content, .tool_calls = completion.tool_calls, .provider_replay = .{ .source = .{ .provider = test_provider, .model = "claude-sonnet-5-5" }, .parts_json = completion.provider_state_json.? } },
                .{ .role = .tool, .tool_call_id = "toolu_A", .tool_name = "read_file", .content = "a" },
                .{ .role = .tool, .tool_call_id = "toolu_B", .tool_name = "skill.run", .content = "b" },
            };
            const body = try build_request(alloc, testRequest("claude-sonnet-5-5", &messages), testSettings(&test_options));
            alloc.free(body);
        }
    }.run, .{});
}

test "anthropic notes fx adds after a turn's final reply move behind it and keep its thinking" {
    const alloc = std.testing.allocator;
    const call = [_]types.ToolCall{.{ .id = "toolu_1", .name = "read_file", .arguments_json = "{\"path\":\"a\"}" }};
    const turn = [_]types.ChatMessage{
        .{ .role = .user, .content = "read a" },
        .{ .role = .assistant, .tool_calls = &call },
        .{ .role = .tool, .tool_call_id = "toolu_1", .tool_name = "read_file", .content = "A" },
    };
    const final_stream = sseEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_f\",\"usage\":{\"input_tokens\":5}}}") ++
        sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"weigh it\",\"signature\":\"\"}}") ++
        sseEvent("content_block_delta", "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"c2lnZg==\"}}") ++
        sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}") ++
        sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"It says A.\"}}") ++
        sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":1}") ++
        sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}") ++
        sseEvent("message_stop", "{\"type\":\"message_stop\"}");
    var result = try testStream(alloc, testRequest("claude-sonnet-5-5", &turn), final_stream, null);
    defer result.deinit(alloc);
    const replay: types.ProviderReplay = .{ .source = .{ .provider = test_provider, .model = "claude-sonnet-5-5" }, .parts_json = result.completed.completion.provider_state_json.? };
    const next = turn ++ [_]types.ChatMessage{
        .{ .role = .user, .content = "Session file evidence: a read" },
        .{ .role = .assistant, .content = "It says A.", .provider_replay = replay },
        .{ .role = .user, .content = "and b?" },
    };
    const body = try build_request(alloc, testRequest("claude-sonnet-5-5", &next), testSettings(&test_options));
    defer alloc.free(body);
    try expectContains(body, "{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"toolu_1\",\"content\":[{\"type\":\"text\",\"text\":\"A\"}]}]},{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"weigh it\",\"signature\":\"c2lnZg==\"},{\"type\":\"text\",\"text\":\"It says A.\"}]},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Session file evidence: a read\"},{\"type\":\"text\",\"text\":\"and b?\"}]}");

    // An earlier edit that moving the note cannot undo keeps the note in place
    // and leaves the thinking out.
    var edited = next;
    edited[0] = .{ .role = .user, .content = "read a, edited" };
    const stripped = try build_request(alloc, testRequest("claude-sonnet-5-5", &edited), testSettings(&test_options));
    defer alloc.free(stripped);
    try std.testing.expect(std.mem.find(u8, stripped, "weigh it") == null);
    try expectContains(stripped, "\"content\":[{\"type\":\"text\",\"text\":\"A\"}]},{\"type\":\"text\",\"text\":\"Session file evidence: a read\"}]}");
}

test "anthropic thinking survives a change of effort and earlier replay formats are ignored" {
    const alloc = std.testing.allocator;
    const first_messages = [_]types.ChatMessage{.{ .role = .user, .content = "read a.txt" }};
    var first_request = testRequest("claude-sonnet-5-5", &first_messages);
    first_request.provider_options.reasoning = types.ReasoningEffort.literal("low");
    var result = try testStream(alloc, first_request, test_thinking_tool_stream, null);
    defer result.deinit(alloc);
    const completion = result.completed.completion;
    const source: model_provider.ProviderSelection = .{ .provider = test_provider, .model = "claude-sonnet-5-5" };
    var next = [_]types.ChatMessage{
        first_messages[0],
        .{ .role = .assistant, .content = completion.content, .tool_calls = completion.tool_calls, .provider_replay = .{ .source = source, .parts_json = completion.provider_state_json.? } },
        .{ .role = .tool, .tool_call_id = "toolu_A", .tool_name = "read_file", .content = "file a" },
        .{ .role = .tool, .tool_call_id = "toolu_B", .tool_name = "skill.run", .content = "ran" },
    };
    var raised = testRequest("claude-sonnet-5-5", &next);
    raised.provider_options.reasoning = types.ReasoningEffort.literal("high");
    const body = try build_request(alloc, raised, testSettings(&test_options));
    defer alloc.free(body);
    try expectContains(body, "\"signature\":\"c2lnbmVk\"");
    try expectContains(body, "\"output_config\":{\"effort\":\"high\"}");

    // A replay in an earlier format is ignored: the reply goes out in its
    // canonical form, without thinking.
    next[1].provider_replay = .{ .source = source, .parts_json = "{\"prefix\":\"x\",\"content\":[],\"_tool_call_ids\":[]}" };
    const legacy_body = try build_request(alloc, testRequest("claude-sonnet-5-5", &next), testSettings(&test_options));
    defer alloc.free(legacy_body);
    try std.testing.expect(std.mem.find(u8, legacy_body, "signature") == null);
    try expectContains(legacy_body, "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"Let me look.\"},{\"type\":\"tool_use\",\"id\":\"toolu_A\"");
}

test "anthropic replies without thinking go back in their own block order" {
    const alloc = std.testing.allocator;
    const start = sseEvent("message_start", "{\"type\":\"message_start\",\"message\":{\"id\":\"msg_o\",\"usage\":{\"input_tokens\":5}}}");
    const text0 = sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"First.\"}}") ++ sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":0}");
    const tool1 = sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_o\",\"name\":\"read_file\",\"input\":{\"path\":\"a\"}}}") ++ sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":1}");
    const text2 = sseEvent("content_block_start", "{\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"text\",\"text\":\"Then.\"}}") ++ sseEvent("content_block_stop", "{\"type\":\"content_block_stop\",\"index\":2}");
    const stop = sseEvent("message_delta", "{\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}") ++ sseEvent("message_stop", "{\"type\":\"message_stop\"}");
    const request = testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "go" }});

    var plain = try testStream(alloc, request, start ++ text0 ++ tool1 ++ stop, null);
    defer plain.deinit(alloc);
    try std.testing.expect(plain.completed.completion.provider_state_json == null);

    var mixed = try testStream(alloc, request, start ++ text0 ++ tool1 ++ text2 ++ stop, null);
    defer mixed.deinit(alloc);
    const completion = mixed.completed.completion;
    try std.testing.expectEqualStrings("First.\n\nThen.", completion.content.?);
    const next = [_]types.ChatMessage{
        .{ .role = .user, .content = "go" },
        .{ .role = .assistant, .content = completion.content, .tool_calls = completion.tool_calls, .provider_replay = .{ .source = .{ .provider = test_provider, .model = "claude-haiku-5-5" }, .parts_json = completion.provider_state_json.? } },
        .{ .role = .tool, .tool_call_id = "toolu_o", .tool_name = "read_file", .content = "A" },
    };
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &next), testSettings(&test_options));
    defer alloc.free(body);
    try expectContains(body, "{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"First.\"},{\"type\":\"tool_use\",\"id\":\"toolu_o\",\"name\":\"read_file\",\"input\":{\"path\":\"a\"}},{\"type\":\"text\",\"text\":\"Then.\"}]}");
}
