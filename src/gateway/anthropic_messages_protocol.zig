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
const Sha256 = std.crypto.hash.sha2.Sha256;

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
    for (options.betas) |beta| try appendBeta(alloc, &out, beta);
    if (sendsBlockBinding(options, model)) try appendBeta(alloc, &out, thinking_binding_beta);
    if (options.compaction != null) try appendBeta(alloc, &out, compaction_beta);
    if (options.context_editing != null) try appendBeta(alloc, &out, context_management_beta);
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
    var buffer: [2 * max_name_bytes]u8 = undefined;
    const wire = try wireName(&buffer, function.name);
    if (functions.items.len == max_selected_tools) return error.TooManyTools;
    for (functions.items) |prior| {
        var prior_buffer: [2 * max_name_bytes]u8 = undefined;
        if (std.mem.eql(u8, try wireName(&prior_buffer, prior.name), wire)) return error.InvalidToolSelection;
    }
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

/// A copy of `schema` with `additionalProperties: false` on every object
/// schema that does not set it.
fn closeObjects(arena: Allocator, schema: std.json.Value) Allocator.Error!std.json.Value {
    switch (schema) {
        .object => |fields| {
            var copy: std.json.ObjectMap = .empty;
            var iterator = fields.iterator();
            while (iterator.next()) |entry| try copy.put(arena, entry.key_ptr.*, try closeObjects(arena, entry.value_ptr.*));
            const kind = fields.get("type");
            const is_object = kind != null and kind.? == .string and std.mem.eql(u8, kind.?.string, "object");
            if (is_object and !fields.contains("additionalProperties")) try copy.put(arena, "additionalProperties", .{ .bool = false });
            return .{ .object = copy };
        },
        .array => |items| {
            var copy = std.json.Array.init(arena);
            for (items.items) |item| try copy.append(try closeObjects(arena, item));
            return .{ .array = copy };
        },
        else => return schema,
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
// Replay. An assistant reply keeps its content blocks exactly as the API sent
// them, so thinking blocks and their signatures go back unchanged:
//   {"prefix":"<hex>","content":[...blocks...],"_tool_call_ids":[...]}
// `prefix` is the digest of the system prompt, tools and messages that
// produced the reply ("compaction" when the reply carries a compaction block,
// whose prefix starts at that block). When the history before the reply has
// changed since, its thinking blocks would fail the API's preserved-thinking
// check, so the request leaves them out.

const association_field = "_tool_call_ids";
const compaction_prefix = "compaction";

const Replay = struct {
    prefix: []const u8,
    content: []const std.json.Value,
    ids: []const std.json.Value,
};

/// Values borrow `arena`.
fn parseReplay(arena: Allocator, raw: []const u8) Error!Replay {
    if (raw.len > types.ProviderReplay.max_bytes) return error.ReplayTooLarge;
    if (json_depth.exceeds(raw)) return error.JsonTooDeep;
    const value = std.json.parseFromSliceLeaky(std.json.Value, arena, raw, .{ .duplicate_field_behavior = .@"error", .parse_numbers = false }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidProviderState,
    };
    if (value != .object) return error.InvalidProviderState;
    const fields = value.object;
    if (fields.count() != 3) return error.InvalidProviderState;
    const prefix = fields.get("prefix") orelse return error.InvalidProviderState;
    const content = fields.get("content") orelse return error.InvalidProviderState;
    const ids = fields.get(association_field) orelse return error.InvalidProviderState;
    if (prefix != .string or content != .array or ids != .array) return error.InvalidProviderState;
    for (content.array.items) |block| {
        if (block != .object) return error.InvalidProviderState;
        const kind = block.object.get("type") orelse return error.InvalidProviderState;
        if (kind != .string) return error.InvalidProviderState;
    }
    for (ids.array.items) |id| if (id != .string) return error.InvalidProviderState;
    return .{ .prefix = prefix.string, .content = content.array.items, .ids = ids.array.items };
}

fn blockType(block: std.json.Value) []const u8 {
    return block.object.get("type").?.string;
}

fn isThinking(block: std.json.Value) bool {
    const kind = blockType(block);
    return std.mem.eql(u8, kind, "thinking") or std.mem.eql(u8, kind, "redacted_thinking");
}

fn blockString(block: std.json.Value, field: []const u8) ?[]const u8 {
    const value = block.object.get(field) orelse return null;
    return if (value == .string) value.string else null;
}

fn writeReplay(writer: *std.Io.Writer, prefix: []const u8, content: []const std.json.Value, ids: []const []const u8) !void {
    try writer.writeAll("{\"prefix\":");
    try std.json.Stringify.value(prefix, .{}, writer);
    try writer.writeAll(",\"content\":[");
    for (content, 0..) |block, index| {
        if (index != 0) try writer.writeByte(',');
        try std.json.Stringify.value(block, .{}, writer);
    }
    try writer.writeAll("],\"" ++ association_field ++ "\":[");
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeByte(',');
        try std.json.Stringify.value(id, .{}, writer);
    }
    try writer.writeAll("]}");
}

/// Keeps the reply's thinking with the calls fx keeps. `text` false drops the
/// text blocks, `reasoning` false drops the whole replay. Returns `replay`
/// itself when nothing changes; a projected copy lives in `alloc`.
pub fn project_replay(alloc: Allocator, replay: ?types.ProviderReplay, calls: []const types.ToolCall, text: bool, reasoning: bool) Error!?types.ProviderReplay {
    const value = replay orelse return null;
    if (!reasoning) return null;
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const parsed = try parseReplay(arena, value.parts_json);
    var kept: std.ArrayList(std.json.Value) = .empty;
    var kept_ids: std.ArrayList([]const u8) = .empty;
    var changed = false;
    for (parsed.content) |block| {
        const kind = blockType(block);
        if (std.mem.eql(u8, kind, "text") and !text) {
            changed = true;
            continue;
        }
        if (std.mem.eql(u8, kind, "tool_use")) {
            const id = blockString(block, "id") orelse return error.InvalidProviderState;
            const keep = for (calls) |call| {
                if (std.mem.eql(u8, call.id, id)) break true;
            } else false;
            if (!keep) {
                changed = true;
                continue;
            }
            try kept_ids.append(arena, id);
        }
        try kept.append(arena, block);
    }
    if (kept_ids.items.len != calls.len) return null;
    if (!changed) return value;
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    writeReplay(&out.writer, parsed.prefix, kept.items, kept_ids.items) catch return error.OutOfMemory;
    return .{ .source = value.source, .parts_json = try out.toOwnedSlice() };
}

// ---------------------------------------------------------------------------
// Request serialization

/// Pure Messages serialization. Caller owns the returned bytes.
pub fn build_request(alloc: Allocator, input: stream_provider.RequestData, settings: Settings) Error![]u8 {
    const rendered = try render(alloc, input, settings);
    return rendered.body;
}

/// The digest a reply to `input` records as its prefix.
pub fn prefixDigest(alloc: Allocator, input: stream_provider.RequestData, settings: Settings) Error![32]u8 {
    const rendered = try render(alloc, input, settings);
    alloc.free(rendered.body);
    return rendered.digest;
}

const Rendered = struct { body: []u8, digest: [32]u8 };

fn render(alloc: Allocator, input: stream_provider.RequestData, settings: Settings) Error!Rendered {
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
    var renderer: Renderer = .{
        .arena = arena,
        .request = request,
        .settings = settings,
        .functions = functions.items,
        .projection = &projection,
        .digest = Sha256.init(.{}),
    };
    renderer.digest.update("fx-anthropic-prefix-v1");
    var out: std.Io.Writer.Allocating = .init(alloc);
    defer out.deinit();
    renderer.write(&out.writer) catch |err| switch (err) {
        error.WriteFailed, error.OutOfMemory => return error.OutOfMemory,
        error.InvalidProviderState, error.ReplayTooLarge, error.JsonTooDeep, error.ImageUnavailable, error.InvalidProviderPrompt, error.InvalidToolName => |failure| return failure,
        else => return error.InvalidToolSchema,
    };
    var digest: [32]u8 = undefined;
    renderer.digest.final(&digest);
    return .{ .body = try out.toOwnedSlice(), .digest = digest };
}

fn toolResultDenied(message: types.ChatMessage) bool {
    const failed = if (message.tool_result_status) |status| status == .failure else false;
    return failed and tool_result_errors.toolPermissionDenialReason(message.content orelse "") != null;
}

const Renderer = struct {
    arena: Allocator,
    request: stream_provider.RequestData,
    settings: Settings,
    functions: []const Function,
    projection: *const tool_call_ids.Projection,
    digest: Sha256,

    fn hashSegment(self: *Renderer, bytes: []const u8) void {
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, bytes.len, .little);
        self.digest.update(&len);
        self.digest.update(bytes);
    }

    fn digestNow(self: *const Renderer) [32]u8 {
        var copy = self.digest;
        var out: [32]u8 = undefined;
        copy.final(&out);
        return out;
    }

    fn write(self: *Renderer, w: *std.Io.Writer) !void {
        const request = self.request;
        const options = self.settings.options;
        const info = builtinModel(request.model);
        try w.writeAll("{\"model\":");
        try std.json.Stringify.value(request.model, .{}, w);
        const limit = request.max_output_tokens orelse if (info) |model| @min(model.max_output_tokens, default_max_tokens) else default_max_tokens;
        try w.print(",\"max_tokens\":{d},\"stream\":true", .{limit});
        if (info != null) {
            try w.writeAll(",\"thinking\":{\"type\":\"adaptive\"");
            try w.print(",\"display\":\"{s}\"", .{@tagName(options.thinking_display)});
            if (sendsBlockBinding(options, request.model)) {
                try w.print(",\"block_binding\":{{\"prefix_mismatch_behavior\":\"{s}\"}}", .{@tagName(options.prefix_mismatch_behavior)});
            }
            try w.writeByte('}');
        }
        if (request.provider_options.reasoning) |*effort| {
            const label = effort.label();
            if (containsName(&efforts, label)) try w.print(",\"output_config\":{{\"effort\":\"{s}\"}}", .{label});
        }
        if (options.prompt_caching) {
            try w.writeAll(",\"cache_control\":");
            try self.writeCacheControl(w);
        }
        try self.writeContextManagement(w);
        try self.writeSystem(w);
        try self.writeTools(w);
        try w.writeAll(",\"messages\":[");
        try self.writeMessages(w);
        try w.writeAll("]}");
    }

    fn writeCacheControl(self: *const Renderer, w: *std.Io.Writer) !void {
        try w.writeAll("{\"type\":\"ephemeral\"");
        if (self.settings.options.cache_ttl == .@"1h") try w.writeAll(",\"ttl\":\"1h\"");
        try w.writeByte('}');
    }

    fn writeContextManagement(self: *const Renderer, w: *std.Io.Writer) !void {
        const options = self.settings.options;
        if (options.compaction == null and options.context_editing == null) return;
        try w.writeAll(",\"context_management\":{\"edits\":[");
        var wrote = false;
        if (options.context_editing) |editing| {
            try w.writeAll("{\"type\":\"clear_tool_uses_20250919\"");
            if (editing.trigger_tokens) |tokens| try w.print(",\"trigger\":{{\"type\":\"input_tokens\",\"value\":{d}}}", .{tokens});
            if (editing.keep_tool_uses) |keep| try w.print(",\"keep\":{{\"type\":\"tool_uses\",\"value\":{d}}}", .{keep});
            if (editing.clear_at_least_tokens) |tokens| try w.print(",\"clear_at_least\":{{\"type\":\"input_tokens\",\"value\":{d}}}", .{tokens});
            if (editing.clear_tool_inputs) try w.writeAll(",\"clear_tool_inputs\":true");
            if (editing.exclude_tools.len != 0) {
                try w.writeAll(",\"exclude_tools\":[");
                for (editing.exclude_tools, 0..) |name, index| {
                    if (index != 0) try w.writeByte(',');
                    try writeWireName(w, name);
                }
                try w.writeByte(']');
            }
            try w.writeByte('}');
            wrote = true;
        }
        if (options.compaction) |compaction| {
            if (wrote) try w.writeByte(',');
            try w.writeAll("{\"type\":\"compact_20260112\"");
            if (compaction.trigger_tokens) |tokens| try w.print(",\"trigger\":{{\"type\":\"input_tokens\",\"value\":{d}}}", .{tokens});
            if (compaction.instructions) |text| {
                try w.writeAll(",\"instructions\":");
                try std.json.Stringify.value(text, .{}, w);
            }
            try w.writeByte('}');
        }
        try w.writeAll("]}");
    }

    /// System blocks hash without their cache markers: moving a marker never
    /// changes the prefix a thinking block is bound to.
    fn writeSystem(self: *Renderer, w: *std.Io.Writer) !void {
        var scratch: std.Io.Writer.Allocating = .init(self.arena);
        var count: usize = 0;
        var markers: usize = 0;
        var last_nonempty: ?usize = null;
        var flagged = false;
        for (self.request.instructions, 0..) |instruction, index| {
            if (instruction.content.?.len == 0) continue;
            last_nonempty = index;
            flagged = flagged or instruction.cache_breakpoint;
        }
        if (last_nonempty == null) {
            self.hashSegment("");
            return;
        }
        try w.writeAll(",\"system\":[");
        for (self.request.instructions, 0..) |instruction, index| {
            const text = instruction.content.?;
            if (text.len == 0) continue;
            if (count != 0) try w.writeByte(',');
            count += 1;
            const start = scratch.written().len;
            try scratch.writer.writeAll("{\"type\":\"text\",\"text\":");
            try std.json.Stringify.value(text, .{}, &scratch.writer);
            try w.writeAll(scratch.written()[start..]);
            try scratch.writer.writeByte('}');
            const marked = self.settings.options.prompt_caching and markers < 2 and
                (if (flagged) instruction.cache_breakpoint else index == last_nonempty.?);
            if (marked) {
                markers += 1;
                try w.writeAll(",\"cache_control\":");
                try self.writeCacheControl(w);
            }
            try w.writeByte('}');
        }
        try w.writeByte(']');
        self.hashSegment(scratch.written());
    }

    fn writeTools(self: *Renderer, w: *std.Io.Writer) !void {
        if (self.functions.len == 0) {
            self.hashSegment("");
            return;
        }
        var scratch: std.Io.Writer.Allocating = .init(self.arena);
        const s = &scratch.writer;
        try s.writeByte('[');
        for (self.functions, 0..) |function, index| {
            if (index != 0) try s.writeByte(',');
            try s.writeAll("{\"name\":");
            try writeWireName(s, function.name);
            try s.writeAll(",\"description\":");
            try model_tool_schema.writeCappedDescriptionJsonString(self.arena, s, function.description);
            try s.writeAll(",\"input_schema\":");
            // Strict tools need every object schema closed.
            const strict = containsName(self.settings.options.strict_tools, function.name);
            switch (function.schema) {
                .builtin => |schema| {
                    var sent = schema;
                    if (strict) sent.additional_properties = false;
                    try model_tool_schema.writeObjectSchema(self.arena, s, sent);
                },
                .dynamic => |schema| try std.json.Stringify.value(if (strict) try closeObjects(self.arena, schema) else schema, .{}, s),
            }
            if (strict) try s.writeAll(",\"strict\":true");
            try s.writeByte('}');
        }
        try s.writeByte(']');
        self.hashSegment(scratch.written());
        try w.writeAll(",\"tools\":");
        try w.writeAll(scratch.written());
        try w.writeAll(",\"tool_choice\":{\"type\":");
        const choice: []const u8 = switch (self.request.tool_choice) {
            .auto => "auto",
            .none => "none",
            .required => if (builtinModel(self.request.model)) |info| (if (info.rejects_forced_tool_choice) "auto" else "any") else "any",
        };
        try std.json.Stringify.value(choice, .{}, w);
        if (self.request.provider_options.parallel_tool_calls) |parallel| {
            if (!parallel and !std.mem.eql(u8, choice, "none")) try w.writeAll(",\"disable_parallel_tool_use\":true");
        }
        try w.writeByte('}');
    }

    const Side = enum { user, assistant };

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
            const parsed = parseReplay(self.arena, replay.parts_json) catch continue;
            if (!std.mem.eql(u8, parsed.prefix, compaction_prefix)) continue;
            for (parsed.content) |block| {
                if (!std.mem.eql(u8, blockType(block), "compaction")) continue;
                const summary = block.object.get("content") orelse continue;
                if (summary == .string) return index;
            }
        }
        return 0;
    }

    fn writeMessages(self: *Renderer, w: *std.Io.Writer) !void {
        const messages = self.request.messages;
        const start = try self.compactionStart();
        if (start != 0) debug_trace.logf("gateway", "anthropic_compaction_history_trimmed omitted_messages={d}", .{start});
        var current: ?Side = null;
        var body: std.Io.Writer.Allocating = .init(self.arena);
        var before: [32]u8 = self.digestNow();
        var wire_count: usize = 0;
        var index = start;
        while (index < messages.len) {
            const message = messages[index];
            const side: Side = if (message.role == .assistant) .assistant else .user;
            var blocks: std.Io.Writer.Allocating = .init(self.arena);
            // A reply's prefix is everything before its wire message, so the
            // open message of the other side counts as already sent.
            const next_before = if (current == side) before else if (current) |open| try self.digestAfter(&body, open) else self.digestNow();
            const first_wire = wire_count == 0 and (current == null or current == side);
            var consumed: usize = 1;
            switch (side) {
                .assistant => try self.writeAssistant(&blocks.writer, message, next_before, first_wire),
                .user => consumed = try self.writeUserRun(&blocks.writer, messages[index..]),
            }
            index += consumed;
            if (blocks.written().len == 0) continue;
            if (current != side) {
                if (current) |open| try self.flush(w, &body, open, &wire_count);
                current = side;
                before = next_before;
            } else try body.writer.writeByte(',');
            try body.writer.writeAll(blocks.written());
        }
        if (current) |open| try self.flush(w, &body, open, &wire_count);
    }

    fn wireMessage(self: *Renderer, body: *std.Io.Writer.Allocating, side: Side) ![]const u8 {
        var message: std.Io.Writer.Allocating = .init(self.arena);
        try message.writer.print("{{\"role\":\"{s}\",\"content\":[", .{@tagName(side)});
        try message.writer.writeAll(body.written());
        try message.writer.writeAll("]}");
        return message.written();
    }

    fn digestAfter(self: *Renderer, body: *std.Io.Writer.Allocating, side: Side) ![32]u8 {
        const saved = self.digest;
        defer self.digest = saved;
        self.hashSegment(try self.wireMessage(body, side));
        return self.digestNow();
    }

    fn flush(self: *Renderer, w: *std.Io.Writer, body: *std.Io.Writer.Allocating, side: Side, wire_count: *usize) !void {
        const message = try self.wireMessage(body, side);
        self.hashSegment(message);
        if (wire_count.* != 0) try w.writeByte(',');
        try w.writeAll(message);
        wire_count.* += 1;
        body.clearRetainingCapacity();
    }

    /// Writes a run of tool results and user messages as the blocks of one
    /// user message, tool results first as the API requires. Returns how
    /// many source messages it used.
    fn writeUserRun(self: *Renderer, w: *std.Io.Writer, messages: []const types.ChatMessage) !usize {
        var count: usize = 0;
        var wrote = false;
        for (messages) |message| {
            if (message.role == .assistant) break;
            count += 1;
            switch (message.role) {
                .tool => try self.writeToolResult(w, message, &wrote),
                else => try self.writeUserBlocks(w, message, &wrote),
            }
        }
        return count;
    }

    fn separator(w: *std.Io.Writer, wrote: *bool) !void {
        if (wrote.*) try w.writeByte(',');
        wrote.* = true;
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
            try w.writeAll("{\"type\":\"text\",\"text\":");
            try std.json.Stringify.value(text, .{}, w);
            try w.writeByte('}');
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
                try w.writeAll("{\"type\":\"text\",\"text\":");
                try std.json.Stringify.value(text, .{}, w);
                try w.writeByte('}');
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

    fn writeAssistant(self: *Renderer, w: *std.Io.Writer, message: types.ChatMessage, before: [32]u8, first_wire: bool) !void {
        if (message.provider_replay) |replay| {
            if (try self.writeReplayed(w, message, replay, before, first_wire)) return;
        }
        var wrote = false;
        if (message.content) |text| if (text.len != 0) {
            try separator(w, &wrote);
            try w.writeAll("{\"type\":\"text\",\"text\":");
            try std.json.Stringify.value(text, .{}, w);
            try w.writeByte('}');
        };
        for (message.tool_calls) |call| {
            try separator(w, &wrote);
            try w.writeAll("{\"type\":\"tool_use\",\"id\":");
            try std.json.Stringify.value(self.projection.resolve(call.id), .{}, w);
            try w.writeAll(",\"name\":");
            try writeWireName(w, call.name);
            try w.writeAll(",\"input\":");
            try writeToolInput(self.arena, w, call.arguments_json);
            try w.writeByte('}');
        }
    }

    /// Writes the reply's own blocks when they still describe this message.
    /// Returns false when the canonical form must be written instead.
    fn writeReplayed(self: *Renderer, w: *std.Io.Writer, message: types.ChatMessage, replay: types.ProviderReplay, before: [32]u8, first_wire: bool) !bool {
        const parsed = parseReplay(self.arena, replay.parts_json) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidProviderState,
        };
        if (parsed.ids.len != message.tool_calls.len) return error.InvalidProviderState;
        for (parsed.ids, message.tool_calls) |id, call| if (!std.mem.eql(u8, id.string, call.id)) return error.InvalidProviderState;
        var text: std.ArrayList(u8) = .empty;
        for (parsed.content) |block| {
            if (!std.mem.eql(u8, blockType(block), "text")) continue;
            const part = blockString(block, "text") orelse return error.InvalidProviderState;
            if (part.len == 0) continue;
            if (text.items.len != 0) try text.appendSlice(self.arena, text_block_separator);
            try text.appendSlice(self.arena, part);
        }
        if (!std.mem.eql(u8, text.items, message.content orelse "")) {
            debug_trace.logf("gateway", "anthropic_replay_text_changed canonical=true", .{});
            return false;
        }
        var hex = std.fmt.bytesToHex(before, .lower);
        const bound = if (std.mem.eql(u8, parsed.prefix, compaction_prefix)) first_wire else std.mem.eql(u8, parsed.prefix, &hex);
        if (!bound) debug_trace.logf("gateway", "anthropic_replay_prefix_changed thinking_omitted=true", .{});
        var wrote = false;
        for (parsed.content) |block| {
            if (!bound and isThinking(block)) continue;
            if (std.mem.eql(u8, blockType(block), "text") and (blockString(block, "text") orelse @as([]const u8, "")).len == 0) continue;
            try separator(w, &wrote);
            try std.json.Stringify.value(block, .{}, w);
        }
        return true;
    }
};

/// Text blocks of one reply join with this in fx's assistant content.
pub const text_block_separator = "\n\n";

fn writeToolInput(arena: Allocator, w: *std.Io.Writer, arguments: []const u8) !void {
    const trimmed = std.mem.trim(u8, arguments, " \t\r\n");
    if (trimmed.len == 0) return w.writeAll("{}");
    const value = std.json.parseFromSliceLeaky(std.json.Value, arena, trimmed, .{ .parse_numbers = false }) catch return w.writeAll("{}");
    if (value != .object) return w.writeAll("{}");
    try std.json.Stringify.value(value, .{}, w);
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
    prefix: [64]u8,
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

    /// `prefix` is the digest of the request this reply answers.
    pub fn init(alloc: Allocator, request: stream_provider.RequestData, prefix: [32]u8, limits: Limits) Error!Reducer {
        var self: Reducer = .{
            .alloc = alloc,
            .arena_state = std.heap.ArenaAllocator.init(alloc),
            .limits = limits,
            .choice = request.tool_choice,
            .prefix = std.fmt.bytesToHex(prefix, .lower),
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
            try self.acceptError(root);
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
            if (nonNull(root, "context_management")) |management| {
                const encoded = try std.json.Stringify.valueAlloc(self.arena(), management, .{});
                debug_trace.logf("gateway", "anthropic_context_management {s}", .{encoded[0..@min(encoded.len, 512)]});
            }
            return .{};
        }
        if (std.mem.eql(u8, kind, "message_stop")) {
            self.phase = .stopped;
            return .{};
        }
        // The stream may grow new event types; ignore them as the API asks.
        return .{};
    }

    fn acceptError(self: *Reducer, root: std.json.ObjectMap) Error!void {
        const detail = try std.json.Stringify.valueAlloc(self.arena(), std.json.Value{ .object = root }, .{});
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
        if (nonNull(fields, "iterations")) |iterations| {
            const encoded = try std.json.Stringify.valueAlloc(self.arena(), iterations, .{});
            debug_trace.logf("gateway", "anthropic_usage_iterations {s}", .{encoded[0..@min(encoded.len, 512)]});
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
        var content: std.ArrayList(std.json.Value) = .empty;
        var ids: std.ArrayList([]const u8) = .empty;
        var compaction = false;
        for (self.blocks.items) |*block| {
            var fields: std.json.ObjectMap = .empty;
            try fields.put(arena_alloc, "type", .{ .string = @tagName(block.kind) });
            switch (block.kind) {
                .text => {
                    if (block.text.items.len != 0) {
                        if (text.items.len != 0) try text.appendSlice(arena_alloc, text_block_separator);
                        try text.appendSlice(arena_alloc, block.text.items);
                    }
                    try fields.put(arena_alloc, "text", .{ .string = block.text.items });
                },
                .thinking => {
                    try fields.put(arena_alloc, "thinking", .{ .string = block.text.items });
                    try fields.put(arena_alloc, "signature", .{ .string = block.signature.items });
                },
                .redacted_thinking => try fields.put(arena_alloc, "data", .{ .string = block.data }),
                .tool_use => {
                    try fields.put(arena_alloc, "id", .{ .string = block.id });
                    try fields.put(arena_alloc, "name", .{ .string = block.name });
                    const input_text = if (block.text.items.len == 0) "{}" else block.text.items;
                    const input = std.json.parseFromSliceLeaky(std.json.Value, arena_alloc, input_text, .{ .parse_numbers = false }) catch null;
                    try fields.put(arena_alloc, "input", if (input != null and input.? == .object) input.? else .{ .object = .empty });
                    try ids.append(arena_alloc, block.id);
                },
                .compaction => {
                    compaction = compaction or block.summary_present;
                    try fields.put(arena_alloc, "content", if (block.summary_present) .{ .string = block.text.items } else .null);
                },
            }
            try content.append(arena_alloc, .{ .object = fields });
        }
        if (compaction) debug_trace.logf("gateway", "anthropic_server_compaction summary_bytes={d}", .{self.reasoning_bytes});

        var state: std.Io.Writer.Allocating = .init(self.alloc);
        defer state.deinit();
        writeReplay(&state.writer, if (compaction) compaction_prefix else &self.prefix, content.items, ids.items) catch return error.OutOfMemory;
        if (state.written().len > types.ProviderReplay.max_bytes) return error.ReplayTooLarge;

        var calls: std.ArrayList(types.ToolCall) = .empty;
        errdefer {
            for (calls.items) |call| {
                self.alloc.free(call.id);
                self.alloc.free(call.name);
                self.alloc.free(call.arguments_json);
            }
            calls.deinit(self.alloc);
        }
        for (self.blocks.items) |block| {
            if (block.kind != .tool_use) continue;
            const id = try self.alloc.dupe(u8, block.id);
            errdefer self.alloc.free(id);
            const name = try self.alloc.dupe(u8, originalName(self.functions, block.name));
            errdefer self.alloc.free(name);
            const arguments = try self.alloc.dupe(u8, if (block.text.items.len == 0) "{}" else block.text.items);
            errdefer self.alloc.free(arguments);
            try calls.append(self.alloc, .{ .id = id, .name = name, .arguments_json = arguments });
        }
        const owned_text = if (text.items.len != 0) try self.alloc.dupe(u8, text.items) else null;
        errdefer if (owned_text) |value| self.alloc.free(value);
        const generation_id = if (self.message_id) |id| try self.alloc.dupe(u8, id) else null;
        errdefer if (generation_id) |id| self.alloc.free(id);
        const failure_detail = if (self.stop_details) |details| try self.alloc.dupe(u8, details) else null;
        errdefer if (failure_detail) |detail| self.alloc.free(detail);
        const owned_calls = try calls.toOwnedSlice(self.alloc);
        errdefer types.freeToolCallSlice(self.alloc, owned_calls);
        const provider_state = try state.toOwnedSlice();
        self.phase = .closed;
        return .{ .completed = .{
            .completion = .{
                .content = owned_text,
                .tool_calls = owned_calls,
                .generation_id = generation_id,
                .provider_state_json = provider_state,
                .provider_failure_detail = failure_detail,
                .finish_reason = reason,
                .finish_reason_raw = .init(raw),
                .usage = self.finalUsage(),
            },
            .ownership = .owned,
        } };
    }
};

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
    settings: Settings,
    limits: Limits,
    events: ?stream_provider.EventSink,
    cancel_flag: *const std.atomic.Value(bool),
    progress: ?sse.Progress,
) Error!stream_provider.Result {
    if (cancel_flag.load(.seq_cst)) return error.Cancelled;
    const prefix = try prefixDigest(alloc, request, settings);
    var reducer = try Reducer.init(alloc, request, prefix, limits);
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
    var options: Options = .{ .strict_tools = &.{"read_file"} };
    const body = try build_request(alloc, request, testSettings(&options));
    defer alloc.free(body);
    var parsed = try parseBody(alloc, body);
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("claude-haiku-5-5", root.get("model").?.string);
    try std.testing.expectEqual(@as(i64, 1024), root.get("max_tokens").?.integer);
    try std.testing.expect(root.get("stream").?.bool);
    try std.testing.expect(root.get("temperature") == null and root.get("top_p") == null and root.get("top_k") == null);
    try expectContains(body, "\"thinking\":{\"type\":\"adaptive\",\"display\":\"summarized\",\"block_binding\":{\"prefix_mismatch_behavior\":\"drop_block\"}}");
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
    Sha256.hash(png, &digest, .{});
    var hex = std.fmt.bytesToHex(digest, .lower);
    const image: types.ImageAttachment = .{ .path = @constCast("pixel.png"), .media_type = @constCast("image/png"), .inline_data = @constCast(png), .snapshot_sha256 = &hex };
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "what color?", .images = &.{image} }};
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &messages), testSettings(&test_options));
    defer alloc.free(body);
    try expectContains(body, "{\"type\":\"image\",\"source\":{\"type\":\"base64\",\"media_type\":\"image/png\",\"data\":\"iVBORw0K");
    try expectContains(body, "\"}},{\"type\":\"text\",\"text\":\"what color?\"}");
}

test "anthropic context management and compaction request shape" {
    const alloc = std.testing.allocator;
    var options: Options = .{
        .compaction = .{ .trigger_tokens = 50_000, .instructions = "Keep open tasks." },
        .context_editing = .{ .trigger_tokens = 3000, .keep_tool_uses = 1, .clear_at_least_tokens = 100, .exclude_tools = &.{"skill.run"} },
    };
    const body = try build_request(alloc, testRequest("claude-haiku-5-5", &.{.{ .role = .user, .content = "x" }}), testSettings(&options));
    defer alloc.free(body);
    try expectContains(body, "\"context_management\":{\"edits\":[{\"type\":\"clear_tool_uses_20250919\",\"trigger\":{\"type\":\"input_tokens\",\"value\":3000},\"keep\":{\"type\":\"tool_uses\",\"value\":1},\"clear_at_least\":{\"type\":\"input_tokens\",\"value\":100},\"exclude_tools\":[\"skill__run\"]},{\"type\":\"compact_20260112\",\"trigger\":{\"type\":\"input_tokens\",\"value\":50000},\"instructions\":\"Keep open tasks.\"}]}");
}

fn testStream(alloc: Allocator, request: stream_provider.RequestData, wire: []const u8, sink: ?stream_provider.EventSink) !stream_provider.Result {
    var source = std.Io.Reader.fixed(wire);
    var cancel = std.atomic.Value(bool).init(false);
    return consume_stream(alloc, &source, request, testSettings(&test_options), .{}, sink, &cancel, null);
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
    try expectContains(state, "{\"type\":\"tool_use\",\"id\":\"toolu_A\",\"name\":\"read_file\",\"input\":{\"path\":\"a.txt\"}}");
    try expectContains(state, "\"_tool_call_ids\":[\"toolu_A\",\"toolu_B\"]");
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
    try expectContains(second, "{\"role\":\"assistant\",\"content\":[{\"type\":\"thinking\",\"thinking\":\"Need the file.\",\"signature\":\"c2lnbmVk\"},{\"type\":\"redacted_thinking\",\"data\":\"b3BhcXVl\"},{\"type\":\"text\",\"text\":\"Let me look.\"},{\"type\":\"tool_use\",\"id\":\"toolu_A\",\"name\":\"read_file\",\"input\":{\"path\":\"a.txt\"}},{\"type\":\"tool_use\",\"id\":\"toolu_B\",\"name\":\"skill__run\",\"input\":{}}]}");
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
    try std.testing.expect(std.mem.find(u8, one.parts_json, "\"type\":\"text\"") == null);
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
    try expectContains(completion.provider_state_json.?, "\"prefix\":\"compaction\"");
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
