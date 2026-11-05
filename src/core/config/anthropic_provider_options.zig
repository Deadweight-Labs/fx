//! Settings that only a configured provider with the `anthropic-messages`
//! protocol reads, from the `anthropic` object of its definition. Parsed and
//! owned by `configured_provider.Registry`; the transport borrows them.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const default_version = "2023-06-01";
pub const default_base_url = "https://api.anthropic.com";
pub const default_api_key_env = "ANTHROPIC_API_KEY";

const max_betas = 16;
const max_beta_bytes = 64;
const max_instructions_bytes = 16 * 1024;
const max_excluded_tools = 64;
const max_tool_name_bytes = 128;
/// The API refuses a lower compaction trigger.
pub const min_compaction_trigger_tokens: u32 = 50_000;

pub const ParseError = Allocator.Error || error{InvalidAnthropicOptions};

pub const ThinkingDisplay = enum { summarized, omitted };
pub const PrefixMismatchBehavior = enum { drop_block, @"error", unset };
pub const CacheTtl = enum { @"5m", @"1h" };

pub const Compaction = struct {
    trigger_tokens: ?u32 = null,
    instructions: ?[]const u8 = null,
};

pub const ContextEditing = struct {
    trigger_tokens: ?u32 = null,
    keep_tool_uses: ?u32 = null,
    clear_at_least_tokens: ?u32 = null,
    clear_tool_inputs: bool = false,
    exclude_tools: []const []const u8 = &.{},
};

pub const Options = struct {
    version: []const u8 = default_version,
    betas: []const []const u8 = &.{},
    prompt_caching: bool = true,
    cache_ttl: CacheTtl = .@"5m",
    thinking_display: ThinkingDisplay = .summarized,
    prefix_mismatch_behavior: PrefixMismatchBehavior = .drop_block,
    compaction: ?Compaction = null,
    context_editing: ?ContextEditing = null,
    /// Tool names whose definitions are sent with `strict: true`.
    strict_tools: []const []const u8 = &.{},
    owned: bool = false,

    pub fn deinit(self: Options, alloc: Allocator) void {
        if (!self.owned) return;
        alloc.free(self.version);
        for (self.betas) |beta| alloc.free(beta);
        alloc.free(self.betas);
        for (self.strict_tools) |name| alloc.free(name);
        alloc.free(self.strict_tools);
        if (self.compaction) |compaction| if (compaction.instructions) |text| alloc.free(text);
        if (self.context_editing) |editing| {
            for (editing.exclude_tools) |name| alloc.free(name);
            alloc.free(editing.exclude_tools);
        }
    }
};

/// Returns owned options; free with `Options.deinit`.
pub fn parse(alloc: Allocator, value: std.json.Value) ParseError!Options {
    try checkFields(value, &.{ "version", "betas", "prompt_caching", "cache_ttl", "thinking_display", "prefix_mismatch_behavior", "compaction", "context_editing", "strict_tools" });
    const fields = value.object;
    var result: Options = .{ .owned = true, .version = undefined };
    result.version = try alloc.dupe(u8, if (fields.get("version")) |version| try versionString(version) else default_version);
    errdefer alloc.free(result.version);
    result.betas = if (fields.get("betas")) |betas| try parseBetas(alloc, betas) else try alloc.alloc([]const u8, 0);
    errdefer {
        for (result.betas) |beta| alloc.free(beta);
        alloc.free(result.betas);
    }
    result.strict_tools = if (fields.get("strict_tools")) |names| try parseNames(alloc, names) else try alloc.alloc([]const u8, 0);
    errdefer {
        for (result.strict_tools) |name| alloc.free(name);
        alloc.free(result.strict_tools);
    }
    if (fields.get("prompt_caching")) |caching| result.prompt_caching = try boolean(caching);
    if (fields.get("cache_ttl")) |ttl| result.cache_ttl = try enumValue(CacheTtl, ttl);
    if (fields.get("thinking_display")) |display| result.thinking_display = try enumValue(ThinkingDisplay, display);
    if (fields.get("prefix_mismatch_behavior")) |behavior| result.prefix_mismatch_behavior = try enumValue(PrefixMismatchBehavior, behavior);
    if (fields.get("compaction")) |compaction| {
        try checkFields(compaction, &.{ "trigger_tokens", "instructions" });
        var parsed: Compaction = .{};
        if (compaction.object.get("trigger_tokens")) |trigger| {
            const tokens = try positive(trigger);
            if (tokens < min_compaction_trigger_tokens) return error.InvalidAnthropicOptions;
            parsed.trigger_tokens = tokens;
        }
        if (compaction.object.get("instructions")) |instructions| {
            if (instructions != .string or instructions.string.len == 0 or instructions.string.len > max_instructions_bytes) return error.InvalidAnthropicOptions;
            if (!std.unicode.utf8ValidateSlice(instructions.string)) return error.InvalidAnthropicOptions;
            parsed.instructions = try alloc.dupe(u8, instructions.string);
        }
        result.compaction = parsed;
    }
    errdefer if (result.compaction) |compaction| if (compaction.instructions) |text| alloc.free(text);
    if (fields.get("context_editing")) |editing| {
        try checkFields(editing, &.{ "trigger_tokens", "keep_tool_uses", "clear_at_least_tokens", "clear_tool_inputs", "exclude_tools" });
        var parsed: ContextEditing = .{};
        if (editing.object.get("trigger_tokens")) |trigger| parsed.trigger_tokens = try positive(trigger);
        if (editing.object.get("keep_tool_uses")) |keep| parsed.keep_tool_uses = try nonNegative(keep);
        if (editing.object.get("clear_at_least_tokens")) |minimum| parsed.clear_at_least_tokens = try positive(minimum);
        if (editing.object.get("clear_tool_inputs")) |inputs| parsed.clear_tool_inputs = try boolean(inputs);
        parsed.exclude_tools = if (editing.object.get("exclude_tools")) |names| try parseNames(alloc, names) else try alloc.alloc([]const u8, 0);
        result.context_editing = parsed;
    }
    return result;
}

fn checkFields(value: std.json.Value, allowed: []const []const u8) ParseError!void {
    if (value != .object) return error.InvalidAnthropicOptions;
    for (value.object.keys()) |key| {
        for (allowed) |name| {
            if (std.mem.eql(u8, key, name)) break;
        } else return error.InvalidAnthropicOptions;
    }
}

fn versionString(value: std.json.Value) ParseError![]const u8 {
    if (value != .string or value.string.len != 10) return error.InvalidAnthropicOptions;
    for (value.string, 0..) |byte, index| {
        const ok = if (index == 4 or index == 7) byte == '-' else std.ascii.isDigit(byte);
        if (!ok) return error.InvalidAnthropicOptions;
    }
    return value.string;
}

fn headerToken(text: []const u8, max_bytes: usize) bool {
    if (text.len == 0 or text.len > max_bytes) return false;
    for (text) |byte| if (!std.ascii.isAlphanumeric(byte) and byte != '-' and byte != '_' and byte != '.') return false;
    return true;
}

fn parseBetas(alloc: Allocator, value: std.json.Value) ParseError![]const []const u8 {
    if (value != .array or value.array.items.len > max_betas) return error.InvalidAnthropicOptions;
    for (value.array.items) |item| if (item != .string or !headerToken(item.string, max_beta_bytes)) return error.InvalidAnthropicOptions;
    return dupeStrings(alloc, value.array.items);
}

fn parseNames(alloc: Allocator, value: std.json.Value) ParseError![]const []const u8 {
    if (value != .array or value.array.items.len > max_excluded_tools) return error.InvalidAnthropicOptions;
    for (value.array.items) |item| if (item != .string or !headerToken(item.string, max_tool_name_bytes)) return error.InvalidAnthropicOptions;
    return dupeStrings(alloc, value.array.items);
}

fn dupeStrings(alloc: Allocator, items: []const std.json.Value) Allocator.Error![]const []const u8 {
    const out = try alloc.alloc([]const u8, items.len);
    var done: usize = 0;
    errdefer {
        for (out[0..done]) |text| alloc.free(text);
        alloc.free(out);
    }
    for (items) |item| {
        out[done] = try alloc.dupe(u8, item.string);
        done += 1;
    }
    return out;
}

fn boolean(value: std.json.Value) ParseError!bool {
    return if (value == .bool) value.bool else error.InvalidAnthropicOptions;
}

fn positive(value: std.json.Value) ParseError!u32 {
    const number = try nonNegative(value);
    return if (number == 0) error.InvalidAnthropicOptions else number;
}

fn nonNegative(value: std.json.Value) ParseError!u32 {
    if (value != .integer or value.integer < 0 or value.integer > std.math.maxInt(u32)) return error.InvalidAnthropicOptions;
    return @intCast(value.integer);
}

fn enumValue(comptime T: type, value: std.json.Value) ParseError!T {
    if (value != .string) return error.InvalidAnthropicOptions;
    return std.meta.stringToEnum(T, value.string) orelse error.InvalidAnthropicOptions;
}

fn parseForTest(alloc: Allocator, json: []const u8) !Options {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, json, .{});
    defer parsed.deinit();
    return parse(alloc, parsed.value);
}

test "anthropic options parse defaults, every field and reject bad values" {
    const alloc = std.testing.allocator;
    const defaults = try parseForTest(alloc, "{}");
    defer defaults.deinit(alloc);
    try std.testing.expectEqualStrings(default_version, defaults.version);
    try std.testing.expect(defaults.prompt_caching);
    try std.testing.expect(defaults.compaction == null and defaults.context_editing == null);
    try std.testing.expectEqual(PrefixMismatchBehavior.drop_block, defaults.prefix_mismatch_behavior);

    const full = try parseForTest(alloc,
        \\{"version":"2026-10-21","betas":["a-2026-01-01"],"prompt_caching":false,"cache_ttl":"1h","thinking_display":"omitted",
        \\"prefix_mismatch_behavior":"error","compaction":{"trigger_tokens":50000,"instructions":"keep tasks"},
        \\"context_editing":{"trigger_tokens":3000,"keep_tool_uses":1,"clear_at_least_tokens":500,"clear_tool_inputs":true,"exclude_tools":["read_file"]},"strict_tools":["write_file"]}
    );
    defer full.deinit(alloc);
    try std.testing.expectEqualStrings("2026-10-21", full.version);
    try std.testing.expectEqualStrings("a-2026-01-01", full.betas[0]);
    try std.testing.expect(!full.prompt_caching);
    try std.testing.expectEqual(CacheTtl.@"1h", full.cache_ttl);
    try std.testing.expectEqual(@as(?u32, 50000), full.compaction.?.trigger_tokens);
    try std.testing.expectEqualStrings("keep tasks", full.compaction.?.instructions.?);
    try std.testing.expectEqual(@as(?u32, 1), full.context_editing.?.keep_tool_uses);
    try std.testing.expectEqualStrings("read_file", full.context_editing.?.exclude_tools[0]);
    try std.testing.expectEqualStrings("write_file", full.strict_tools[0]);

    for ([_][]const u8{
        "{\"unknown\":1}",
        "{\"version\":\"2023-6-01\"}",
        "{\"betas\":[\"bad header\"]}",
        "{\"compaction\":{\"trigger_tokens\":1000}}",
        "{\"compaction\":{\"instructions\":\"\"}}",
        "{\"context_editing\":{\"keep_tool_uses\":-1}}",
        "{\"thinking_display\":\"loud\"}",
        "[]",
    }) |json| try std.testing.expectError(error.InvalidAnthropicOptions, parseForTest(alloc, json));
}

test "anthropic options release partial allocations" {
    try std.testing.checkAllAllocationFailures(@import("../shared/testing_allocator.zig").no_resize, struct {
        fn run(alloc: Allocator) !void {
            const options = try parseForTest(alloc,
                \\{"betas":["a","b"],"strict_tools":["s"],"compaction":{"instructions":"x"},"context_editing":{"exclude_tools":["t1","t2"]}}
            );
            options.deinit(alloc);
        }
    }.run, .{});
}
