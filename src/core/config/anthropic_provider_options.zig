//! Settings that only a configured provider with the `anthropic-messages`
//! protocol reads, from the `anthropic` object of its definition. Parsed and
//! owned by `configured_provider.Registry`; the transport borrows them.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const default_version = "2023-06-01";
pub const default_base_url = "https://api.anthropic.com";
pub const default_api_key_env = "ANTHROPIC_API_KEY";

const max_list_bytes = 4096;

pub const ParseError = Allocator.Error || error{InvalidAnthropicOptions};

pub const ThinkingDisplay = enum { summarized, omitted };
pub const PrefixMismatchBehavior = enum { drop_block, @"error", unset };
pub const CacheTtl = enum { @"5m", @"1h" };

/// The settings in the order `parse` reads them; the first four are text.
const Field = enum { version, betas, strict_tools, context_management, prompt_caching, cache_ttl, thinking_display, prefix_mismatch_behavior };

pub const Options = struct {
    version: []const u8 = default_version,
    /// `anthropic-beta` values to add, comma-separated.
    betas: []const u8 = "",
    /// Names of the tools sent with `strict: true`, comma-separated.
    strict_tools: []const u8 = "",
    /// The request's `context_management` object, as JSON, sent as it is:
    /// server-side compaction and context editing in the API's own terms.
    context_management: ?[]const u8 = null,
    prompt_caching: bool = true,
    cache_ttl: CacheTtl = .@"5m",
    /// `omitted` returns thinking blocks with only their signature, which
    /// keeps every request that replays them small.
    thinking_display: ThinkingDisplay = .omitted,
    prefix_mismatch_behavior: PrefixMismatchBehavior = .drop_block,
    /// Holds every string above that is not a default.
    storage: ?[]u8 = null,

    pub fn deinit(self: Options, alloc: Allocator) void {
        if (self.storage) |bytes| alloc.free(bytes);
    }

    /// Whether `name` is in a comma-separated `list`.
    pub fn listed(list: []const u8, name: []const u8) bool {
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |entry| if (entry.len != 0 and std.mem.eql(u8, entry, name)) return true;
        return false;
    }
};

/// Returns owned options; free with `Options.deinit`.
pub fn parse(alloc: Allocator, value: std.json.Value) ParseError!Options {
    return parseInto(alloc, value) catch |err| switch (err) {
        error.WriteFailed => error.OutOfMemory,
        else => |failure| failure,
    };
}

fn parseInto(alloc: Allocator, value: std.json.Value) (ParseError || std.Io.Writer.Error)!Options {
    if (value != .object) return error.InvalidAnthropicOptions;
    var result: Options = .{};
    var text: std.Io.Writer.Allocating = .init(alloc);
    defer text.deinit();
    var spans: [4]?[2]usize = @splat(null);
    var it = value.object.iterator();
    while (it.next()) |entry| {
        const field = std.meta.stringToEnum(Field, entry.key_ptr.*) orelse return error.InvalidAnthropicOptions;
        const item = entry.value_ptr.*;
        const start = text.written().len;
        switch (field) {
            .version => try writeToken(&text.writer, item),
            .betas, .strict_tools => {
                if (item != .array) return error.InvalidAnthropicOptions;
                for (item.array.items, 0..) |name, index| {
                    if (index != 0) try text.writer.writeByte(',');
                    try writeToken(&text.writer, name);
                }
                if (text.written().len - start > max_list_bytes) return error.InvalidAnthropicOptions;
            },
            .context_management => {
                if (item != .object) return error.InvalidAnthropicOptions;
                try std.json.Stringify.value(item, .{}, &text.writer);
            },
            .prompt_caching => {
                if (item != .bool) return error.InvalidAnthropicOptions;
                result.prompt_caching = item.bool;
            },
            .cache_ttl => result.cache_ttl = try enumValue(CacheTtl, item),
            .thinking_display => result.thinking_display = try enumValue(ThinkingDisplay, item),
            .prefix_mismatch_behavior => result.prefix_mismatch_behavior = try enumValue(PrefixMismatchBehavior, item),
        }
        if (@backingInt(field) < spans.len) spans[@backingInt(field)] = .{ start, text.written().len };
    }
    if (text.written().len == 0 and spans[@backingInt(Field.context_management)] == null) return result;
    const storage = try text.toOwnedSlice();
    result.storage = storage;
    if (spans[0]) |span| result.version = storage[span[0]..span[1]];
    if (spans[1]) |span| result.betas = storage[span[0]..span[1]];
    if (spans[2]) |span| result.strict_tools = storage[span[0]..span[1]];
    if (spans[3]) |span| result.context_management = storage[span[0]..span[1]];
    return result;
}

/// A header value or tool name: letters, digits, `-`, `_` and `.`.
fn writeToken(writer: *std.Io.Writer, value: std.json.Value) (ParseError || std.Io.Writer.Error)!void {
    if (value != .string or value.string.len == 0 or value.string.len > 128) return error.InvalidAnthropicOptions;
    for (value.string) |byte| if (!std.ascii.isAlphanumeric(byte) and byte != '-' and byte != '_' and byte != '.') return error.InvalidAnthropicOptions;
    try writer.writeAll(value.string);
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
    const defaults = try parseForTest(alloc, "{\"prompt_caching\":true}");
    defer defaults.deinit(alloc);
    try std.testing.expectEqualStrings(default_version, defaults.version);
    try std.testing.expect(defaults.storage == null and defaults.context_management == null);
    try std.testing.expectEqual(PrefixMismatchBehavior.drop_block, defaults.prefix_mismatch_behavior);

    const full = try parseForTest(alloc,
        \\{"version":"2026-10-21","betas":["a-2026-01-01","b"],"prompt_caching":false,"cache_ttl":"1h","thinking_display":"summarized",
        \\"prefix_mismatch_behavior":"error","context_management":{"edits":[{"type":"compact_20260112"}]},"strict_tools":["write_file","skill.run"]}
    );
    defer full.deinit(alloc);
    try std.testing.expectEqualStrings("2026-10-21", full.version);
    try std.testing.expectEqualStrings("a-2026-01-01,b", full.betas);
    try std.testing.expect(!full.prompt_caching);
    try std.testing.expectEqual(CacheTtl.@"1h", full.cache_ttl);
    try std.testing.expectEqual(ThinkingDisplay.summarized, full.thinking_display);
    try std.testing.expectEqualStrings("{\"edits\":[{\"type\":\"compact_20260112\"}]}", full.context_management.?);
    try std.testing.expect(Options.listed(full.strict_tools, "skill.run") and !Options.listed(full.strict_tools, "write"));

    for ([_][]const u8{
        "{\"unknown\":1}",
        "{\"version\":\"2023 06 01\"}",
        "{\"betas\":[\"bad header\"]}",
        "{\"betas\":\"one\"}",
        "{\"context_management\":[]}",
        "{\"prompt_caching\":1}",
        "{\"thinking_display\":\"loud\"}",
        "[]",
    }) |json| try std.testing.expectError(error.InvalidAnthropicOptions, parseForTest(alloc, json));
}

test "anthropic options release partial allocations" {
    try std.testing.checkAllAllocationFailures(@import("../shared/testing_allocator.zig").no_resize, struct {
        fn run(alloc: Allocator) !void {
            const options = try parseForTest(alloc,
                \\{"betas":["a","b"],"strict_tools":["s"],"context_management":{"edits":[]}}
            );
            options.deinit(alloc);
        }
    }.run, .{});
}
