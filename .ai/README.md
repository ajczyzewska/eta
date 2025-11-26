# AI Context Documentation

This directory contains context documentation optimized for AI agents.

## File: CONTEXT.md (5KB, ~162 lines)

**Single-file** comprehensive reference for AI agents to understand this project without consuming excessive context.

### What's Inside

- **Quick Start** - What the project does in one line
- **Core Info** - Tech stack, CLI usage, key specs
- **Architecture** - Key functions with line numbers
- **Processing Flow** - Visual pipeline diagram
- **Content Filtering** - Skip rules and chapter detection
- **Key Constants** - Critical config values and locations
- **Optimization Profiles** - Performance settings
- **Common Modifications** - Code examples for frequent changes
- **Performance Metrics** - CPU vs GPU benchmarks
- **Common Issues** - Quick troubleshooting
- **Testing Checklist** - Validation steps
- **Quick Commands** - Essential bash commands

### Usage

```bash
# AI agents: Read this first
cat .ai/CONTEXT.md

# Get specific info quickly (search for keywords)
grep -A 5 "Architecture" .ai/CONTEXT.md
```

### Design Philosophy

**Optimized for:**
- ✅ Low token consumption (~2K tokens vs 20K+ for verbose docs)
- ✅ Quick scanning (tables, bullet points)
- ✅ Direct answers (no verbose explanations)
- ✅ Line numbers for fast code navigation
- ✅ Copy-paste code examples

**Trade-offs:**
- Minimal prose - assumes technical audience
- Tables over paragraphs - easier to scan
- Abbreviated where clear - saves tokens
- No redundancy - each fact stated once

### When to Use

- First-time project understanding (3 min read)
- Quick reference during coding
- Finding line numbers for functions
- Common modification patterns
- Troubleshooting issues

### Maintenance

Update CONTEXT.md when:
- Adding/removing major functions
- Changing CLI interface
- Modifying key constants
- Adding common issues/solutions
- Changing architecture significantly

Keep it under 200 lines and 6KB for optimal AI consumption.

---

**Previous Approach:** 5 files, 59KB total, ~25 min read time
**New Approach:** 1 file, 5KB, ~3 min read time
**Token Savings:** ~90% reduction
