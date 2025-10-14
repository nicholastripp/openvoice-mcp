# Raspberry Pi Unicode Testing Guide

**Purpose**: Test Picovoice error message rendering on Raspberry Pi following PR #1 merge

**Context**: PR #1 added Unicode box-drawing characters (━) and emoji (⚠️) to Picovoice error messages. Raspberry Pi terminals may not support UTF-8 encoding by default.

## Quick Test (5 minutes)

### Step 1: Check Encoding Support

SSH into Raspberry Pi and run:

```bash
cd ~/openvoice-mcp  # or wherever installed
python3 tests/test_pi_unicode.py
```

**Expected Output (Success)**:
```
✓ Terminal encoding supports UTF-8
✓ Picovoice error messages should display correctly
```

**Expected Output (Failure)**:
```
⚠️  WARNING: Terminal encoding is not UTF-8
Recommended fixes:
  1. Set environment variable: export PYTHONIOENCODING=utf-8
  ...
```

### Step 2: Simulate Error Messages

```bash
python3 tests/test_picovoice_errors.py
```

This will display the actual error messages without triggering real Picovoice errors.

**Look for**:
- Clean horizontal lines (━━━━━)
- Warning symbol (⚠️) at start of error titles
- Bullet points (•) in instructions
- No `?` or broken characters

## Detailed Testing Procedures

### Option A: Test with Invalid Access Key (Safest)

1. **Backup current .env**:
   ```bash
   cp .env .env.backup
   ```

2. **Set invalid key temporarily**:
   ```bash
   # Edit .env
   nano .env

   # Change PICOVOICE_ACCESS_KEY to:
   PICOVOICE_ACCESS_KEY=invalid_test_key_12345
   ```

3. **Run the assistant**:
   ```bash
   source venv/bin/activate
   python src/main.py
   ```

4. **Observe the error message**:
   - Should see formatted error with Unicode characters
   - Or see broken characters if encoding not supported

5. **Restore .env**:
   ```bash
   mv .env.backup .env
   ```

### Option B: Check Logs Only

If error messages were previously logged:

```bash
cat logs/assistant.log | grep -A 20 "PICOVOICE"
```

Look for encoding issues like `\u2501` or `?` characters.

## Interpreting Results

### ✅ Success Indicators
- Box lines render as: `━━━━━━━`
- Warning shows as: `⚠️  PICOVOICE...`
- Bullets show as: `  • Item`
- No Python UnicodeEncodeError in logs

### ❌ Failure Indicators
- Box lines render as: `???????` or `-------`
- Warning shows as: `[?]` or question marks
- Python error: `UnicodeEncodeError: 'ascii' codec can't encode`
- Garbled characters in terminal

## Fixes if Unicode Not Supported

### Quick Fix: Environment Variable

Add to Pi's `~/.bashrc`:
```bash
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
```

Then:
```bash
source ~/.bashrc
```

### System Fix: Locale Configuration

```bash
sudo raspi-config
```

Navigate to:
1. **Localisation Options**
2. **Change Locale**
3. Select: `en_US.UTF-8 UTF-8`
4. Set as default
5. **Reboot**

### Code Fix: Graceful Fallback

If the above don't work, we have a prepared patch to add ASCII fallback.

See: `tests/porcupine_unicode_fallback.patch`

This will automatically detect encoding and use ASCII characters (`-`, `[!]`) when UTF-8 is not available.

## Test Results Template

Fill this out after testing:

```
## Raspberry Pi Unicode Test Results

**Date**: _____
**Pi Model**: _____
**OS Version**: _____
**SSH Client**: _____

### Encoding Check (test_pi_unicode.py)
- [ ] UTF-8 supported
- [ ] ASCII only

### Error Simulation (test_picovoice_errors.py)
- [ ] Unicode characters rendered correctly
- [ ] Broken/garbled characters observed

### Live Test (Invalid Key)
- [ ] Error message displayed correctly
- [ ] Encoding issues observed
- [ ] Log file shows correct encoding

### Fix Applied (if needed)
- [ ] Environment variable (PYTHONIOENCODING)
- [ ] Locale configuration (raspi-config)
- [ ] Code patch (ASCII fallback)
- [ ] No fix needed - works out of box

### Screenshots/Logs
(Attach terminal screenshot or log excerpt here)
```

## Next Steps Based on Results

### If Tests Pass
1. ✓ Mark as tested on Raspberry Pi
2. ✓ Update session_log.md with test results
3. ✓ Close testing task

### If Tests Fail
1. Try quick fix (environment variable)
2. If still failing, apply locale configuration
3. If still failing, apply graceful fallback patch
4. Re-test after fix
5. Document fix in session_log.md
6. Consider creating follow-up PR for fallback

## Questions?

- **Q**: Will this break anything if encoding works?
- **A**: No, these are read-only tests. They don't modify any files.

- **Q**: Can I test on macOS first?
- **A**: Yes! Run the same scripts on macOS to see the "correct" output, then compare with Pi.

- **Q**: What if I don't have Picovoice configured?
- **A**: Use `test_picovoice_errors.py` which simulates errors without needing Picovoice.

## References

- PR #1: https://github.com/nicholastripp/openvoice-mcp/pull/1
- Unicode box drawing: https://unicode-table.com/en/2501/
- Python encoding docs: https://docs.python.org/3/library/codecs.html
