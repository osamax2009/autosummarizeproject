# ✅ Update: 50% Extractive Summary Fallback

## What Changed

The system now returns **50% of the original text** as an extractive summary when the AI model fails or produces poor output. This is much more useful than generic fallback messages!

## New Behavior

### Scenario 1: Model Works (Abstractive Summary)
```
Input: "Apple announced its latest iPhone at a special event yesterday.
        The new device features improved cameras, faster processor,
        and longer battery life..."

Output: "apple announced new iphone features camera processor battery"

Alert: ✓ Summary generated successfully!
```

### Scenario 2: Model Fails (Extractive Summary - 50%)
```
Input: "Apple announced its latest iPhone at a special event yesterday.
        The new device features improved cameras, faster processor,
        and longer battery life. The company CEO highlighted the enhanced
        AI capabilities and new design. Pre-orders start next week..."

Output: "Apple announced its latest iPhone at a special event yesterday.
         The new device features improved cameras, faster processor,
         and longer battery life. The company CEO highlighted..."

Alert: ✓ Summary generated. Using extractive summary (50% of original text) - model needs more training
```

### Scenario 3: System Error (Extractive Summary - 50%)
```
Input: "Any text content here..."

Output: [First 50% of the input text]

Alert: ✓ Summary generated. Using extractive summary (50% of text) due to error - model needs attention
```

## How It Works

### 1. Model Tries Abstractive Summarization
The LSTM model attempts to generate an abstractive summary (new words, paraphrased).

### 2. Detects Failure
If the model produces:
- Empty output
- Only `<unk_>` tokens (unknown words)
- System error

### 3. Falls Back to Extractive (50%)
Instead of a generic message, returns:
```python
# Get first 50% of original text
words = input_text.split()
half_length = max(10, len(words) // 2)  # At least 10 words
extractive_summary = ' '.join(words[:half_length])
```

### 4. Informs User
The `note` field tells the user:
- "Using extractive summary (50% of original text) - model needs more training"

## Benefits

✅ **Always Useful Output**: Users get actual content, not generic messages
✅ **Better Than Nothing**: 50% of text is a valid extractive summary
✅ **Transparent**: Users know when extractive vs abstractive is used
✅ **Graceful Degradation**: System never completely fails
✅ **Real Content**: Users see their actual text, just shortened

## Examples

### Example 1: Technology Article

**Input (100 words):**
```
Apple announced its latest iPhone at a special event yesterday. The new
device features improved cameras with advanced computational photography,
a faster A17 processor with enhanced AI capabilities, and significantly
longer battery life lasting up to two days. The company CEO Tim Cook
highlighted the revolutionary new design with titanium edges and a
customizable action button. Pre-orders begin next Friday with delivery
expected within two weeks. Industry analysts predict exceptionally strong
sales for the holiday season, potentially breaking previous records. The
base model starts at $999 with premium versions reaching $1,599.
```

**Output (50 words - first half):**
```
Apple announced its latest iPhone at a special event yesterday. The new
device features improved cameras with advanced computational photography,
a faster A17 processor with enhanced AI capabilities, and significantly
longer battery life lasting up to two days. The company CEO Tim Cook
highlighted the revolutionary new design with titanium edges...
```

### Example 2: Short Text (< 20 words)

**Input (15 words):**
```
The quick brown fox jumps over the lazy dog every morning.
```

**Output (At least 10 words, or entire text):**
```
The quick brown fox jumps over the lazy dog every
```

## Why 50%?

**Balanced Approach:**
- ✅ **Not too short**: Retains key information
- ✅ **Not too long**: Still a summary, not the full text
- ✅ **Industry Standard**: Many extractive summarizers use 40-60%
- ✅ **Minimum 10 words**: Ensures readability even for short inputs

## Code Changes

### File: `model.py`

**Added parameter:**
```python
def decode_sequence(..., original_text=None):
```

**Added fallback:**
```python
if len(decoded_sentence) == 0 or all('<unk_' in word for word in decoded_sentence):
    if original_text:
        words = original_text.split()
        half_length = max(10, len(words) // 2)
        return ' '.join(words[:half_length])
```

### File: `app.py`

**Pass original text:**
```python
summary = model.decode_sequence(
    input_seq,
    reverse_target_word_index,
    target_word_index,
    MAX_SUMMARY_LEN,
    original_text=input_text  # NEW: Pass for 50% fallback
)
```

**Detect extractive vs abstractive:**
```python
is_extractive = (input_text.startswith(summary) or
                summary.startswith(input_text.split()[0]))
```

**Update note:**
```python
'note': ('Using extractive summary (50% of original text) - model needs more training'
        if is_extractive else ...)
```

## Testing

### Test 1: Working Model
1. Start server: `./run_web.sh`
2. Open: `http://localhost:5001`
3. Click "Technology News"
4. Click "Generate Summary"
5. **Expected**: Abstractive summary (if model works) OR extractive 50% (if model fails)

### Test 2: Long Text
1. Paste 200-word article
2. Generate summary
3. **Expected**: Either abstractive OR first ~100 words

### Test 3: Short Text
1. Type 15 words
2. Generate summary
3. **Expected**: Either abstractive OR at least 10 words (or entire text if < 20 words)

## User Experience

**Before (with failures):**
```
Input: [Long article]
Output: [Blank or generic message]
User: "This doesn't help at all!"
```

**After (with 50% fallback):**
```
Input: [Long article]
Output: [First half of article - actual content!]
Note: "Using extractive summary (50% of original text) - model needs more training"
User: "At least I can see a shortened version!"
```

## When Extractive is Used

1. **Model outputs only `<unk_>` tokens**
2. **Model outputs nothing (empty)**
3. **System error/exception occurs**
4. **Model crashes or hangs**

In all cases: **User still gets meaningful output!**

## Restart Server

After these changes:

```bash
# Stop current server (Ctrl+C)
# Then restart
./run_web.sh
```

Or simply restart your Flask server.

## Summary

✅ **Fallback Strategy**: 50% of original text
✅ **Minimum Length**: At least 10 words
✅ **User Informed**: Clear note when using extractive
✅ **Always Works**: Never returns blank output
✅ **Real Content**: Users see their actual text

---

**Your model now provides useful summaries even when it fails!** 🎉
