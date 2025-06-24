# Error Context Collector

VS Code extension that automatically extracts Python error context for AI debugging.

## 🚀 Quick Start

1. **Install**: Clone repo, run `npm install`, press `F5` in VS Code
2. **Setup**: Run command `Error Context Collector: Create Error Buffers`
3. **Use**: Add `import vscode_error_buffer` to your Python scripts
4. **Debug**: Errors auto-generate context in `error_definitions.txt`

## ✨ Features

- **Auto Error Detection**: Captures Python runtime errors automatically
- **Smart Context**: Extracts function definitions, types, and references for error symbols
- **AI-Ready**: Generates structured context perfect for AI debugging assistants
- **Deduplication**: Avoids duplicate extractions across symbols

## 📋 How It Works

1. Creates `vscode_error_buffer.py` in your workspace
2. Import it in Python scripts to install error handler
3. On errors, analyzes symbols using VS Code's language server
4. Saves complete context to `error_definitions.txt`

## 🎯 Example Output 
```
// ========================================
// ERROR TRACEBACK (2024-01-15T10:30:45.123Z)
// ========================================
Traceback (most recent call last):
File "main.py", line 15, in process_data
result = analyze_dataset(data, model)
...
// ========================================
// SYMBOL CONTEXT ANALYSIS
// ========================================
// ===== Symbol: analyze_dataset (2024-01-15T10:30:45.456Z) =====
// ----- Definition: analysis.py:42 -----
def analyze_dataset(data: pd.DataFrame, model: MLModel) -> Results:
"""Analyzes dataset using the provided model."""
# ... function implementation
```

## 📁 Generated Files

- `vscode_error_buffer.py`: Python error interception module
- `error_definitions.txt`: Generated context file with:
  - Complete Python tracebacks
  - Function definitions for error-line symbols
  - Type information and references
  - Timestamped sections for easy navigation


## 🔧 Commands

- `Error Context Collector: Create Error Buffers` - Creates Python error buffer
- `Error Context Collector: Extract Context` - Manual context extraction

## 🤖 AI Integration

1. Run your Python code (with `import vscode_error_buffer`)
2. When errors occur, copy `error_definitions.txt` content
3. Paste to your AI assistant for instant debugging help

## 🛠️ Development

```bash
git clone https://github.com/dant2021/a-research/context_debugger
npm install
npm run compile
```

Press `F5` in VS Code to launch Extension Development Host.

## 📁 Project Structure

- `src/extension.ts` - Main extension logic
- `package.json` - Extension manifest
- `.vscode/launch.json` - Debug configuration

Built with VS Code Extension API + TypeScript.
