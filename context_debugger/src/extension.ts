import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';

console.log('🚀 Extension file loaded - about to activate');

const processedDefinitions = new Set<string>(); // Track to avoid duplicates
const maxFileSize = 100 * 1024; // 100KB limit
const contextFileName = 'error_definitions.txt';
const errorBufferFileName = 'vscode_error_buffer.json';

interface SymbolInfo {
    name: string;
    position: vscode.Position;
    range: vscode.Range;
}

interface SymbolContext {
    symbol: string;
    definition?: vscode.Location[];
    declaration?: vscode.Location[];
    typeDefinition?: vscode.Location[];
    references?: vscode.Location[];
    contextLines: number;
}

interface DebugStats {
    totalSymbols: number;
    definitionsExtracted: number;
    declarationsExtracted: number;
    typeDefinitionsExtracted: number;
    referencesExtracted: number;
    totalLinesAdded: number;
    linesPerStep: Map<string, number>;
}

export function activate(context: vscode.ExtensionContext) {
    console.log('🎯 ACTIVATE FUNCTION CALLED');
    console.log('Error Context Collector is now active');
    
    vscode.window.showInformationMessage('Error Context Collector activated - monitoring terminal for errors');

    // Check if terminal data API is available (for logging purposes)
    if ('onDidWriteTerminalData' in vscode.window) {
        console.log('✅ onDidWriteTerminalData is available');
    } else {
        console.warn('❌ onDidWriteTerminalData is not available in this environment');
    }

    // Create error buffer modules in workspace
    createErrorBufferModules();
    
    // Set up diagnostics monitoring and commands
    setupDiagnosticsAndCommands(context);
    
    // Monitor for error buffer file
    watchForErrorBuffer();
}

function createErrorBufferModules() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) {
        console.log('No workspace folder open - error buffer will be created when workspace is opened');
        return; 
    }

    const workspaceRoot = workspaceFolders[0].uri.fsPath;

    // Create Python error buffer only
    const pythonBuffer = `import sys
import traceback
import json
import os
from datetime import datetime

print("[VS Code Error Collector] Python error buffer active")

def log_error_context(exc_type, exc_value, exc_traceback):
    if exc_type is KeyboardInterrupt:
        return
    
    # FIX: Print error normally first, then save
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
    
    error_info = {
        "language": "python",
        "timestamp": datetime.now().isoformat(),
        "error_output": ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
        "exit_code": 1
    }
    
    # Write to error buffer file
    buffer_file = os.path.join(os.getcwd(), "vscode_error_buffer.json")
    with open(buffer_file, 'w') as f:
        json.dump(error_info, f, indent=2)
    
    print(f"[VS Code Error Collector] Error context saved to {buffer_file}")

# Install the error handler
sys.excepthook = log_error_context
`;

    try {
        const fs = require('fs');
        
        // Write Python buffer
        const pythonPath = path.join(workspaceRoot, 'vscode_error_buffer.py');
        fs.writeFileSync(pythonPath, pythonBuffer);
        console.log('Created Python error buffer module');

        vscode.window.showInformationMessage('✅ Python error buffer module created successfully!');
    } catch (error) {
        console.error('Failed to create Python error buffer module:', error);
        vscode.window.showErrorMessage('Failed to create Python error buffer module');
    }
}

function setupDiagnosticsAndCommands(context: vscode.ExtensionContext) {
    // Register commands
    const createBuffersCommand = vscode.commands.registerCommand('errorContextCollector.createBuffers', () => {
        createErrorBufferModules();
    });

    const extractContextCommand = vscode.commands.registerCommand('errorContextCollector.extractContext', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }

        const position = editor.selection.active;
        await analyzeErrorLineSymbols(editor.document.uri.fsPath, position.line);
    });

    context.subscriptions.push(createBuffersCommand, extractContextCommand);
}

function watchForErrorBuffer() {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return;

    const workspaceRoot = workspaceFolders[0].uri.fsPath;
    const bufferPath = path.join(workspaceRoot, errorBufferFileName);

    console.log(`Error buffer file created: ${bufferPath}`);

    // Use file system watcher to detect when error buffer is written
    const watcher = vscode.workspace.createFileSystemWatcher(bufferPath);
    
    watcher.onDidCreate(() => processErrorBuffer(bufferPath));
    watcher.onDidChange(() => processErrorBuffer(bufferPath));
}

async function processErrorBuffer(bufferPath: string) {
    try {
        const fs = require('fs');
        const content = fs.readFileSync(bufferPath, 'utf8');
        const errorInfo = JSON.parse(content);
        
        console.log('Processing Python error buffer:', errorInfo);
        
        if (errorInfo.language === 'python') {
            await processErrorOutput(errorInfo.error_output);
        } else {
            console.log('Ignoring non-Python error:', errorInfo.language);
        }
        
        // Clean up the buffer file
        fs.unlinkSync(bufferPath);
    } catch (error) {
        console.error('Failed to process error buffer:', error);
    }
}

async function processErrorOutput(errorOutput: string) {
    console.log('Processing Python error output:', errorOutput.substring(0, 200) + '...');
    
    // Add pure traceback at the top of the file first
    await addTracebackToTop(errorOutput);
    
    // Extract file paths and line numbers from Python stacktrace
    const errorInfo = extractPythonErrorInfo(errorOutput);
    console.log('Found error info:', errorInfo.length, errorInfo);
    
    if (errorInfo.length === 0) {
        console.log('No error locations found in Python stacktrace');
        return;
    }

    // Find the last meaningful frame
    const meaningfulFrame = findLastMeaningfulFrame(errorInfo);
    console.log('Last meaningful frame:', meaningfulFrame);

    if (meaningfulFrame) {
        // Analyze all symbols on the error line
        await analyzeErrorLineSymbols(meaningfulFrame.filePath, meaningfulFrame.lineNumber);
    }
}

async function addTracebackToTop(errorOutput: string) {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return;

    const workspaceRoot = workspaceFolders[0].uri.fsPath;
    const definitionsPath = path.join(workspaceRoot, contextFileName);

    try {
        const fs = require('fs');
        const timestamp = new Date().toISOString();
        
        const header = `
// ========================================
// ERROR TRACEBACK (${timestamp})
// ========================================

${errorOutput}

// ========================================
// SYMBOL CONTEXT ANALYSIS
// ========================================
`;
        
        // Write to the beginning of the file (or create if doesn't exist)
        fs.writeFileSync(definitionsPath, header);
        
        console.log('✅ Added Python traceback to top of error_definitions.txt');
        
    } catch (error) {
        console.error('Failed to add traceback to file:', error);
    }
}

function findLastMeaningfulFrame(frames: any[]): any | null {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    const workspaceRoot = workspaceFolders ? workspaceFolders[0].uri.fsPath : '';

    // Strategy 1: Last frame in user code
    const userFrames = frames.filter(frame => 
        frame.filePath.includes(workspaceRoot) && 
        !frame.filePath.includes('__pycache__') &&
        !frame.filePath.includes('node_modules')
    );
    
    if (userFrames.length > 0) {
        console.log(`Found ${userFrames.length} user code frames, using last one`);
        return userFrames[userFrames.length - 1];
    }

    // Strategy 2: Last frame before deep library calls
    for (let i = frames.length - 1; i >= 0; i--) {
        const frame = frames[i];
        if (!isDeepLibraryCode(frame.filePath)) {
            console.log(`Using frame ${i} as last meaningful: ${frame.filePath}`);
            return frame;
        }
    }

    // Fallback: Last frame
    console.log('Using fallback: last frame');
    return frames[frames.length - 1];
}

function isDeepLibraryCode(filePath: string): boolean {
    return filePath.includes('site-packages/') ||
           filePath.includes('__pycache__/') ||
           filePath.includes('/usr/lib/python') ||
           filePath.includes('\\Lib\\') ||
           filePath.includes('\\lib\\python');
}

async function analyzeErrorLineSymbols(filePath: string, lineNumber: number) {
    console.log(`🔍 Analyzing symbols on line ${lineNumber} in ${filePath}`);
    
    const debugStats: DebugStats = {
        totalSymbols: 0,
        definitionsExtracted: 0,
        declarationsExtracted: 0,
        typeDefinitionsExtracted: 0,
        referencesExtracted: 0,
        totalLinesAdded: 0,
        linesPerStep: new Map()
    };

    // Global deduplication across all symbols
    const globalExtractedLocations = new Set<string>();

    try {
        const document = await vscode.workspace.openTextDocument(filePath);
        const line = document.lineAt(lineNumber);
        
        console.log(`📝 Error line content: "${line.text}"`);
        
        // Extract all symbols from the line (comments stripped)
        const symbols = extractSymbolsFromLine(line.text, lineNumber);
        debugStats.totalSymbols = symbols.length;

        // Analyze each symbol
        for (const symbol of symbols) {
            console.log(`\n🔬 Analyzing symbol: "${symbol.name}"`);
            const symbolContext = await analyzeSymbol(document, symbol, debugStats);
            await saveSymbolContext(symbolContext, debugStats, globalExtractedLocations);
        }

        // Print debug stats
        printDebugStats(debugStats);

    } catch (error) {
        console.error('Failed to analyze error line symbols:', error);
    }
}

function extractSymbolsFromLine(lineText: string, lineNumber: number): SymbolInfo[] {
    // Remove comments first to avoid parsing comment content
    const cleanLine = lineText.split('#')[0].trim();
    
    // Extract identifiers from the clean line
    const identifierRegex = /[a-zA-Z_][a-zA-Z0-9_]*/g;
    const symbols: SymbolInfo[] = [];
    let match;

    while ((match = identifierRegex.exec(cleanLine)) !== null) {
        const symbolName = match[0];
        const startChar = match.index;
        const endChar = match.index + symbolName.length;
        
        symbols.push({
            name: symbolName,
            position: new vscode.Position(lineNumber, startChar),
            range: new vscode.Range(
                new vscode.Position(lineNumber, startChar),
                new vscode.Position(lineNumber, endChar)
            )
        });
    }

    console.log(`🎯 Found ${symbols.length} symbols (comments stripped): ${symbols.map(s => s.name).join(', ')}`);
    return symbols;
}

async function analyzeSymbol(document: vscode.TextDocument, symbol: SymbolInfo, debugStats: DebugStats): Promise<SymbolContext> {
    const symbolContext: SymbolContext = {
        symbol: symbol.name,
        contextLines: 0
    };

    try {
        // Get definition
        const definitions = await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeDefinitionProvider',
            document.uri,
            symbol.position
        );
        symbolContext.definition = definitions;
        if (definitions && definitions.length > 0) {
            debugStats.definitionsExtracted++;
            console.log(`  ✅ Definition found: ${definitions[0].uri.fsPath}:${definitions[0].range.start.line}`);
        } else {
            console.log(`  ❌ No definition found`);
        }

        // Get declaration
        const declarations = await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeDeclarationProvider',
            document.uri,
            symbol.position
        );
        symbolContext.declaration = declarations;
        if (declarations && declarations.length > 0) {
            debugStats.declarationsExtracted++;
            console.log(`  ✅ Declaration found: ${declarations[0].uri.fsPath}:${declarations[0].range.start.line}`);
        } else {
            console.log(`  ❌ No declaration found`);
        }

        // Get type definition
        const typeDefinitions = await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeTypeDefinitionProvider',
            document.uri,
            symbol.position
        );
        symbolContext.typeDefinition = typeDefinitions;
        if (typeDefinitions && typeDefinitions.length > 0) {
            debugStats.typeDefinitionsExtracted++;
            console.log(`  ✅ Type definition found: ${typeDefinitions[0].uri.fsPath}:${typeDefinitions[0].range.start.line}`);
        } else {
            console.log(`  ❌ No type definition found`);
        }

        // Get references (limit to first 5 to avoid spam)
        const references = await vscode.commands.executeCommand<vscode.Location[]>(
            'vscode.executeReferenceProvider',
            document.uri,
            symbol.position
        );
        symbolContext.references = references?.slice(0, 5);
        if (references && references.length > 0) {
            debugStats.referencesExtracted++;
            console.log(`  ✅ ${references.length} references found (showing first 5)`);
        } else {
            console.log(`  ❌ No references found`);
        }

    } catch (error) {
        console.error(`  ❌ Error analyzing symbol ${symbol.name}:`, error);
    }

    return symbolContext;
}

async function saveSymbolContext(
    symbolContext: SymbolContext, 
    debugStats: DebugStats,
    globalExtractedLocations: Set<string>
) {
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (!workspaceFolders) return;

    const workspaceRoot = workspaceFolders[0].uri.fsPath;
    const definitionsPath = path.join(workspaceRoot, contextFileName);

    let linesAdded = 0;
    
    try {
        const fs = require('fs');
        const timestamp = new Date().toISOString();
        
        // Header for this symbol
        const header = `\n// ===== Symbol: ${symbolContext.symbol} (${timestamp}) =====\n`;
        fs.appendFileSync(definitionsPath, header);
        linesAdded += 2;

        // Collect all unique locations with global deduplication
        const uniqueLocations = new Map<string, { locations: vscode.Location[], type: string }>();
        
        const addLocationsIfUnique = (locations: vscode.Location[] | undefined, type: string) => {
            if (!locations) return;
            
            for (const loc of locations) {
                const key = `${loc.uri.fsPath}:${loc.range.start.line}`;
                
                // Skip if already extracted globally
                if (globalExtractedLocations.has(key)) {
                    console.log(`  🔄 Skipping duplicate location: ${key}`);
                    continue;
                }
                
                if (!uniqueLocations.has(key)) {
                    uniqueLocations.set(key, { locations: [loc], type });
                    globalExtractedLocations.add(key); // Mark as extracted globally
                }
            }
        };

        addLocationsIfUnique(symbolContext.definition, 'Definition');
        addLocationsIfUnique(symbolContext.declaration, 'Declaration');
        addLocationsIfUnique(symbolContext.typeDefinition, 'Type Definition');

        // Extract unique locations only
        for (const [key, { locations, type }] of uniqueLocations) {
            const stepLines = await extractAndSaveLocations(locations, type, definitionsPath);
            linesAdded += stepLines;
            debugStats.linesPerStep.set(`${symbolContext.symbol}-${type}`, stepLines);
        }

        // Handle references separately (limit to 2, also deduplicated)
        if (symbolContext.references && symbolContext.references.length > 0) {
            const uniqueReferences = symbolContext.references.slice(0, 2).filter(ref => {
                const key = `${ref.uri.fsPath}:${ref.range.start.line}`;
                if (globalExtractedLocations.has(key)) {
                    return false;
                }
                globalExtractedLocations.add(key);
                return true;
            });
            
            if (uniqueReferences.length > 0) {
                const stepLines = await extractAndSaveLocations(uniqueReferences, 'References', definitionsPath);
                linesAdded += stepLines;
                debugStats.linesPerStep.set(`${symbolContext.symbol}-references`, stepLines);
            }
        }

        symbolContext.contextLines = linesAdded;
        debugStats.totalLinesAdded += linesAdded;
        
        console.log(`  📊 Added ${linesAdded} lines for symbol "${symbolContext.symbol}" (globally deduplicated)`);

    } catch (error) {
        console.error('Failed to save symbol context:', error);
    }
}

async function extractAndSaveLocations(locations: vscode.Location[], type: string, definitionsPath: string): Promise<number> {
    let totalLines = 0;
    const fs = require('fs');

    for (const location of locations) {
        try {
            const document = await vscode.workspace.openTextDocument(location.uri);
            const definition = extractFunctionDefinition(document, location.range.start.line);
            
            if (definition.trim()) {
                const header = `\n// ----- ${type}: ${path.basename(location.uri.fsPath)}:${location.range.start.line} -----\n`;
                const content = header + definition + '\n';
                
                fs.appendFileSync(definitionsPath, content);
                const lineCount = content.split('\n').length;
                totalLines += lineCount;
                
                console.log(`    📝 Extracted ${lineCount} lines from ${type}`);
            }
        } catch (error) {
            console.error(`Failed to extract ${type} from ${location.uri.fsPath}:`, error);
        }
    }

    return totalLines;
}

function printDebugStats(stats: DebugStats) {
    console.log('\n📊 === DEBUGGING STATS ===');
    console.log(`🎯 Total symbols analyzed: ${stats.totalSymbols}`);
    console.log(`✅ Definitions extracted: ${stats.definitionsExtracted}`);
    console.log(`✅ Declarations extracted: ${stats.declarationsExtracted}`);
    console.log(`✅ Type definitions extracted: ${stats.typeDefinitionsExtracted}`);
    console.log(`✅ References extracted: ${stats.referencesExtracted}`);
    console.log(`📝 Total lines added to context: ${stats.totalLinesAdded}`);
    
    console.log('\n📋 Lines per step:');
    for (const [step, lines] of stats.linesPerStep.entries()) {
        console.log(`  ${step}: ${lines} lines`);
    }
    console.log('========================\n');
}

function extractPythonErrorInfo(errorOutput: string): any[] {
    const errorInfo = [];
    
    // Python stacktrace format: File "path", line N, in function
    const pythonRegex = /File "([^"]+)", line (\d+), in (.+)/g;
    let match;
    
    while ((match = pythonRegex.exec(errorOutput)) !== null) {
        errorInfo.push({
            filePath: match[1],
            lineNumber: parseInt(match[2]) - 1, // VS Code uses 0-based line numbers
            functionName: match[3].trim()
        });
    }
    
    return errorInfo;
}

function extractFunctionDefinition(document: vscode.TextDocument, startLine: number): string {
    try {
        const startLineText = document.lineAt(startLine).text;
        const baseIndent = getIndentLevel(startLineText);
        
        let endLine = startLine;
        
        // Find the end of the function by looking for the next line with same or less indentation
        for (let i = startLine + 1; i < document.lineCount; i++) {
            const line = document.lineAt(i);
            const lineText = line.text.trim();
            
            // Skip empty lines and comments
            if (lineText === '' || lineText.startsWith('#') || lineText.startsWith('//')) {
                continue;
            }
            
            const currentIndent = getIndentLevel(line.text);
            
            // If we find a line with same or less indentation, we've reached the end
            if (currentIndent <= baseIndent) {
                break;
            }
            
            endLine = i;
        }
        
        // Extract the function text
        const range = new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length);
        return document.getText(range);
        
    } catch (error) {
        console.error('Failed to extract function definition:', error);
        return '';
    }
}

function getIndentLevel(line: string): number {
    const match = line.match(/^(\s*)/);
    return match ? match[1].length : 0;
}

export function deactivate() {
    console.log('Error Context Collector deactivated');
} 
