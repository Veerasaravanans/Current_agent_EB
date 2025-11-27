"""
gui_controller.py - Enhanced GUI with RAG/LangChain Support

NEW FEATURES:
- RAG mode indicator and toggle
- Real-time prompt statistics
- LangChain reasoning display
- LangGraph workflow visualization
- GPU acceleration status
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox,
        QProgressBar, QGroupBox, QCheckBox, QTabWidget, QTableWidget,
        QTableWidgetItem, QFileDialog, QMessageBox, QStatusBar
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QColor, QPalette
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not installed. Install with: pip install PyQt6")

from prompt_driven_agent import PromptDrivenAgent
from config import AgentConfig

# Check RAG availability
try:
    from rag_prompt_manager import RAGPromptManager
    from automotive_prompts import is_rag_enabled, get_mode_info
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class TestExecutionThread(QThread):
    """Thread for running tests without blocking GUI."""
    
    # Signals
    progress_update = pyqtSignal(str)
    step_completed = pyqtSignal(dict)
    test_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, agent: PromptDrivenAgent, test_id: str):
        super().__init__()
        self.agent = agent
        self.test_id = test_id
    
    def run(self):
        """Execute test in background thread."""
        try:
            self.progress_update.emit(f"Starting test: {self.test_id}")
            
            # Run test
            result = self.agent.run_test_by_id(self.test_id)
            
            self.test_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))


class EnhancedGUI(QMainWindow):
    """
    Enhanced GUI with RAG, LangChain, and LangGraph support.
    
    Features:
    - RAG mode indicator
    - Real-time statistics
    - GPU acceleration status
    - LangChain reasoning display
    - Workflow visualization
    """
    
    def __init__(self):
        super().__init__()
        
        if not PYQT_AVAILABLE:
            print("ERROR: PyQt6 not available")
            sys.exit(1)
        
        print("  → Initializing GUI components...")
        
        self.agent = None
        self.test_thread = None
        
        try:
            print("  → Creating UI...")
            self.init_ui()
            print("  → UI created successfully")
            
            print("  → Checking system status...")
            self.check_system_status()
            print("  → System status checked")
        except Exception as e:
            print(f"  ❌ Error during GUI initialization: {e}")
            import traceback
            traceback.print_exc()
            
            # Create fallback error display
            print("  → Creating fallback error display...")
            try:
                central_widget = QWidget()
                self.setCentralWidget(central_widget)
                layout = QVBoxLayout()
                central_widget.setLayout(layout)
                
                error_label = QLabel(
                    f"<h2>GUI Initialization Error</h2>"
                    f"<p><b>Error:</b> {type(e).__name__}: {e}</p>"
                    f"<p>Check console for full traceback.</p>"
                    f"<p>Try:</p>"
                    f"<ul>"
                    f"<li>Restart Ollama: <code>ollama serve</code></li>"
                    f"<li>Check config.py model name</li>"
                    f"<li>Run: <code>python init_rag_system.py</code></li>"
                    f"</ul>"
                )
                error_label.setWordWrap(True)
                layout.addWidget(error_label)
                
                central_widget.show()
                self.update()
                QApplication.processEvents()
                
                print("  → Fallback error display shown")
            except:
                print("  ❌ Could not even create fallback display")
            
            raise
    
    def init_ui(self):
        """Initialize user interface - SIMPLIFIED WORKING PATTERN."""
        self.setWindowTitle("Neural AI Agent - Enhanced with RAG + LangChain")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        print("    - Building UI...")
        
        # ===== HEADER =====
        header = QGroupBox("System Status")
        header_layout = QHBoxLayout()
        
        self.mode_label = QLabel("Traditional")
        self.mode_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        header_layout.addWidget(QLabel("Mode:"))
        header_layout.addWidget(self.mode_label)
        header_layout.addStretch()
        
        self.gpu_label = QLabel("N/A")
        header_layout.addWidget(QLabel("GPU:"))
        header_layout.addWidget(self.gpu_label)
        
        header.setLayout(header_layout)
        main_layout.addWidget(header)
        
        # ===== TABS =====
        tabs = QTabWidget()
        
        # Control tab
        control_tab = QWidget()
        control_layout = QVBoxLayout()
        
        control_layout.addWidget(QLabel("Test Input:"))
        self.test_input = QLineEdit()
        self.test_input.setPlaceholderText("Enter test IDs (comma-separated)")
        control_layout.addWidget(self.test_input)
        
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("▶ Start Tests")
        self.start_button.clicked.connect(self.start_tests)
        self.stop_button = QPushButton("⏹ Stop Tests")
        self.stop_button.clicked.connect(self.stop_tests)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        control_layout.addLayout(button_layout)
        
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        control_tab.setLayout(control_layout)
        tabs.addTab(control_tab, "Control")
        
        # Log tab
        log_tab = QWidget()
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_tab.setLayout(log_layout)
        tabs.addTab(log_tab, "Log")
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        self.rag_toggle = QCheckBox("Enable RAG Mode")
        self.rag_toggle.setChecked(True)
        settings_layout.addWidget(self.rag_toggle)
        
        settings_layout.addWidget(QLabel("Max Retries:"))
        self.retry_spin = QLineEdit("10")
        settings_layout.addWidget(self.retry_spin)
        
        settings_layout.addStretch()
        settings_tab.setLayout(settings_layout)
        tabs.addTab(settings_tab, "Settings")
        
        # RAG Stats tab
        rag_tab = QWidget()
        rag_layout = QVBoxLayout()
        self.rag_stats_text = QTextEdit()
        self.rag_stats_text.setReadOnly(True)
        rag_layout.addWidget(self.rag_stats_text)
        rag_tab.setLayout(rag_layout)
        tabs.addTab(rag_tab, "RAG Stats")
        
        main_layout.addWidget(tabs)
        
        # ===== STATUS BAR =====
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        
        print("    - UI built successfully")
    
    def start_tests(self):
        """Start test execution."""
        test_ids = self.test_input.text().strip()
        if not test_ids:
            self.log("Please enter test IDs")
            return
        
        self.log(f"Starting tests: {test_ids}")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        # Agent execution will happen here when initialized
        if self.agent:
            # Run tests in background
            pass
        else:
            self.log("Agent not initialized yet")
    
    def stop_tests(self):
        """Stop test execution."""
        self.log("Stopping tests...")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def create_header(self) -> QWidget:
        """Create header with system info."""
        header = QGroupBox("System Status")
        layout = QHBoxLayout()
        
        # Mode indicator
        self.mode_label = QLabel()
        self.mode_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.mode_label)
        
        layout.addStretch()
        
        # GPU status
        self.gpu_label = QLabel()
        layout.addWidget(QLabel("GPU:"))
        layout.addWidget(self.gpu_label)
        
        layout.addStretch()
        
        # RAG toggle
        self.rag_toggle = QCheckBox("Enable RAG")
        self.rag_toggle.setChecked(AgentConfig.RAG_ENABLED)
        self.rag_toggle.stateChanged.connect(self.toggle_rag_mode)
        layout.addWidget(self.rag_toggle)
        
        header.setLayout(layout)
        return header
    
    def create_tabs(self) -> QTabWidget:
        """Create tab widget."""
        tabs = QTabWidget()
        
        tabs.addTab(self.create_test_tab(), "Test Execution")
        tabs.addTab(self.create_rag_tab(), "RAG Statistics")
        tabs.addTab(self.create_settings_tab(), "Settings")
        tabs.addTab(self.create_log_tab(), "Logs")
        
        return tabs
    
    def create_test_tab(self) -> QWidget:
        """Create test execution tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Test ID input
        input_group = QGroupBox("Test Selection")
        input_layout = QHBoxLayout()
        
        input_layout.addWidget(QLabel("Test ID:"))
        self.test_id_input = QLineEdit()
        self.test_id_input.setPlaceholderText("e.g., NAID-24430")
        input_layout.addWidget(self.test_id_input)
        
        self.run_button = QPushButton("Run Test")
        self.run_button.clicked.connect(self.run_test)
        input_layout.addWidget(self.run_button)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Progress
        progress_group = QGroupBox("Execution Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(200)
        progress_layout.addWidget(self.progress_text)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Results
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Step", "Description", "Expected", "Result"
        ])
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_rag_tab(self) -> QWidget:
        """Create RAG statistics tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Statistics display
        stats_group = QGroupBox("RAG Statistics")
        stats_layout = QVBoxLayout()
        
        self.rag_stats_text = QTextEdit()
        self.rag_stats_text.setReadOnly(True)
        stats_layout.addWidget(self.rag_stats_text)
        
        refresh_button = QPushButton("Refresh Statistics")
        refresh_button.clicked.connect(self.refresh_rag_stats)
        stats_layout.addWidget(refresh_button)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Prompt search
        search_group = QGroupBox("Semantic Search")
        search_layout = QVBoxLayout()
        
        search_input_layout = QHBoxLayout()
        search_input_layout.addWidget(QLabel("Query:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("e.g., How to turn on AC?")
        search_input_layout.addWidget(self.search_input)
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search_prompts)
        search_input_layout.addWidget(search_button)
        
        search_layout.addLayout(search_input_layout)
        
        self.search_results = QTextEdit()
        self.search_results.setReadOnly(True)
        search_layout.addWidget(self.search_results)
        
        search_group.setLayout(search_layout)
        layout.addWidget(search_group)
        
        tab.setLayout(layout)
        return tab
    
    def create_settings_tab(self) -> QWidget:
        """Create settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Agent settings
        agent_group = QGroupBox("Agent Settings")
        agent_layout = QVBoxLayout()
        
        # Device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("ADB Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto-detect")
        device_layout.addWidget(self.device_combo)
        
        refresh_devices = QPushButton("Refresh")
        refresh_devices.clicked.connect(self.refresh_devices)
        device_layout.addWidget(refresh_devices)
        
        agent_layout.addLayout(device_layout)
        
        # Voice toggle
        self.voice_toggle = QCheckBox("Enable Voice Interface")
        self.voice_toggle.setChecked(AgentConfig.VOICE_ENABLED)
        agent_layout.addWidget(self.voice_toggle)
        
        # Max retries
        retry_layout = QHBoxLayout()
        retry_layout.addWidget(QLabel("Max Retries:"))
        self.retry_spin = QLineEdit(str(AgentConfig.RETRY_SETTINGS['max_retries']))
        retry_layout.addWidget(self.retry_spin)
        agent_layout.addLayout(retry_layout)
        
        agent_group.setLayout(agent_layout)
        layout.addWidget(agent_group)
        
        # RAG settings
        rag_group = QGroupBox("RAG Settings")
        rag_layout = QVBoxLayout()
        
        context_layout = QHBoxLayout()
        context_layout.addWidget(QLabel("Max Context Size:"))
        self.context_size = QLineEdit(
            str(AgentConfig.RAG_SETTINGS['max_context_size'])
        )
        context_layout.addWidget(self.context_size)
        rag_layout.addLayout(context_layout)
        
        rebuild_button = QPushButton("Rebuild Vector Index")
        rebuild_button.clicked.connect(self.rebuild_index)
        rag_layout.addWidget(rebuild_button)
        
        rag_group.setLayout(rag_layout)
        layout.addWidget(rag_group)
        
        layout.addStretch()
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        tab.setLayout(layout)
        return tab
    
    def create_log_tab(self) -> QWidget:
        """Create log viewer tab."""
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)
        
        # Controls
        controls = QHBoxLayout()
        
        clear_button = QPushButton("Clear Logs")
        clear_button.clicked.connect(self.log_text.clear)
        controls.addWidget(clear_button)
        
        export_button = QPushButton("Export Logs")
        export_button.clicked.connect(self.export_logs)
        controls.addWidget(export_button)
        
        controls.addStretch()
        
        layout.addLayout(controls)
        
        tab.setLayout(layout)
        return tab
    
    def create_status_bar_widget(self) -> QWidget:
        """Create custom status bar widget."""
        widget = QWidget()
        layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        self.performance_label = QLabel()
        layout.addWidget(self.performance_label)
        
        widget.setLayout(layout)
        return widget
    
    def check_system_status(self):
        """Check system status and update UI."""
        # Check RAG mode
        if RAG_AVAILABLE and is_rag_enabled():
            self.mode_label.setText("RAG + LangChain (Enhanced)")
            self.mode_label.setStyleSheet("color: green;")
        else:
            self.mode_label.setText("Traditional (Fallback)")
            self.mode_label.setStyleSheet("color: orange;")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_label.setText(f"✅ {gpu_name}")
                self.gpu_label.setStyleSheet("color: green;")
                self.log("GPU acceleration available!")
            else:
                self.gpu_label.setText("❌ Not available (CPU mode)")
                self.gpu_label.setStyleSheet("color: orange;")
        except:
            self.gpu_label.setText("❌ PyTorch not installed")
            self.gpu_label.setStyleSheet("color: red;")
        
        # Initialize agent
        # DELAY: Use QTimer to initialize after event loop starts
        # This prevents blocking the GUI from showing if initialization fails
        QTimer.singleShot(100, self.initialize_agent)
    
    def initialize_agent(self):
        """Initialize AI agent."""
        try:
            # Defensive check: Ensure widgets exist
            if not hasattr(self, 'log_text'):
                print("ERROR: log_text widget not found!")
                return
            if not hasattr(self, 'rag_toggle'):
                print("ERROR: rag_toggle widget not found!")
                return
            if not hasattr(self, 'retry_spin'):
                print("ERROR: retry_spin widget not found!")
                return
            
            self.log("Initializing AI Agent...")
            self.status_label.setText("Initializing agent...")
            
            enable_rag = self.rag_toggle.isChecked()
            enable_voice = False  # Disable voice in regular GUI
            
            # Get retry value safely
            try:
                max_retries = int(self.retry_spin.text())
            except:
                max_retries = 10  # Default
            
            self.agent = PromptDrivenAgent(
                enable_rag=enable_rag,
                enable_voice=enable_voice,
                max_retries=max_retries
            )
            
            self.log("✅ Agent initialized successfully")
            self.status_label.setText("Agent Ready")
            
            # Load RAG stats if available
            if enable_rag:
                try:
                    self.refresh_rag_stats()
                except Exception as e:
                    self.log(f"Could not load RAG stats: {e}")
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Failed to initialize agent: {e}"
            
            # Print to console (always visible)
            print("\n" + "="*80)
            print("AGENT INITIALIZATION ERROR")
            print("="*80)
            print(f"{type(e).__name__}: {e}")
            print("\nTraceback:")
            traceback.print_exc()
            print("="*80)
            
            # Try to log (might fail if widgets broken)
            try:
                self.log(error_msg)
                self.log(f"Error type: {type(e).__name__}")
                self.log("Traceback:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        self.log(f"  {line}")
                self.status_label.setText("Agent initialization failed")
            except:
                print("Could not write to log widget")
            
            # Show error dialog (try/catch in case it fails)
            try:
                QMessageBox.critical(
                    self, 
                    "Agent Initialization Error", 
                    f"Failed to initialize agent:\n\n{type(e).__name__}: {e}\n\nCheck console for details."
                )
            except Exception as dialog_error:
                print(f"Could not show error dialog: {dialog_error}")
    
    def toggle_rag_mode(self, state):
        """Toggle RAG mode."""
        enable = (state == Qt.CheckState.Checked.value)
        AgentConfig.RAG_ENABLED = enable
        
        # Reinitialize agent
        self.initialize_agent()
    
    def run_test(self):
        """Run test in background thread."""
        test_id = self.test_id_input.text().strip()
        
        if not test_id:
            QMessageBox.warning(self, "Warning", "Please enter a Test ID")
            return
        
        if not self.agent:
            QMessageBox.critical(self, "Error", "Agent not initialized")
            return
        
        # Disable button
        self.run_button.setEnabled(False)
        self.status_label.setText(f"Running test: {test_id}")
        
        # Clear previous results
        self.progress_text.clear()
        self.results_table.setRowCount(0)
        self.progress_bar.setValue(0)
        
        # Create and start thread
        self.test_thread = TestExecutionThread(self.agent, test_id)
        self.test_thread.progress_update.connect(self.update_progress)
        self.test_thread.step_completed.connect(self.update_step)
        self.test_thread.test_completed.connect(self.test_finished)
        self.test_thread.error_occurred.connect(self.test_error)
        
        self.test_thread.start()
    
    def update_progress(self, message: str):
        """Update progress text."""
        self.progress_text.append(message)
        self.log(message)
    
    def update_step(self, step_data: Dict):
        """Update step in results table."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(str(step_data.get('step_number', ''))))
        self.results_table.setItem(row, 1, QTableWidgetItem(step_data.get('description', '')))
        self.results_table.setItem(row, 2, QTableWidgetItem(step_data.get('expected', '')))
        
        result = step_data.get('result', 'FAIL')
        result_item = QTableWidgetItem(result)
        
        if result == 'PASS':
            result_item.setForeground(QColor('green'))
        else:
            result_item.setForeground(QColor('red'))
        
        self.results_table.setItem(row, 3, result_item)
        
        # Update progress bar
        total_steps = step_data.get('total_steps', 1)
        progress = int((row + 1) / total_steps * 100)
        self.progress_bar.setValue(progress)
    
    def test_finished(self, result: Dict):
        """Handle test completion."""
        success = result.get('success', False)
        exec_time = result.get('execution_time', 0)
        
        status = "✅ PASSED" if success else "❌ FAILED"
        message = f"{status} - Test completed in {exec_time:.1f}s"
        
        self.status_label.setText(message)
        self.progress_text.append(f"\n{message}")
        self.progress_bar.setValue(100)
        
        self.log(message)
        
        # Re-enable button
        self.run_button.setEnabled(True)
        
        # Show result dialog
        QMessageBox.information(self, "Test Complete", message)
    
    def test_error(self, error: str):
        """Handle test error."""
        self.status_label.setText(f"❌ Error: {error}")
        self.progress_text.append(f"\n❌ ERROR: {error}")
        
        self.log(f"ERROR: {error}")
        
        self.run_button.setEnabled(True)
        
        QMessageBox.critical(self, "Test Error", f"Test failed with error:\n{error}")
    
    def refresh_rag_stats(self):
        """Refresh RAG statistics."""
        if not RAG_AVAILABLE or not is_rag_enabled():
            self.rag_stats_text.setText("RAG mode not enabled")
            return
        
        try:
            from automotive_prompts import get_mode_info
            
            mode_info = get_mode_info()
            
            stats_text = "=" * 60 + "\n"
            stats_text += "RAG STATISTICS\n"
            stats_text += "=" * 60 + "\n\n"
            
            stats_text += f"Mode: {mode_info.get('mode', 'Unknown')}\n"
            stats_text += f"Description: {mode_info.get('description', '')}\n\n"
            
            stats_text += "Benefits:\n"
            for benefit in mode_info.get('benefits', []):
                stats_text += f"  ✓ {benefit}\n"
            
            stats_text += "\nStatistics:\n"
            stats = mode_info.get('statistics', {})
            for key, value in stats.items():
                stats_text += f"  • {key}: {value}\n"
            
            self.rag_stats_text.setText(stats_text)
            self.log("RAG statistics refreshed")
            
        except Exception as e:
            self.rag_stats_text.setText(f"Error loading statistics:\n{e}")
    
    def search_prompts(self):
        """Search prompts semantically."""
        if not RAG_AVAILABLE or not is_rag_enabled():
            QMessageBox.warning(self, "Warning", "RAG mode not enabled")
            return
        
        query = self.search_input.text().strip()
        
        if not query:
            QMessageBox.warning(self, "Warning", "Please enter a search query")
            return
        
        try:
            from automotive_prompts import search_prompts
            
            self.log(f"Searching: {query}")
            
            results = search_prompts(query, n_results=5)
            
            if results:
                results_text = f"Found {len(results)} relevant chunks:\n\n"
                
                for i, result in enumerate(results, 1):
                    source = result['metadata']['source_file']
                    text = result['text'][:200].replace('\n', ' ')
                    
                    results_text += f"{i}. [{source}]\n"
                    results_text += f"   {text}...\n\n"
                
                self.search_results.setText(results_text)
                self.log(f"Found {len(results)} results")
            else:
                self.search_results.setText("No results found")
                self.log("No results found")
                
        except Exception as e:
            self.search_results.setText(f"Search error:\n{e}")
            self.log(f"Search error: {e}")
    
    def rebuild_index(self):
        """Rebuild RAG vector index."""
        if not RAG_AVAILABLE:
            QMessageBox.warning(self, "Warning", "RAG not available")
            return
        
        reply = QMessageBox.question(
            self,
            "Rebuild Index",
            "This will rebuild the entire vector index. This may take 2-5 minutes. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.log("Rebuilding vector index...")
                self.status_label.setText("Rebuilding index...")
                
                from rag_prompt_manager import get_rag_manager
                manager = get_rag_manager()
                manager.rebuild_index()
                
                self.log("✅ Index rebuilt successfully")
                self.status_label.setText("Index rebuilt")
                
                QMessageBox.information(self, "Success", "Vector index rebuilt successfully")
                
                self.refresh_rag_stats()
                
            except Exception as e:
                self.log(f"❌ Failed to rebuild index: {e}")
                QMessageBox.critical(self, "Error", f"Failed to rebuild index:\n{e}")
    
    def refresh_devices(self):
        """Refresh ADB devices list."""
        # TODO: Implement ADB device detection
        self.log("Refreshing devices...")
        pass
    
    def save_settings(self):
        """Save settings."""
        try:
            # Update config
            AgentConfig.RAG_ENABLED = self.rag_toggle.isChecked()
            AgentConfig.VOICE_ENABLED = self.voice_toggle.isChecked()
            AgentConfig.RETRY_SETTINGS['max_retries'] = int(self.retry_spin.text())
            AgentConfig.RAG_SETTINGS['max_context_size'] = int(self.context_size.text())
            
            self.log("Settings saved")
            QMessageBox.information(self, "Success", "Settings saved successfully")
            
            # Reinitialize agent
            self.initialize_agent()
            
        except Exception as e:
            self.log(f"Failed to save settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
    
    def export_logs(self):
        """Export logs to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Logs",
            "agent_logs.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.log_text.toPlainText())
                
                self.log(f"Logs exported to: {filename}")
                QMessageBox.information(self, "Success", f"Logs exported to:\n{filename}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export logs:\n{e}")
    
    def log(self, message: str):
        """Add message to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def main():
    """Launch GUI."""
    try:
        if not PYQT_AVAILABLE:
            print("❌ ERROR: PyQt6 not installed")
            print("Install with: pip install PyQt6")
            sys.exit(1)
        
        print("=" * 80)
        print("  Neural AI Agent - GUI Mode")
        print("  Enhanced with RAG + LangChain + LangGraph")
        print("=" * 80)
        print("\nStarting GUI...")
        
        # Pre-flight checks
        print("\n✓ Running pre-flight checks...")
        
        # Check 1: Verify imports
        print("  ✓ Checking core imports...")
        try:
            from config import AgentConfig
            print(f"    - Config loaded (Model: {AgentConfig.LANGCHAIN_SETTINGS.get('model', 'unknown')})")
        except Exception as e:
            print(f"    ❌ Config import failed: {e}")
            raise
        
        # Check 2: Verify Ollama connection
        print("  ✓ Checking Ollama connection...")
        try:
            import ollama
            ollama.list()
            print("    - Ollama server: Connected")
        except Exception as e:
            print(f"    ⚠️  Ollama check failed: {e}")
            print("    Note: Agent will work in limited mode")
        
        print("\n✅ Pre-flight checks complete")
        
        app = QApplication(sys.argv)
        
        # Set application style
        app.setStyle('Fusion')
        
        # Create and show window
        print("Creating GUI window...")
        window = EnhancedGUI()
        
        print("Showing GUI window...")
        window.show()
        
        print("✅ GUI launched successfully!")
        print("\nGUI is now running. Window should be visible on your screen.")
        print("If you see a blank window, check the console for errors.\n")
        
        # Keep window open - this blocks until user closes window
        exit_code = app.exec()
        print(f"\nGUI closed with exit code: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR: GUI failed to start")
        print("=" * 80)
        print(f"\nError Details:")
        print(f"  {type(e).__name__}: {e}")
        print("\nFull Traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("Troubleshooting Steps:")
        print("  1. Check Ollama is running: ollama serve")
        print("  2. Verify model installed: ollama list")
        print("  3. Check config.py model name matches installed model")
        print("  4. Verify all dependencies: python init_rag_system.py")
        print("  5. Run diagnostics: python check_gui_dependencies.py")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()