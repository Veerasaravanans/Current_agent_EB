"""
gui_controller_voice.py - GUI with Enhanced Voice Interface

VOICE FEATURES:
- TTS: AI agent speaks actions and thinking
- STT: Architect provides voice commands
- Visual voice indicators
- Microphone activation button
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

from prompt_driven_agent import PromptDrivenAgent
from config import AgentConfig

# Voice interface
try:
    from voice_interface import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️ Voice interface not available")


class VoiceEnabledGUI(QMainWindow):
    """
    GUI with full voice interface support.
    
    Voice Features:
    1. TTS (Text-to-Speech): Agent speaks its actions
    2. STT (Speech-to-Text): Architect speaks commands
    3. Visual indicators for voice activity
    4. Microphone button for voice input
    """
    
    def __init__(self):
        super().__init__()
        
        if not PYQT_AVAILABLE:
            print("ERROR: PyQt6 not available")
            sys.exit(1)
        
        print("  → Initializing GUI components...")
        
        self.agent = None
        self.voice = None
        self.voice_enabled = False
        
        try:
            print("  → Creating UI...")
            self.init_ui()
            print("  → UI created successfully")
            
            print("  → Initializing voice...")
            self.init_voice()
            print("  → Voice initialized")
            
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
        """Initialize UI with voice controls - SIMPLIFIED WORKING PATTERN."""
        self.setWindowTitle("Neural AI Agent - Voice Enabled GUI")
        self.setGeometry(100, 100, 1200, 850)
        
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
        header_layout.addStretch()
        
        self.voice_status_label = QLabel("Disabled")
        self.voice_status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        header_layout.addWidget(QLabel("🎤 Voice:"))
        header_layout.addWidget(self.voice_status_label)
        
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
        
        main_layout.addWidget(tabs)
        
        # ===== VOICE PANEL =====
        voice_panel = QGroupBox("🎤 Voice Interface")
        voice_layout = QHBoxLayout()
        
        self.voice_toggle = QCheckBox("Enable Voice")
        self.voice_toggle.setChecked(False)
        self.voice_toggle.stateChanged.connect(self.toggle_voice)
        voice_layout.addWidget(self.voice_toggle)
        
        self.tts_indicator = QLabel("🔊 TTS: OFF")
        voice_layout.addWidget(self.tts_indicator)
        
        self.stt_indicator = QLabel("🎤 STT: OFF")
        voice_layout.addWidget(self.stt_indicator)
        
        voice_layout.addStretch()
        voice_panel.setLayout(voice_layout)
        main_layout.addWidget(voice_panel)
        
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
    
    def toggle_voice(self):
        """Toggle voice interface on/off."""
        if self.voice_toggle.isChecked():
            self.log("Voice enabled")
            self.voice_status_label.setText("Enabled")
            self.tts_indicator.setText("🔊 TTS: ON")
            self.stt_indicator.setText("🎤 STT: ON")
        else:
            self.log("Voice disabled")
            self.voice_status_label.setText("Disabled")
            self.tts_indicator.setText("🔊 TTS: OFF")
            self.stt_indicator.setText("🎤 STT: OFF")
    
    def create_header_with_voice(self) -> QWidget:
        """Create header with voice status indicator."""
        header = QGroupBox("System Status")
        layout = QHBoxLayout()
        
        # Mode
        self.mode_label = QLabel()
        self.mode_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(QLabel("Mode:"))
        layout.addWidget(self.mode_label)
        
        layout.addStretch()
        
        # GPU
        self.gpu_label = QLabel()
        layout.addWidget(QLabel("GPU:"))
        layout.addWidget(self.gpu_label)
        
        layout.addStretch()
        
        # Voice status (NEW!)
        self.voice_status_label = QLabel()
        self.voice_status_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(QLabel("🎤 Voice:"))
        layout.addWidget(self.voice_status_label)
        
        header.setLayout(layout)
        return header
    
    def create_voice_panel(self) -> QWidget:
        """Create voice control panel at bottom."""
        panel = QGroupBox("🎤 Voice Interface")
        layout = QHBoxLayout()
        
        # Voice toggle
        self.voice_toggle = QCheckBox("Enable Voice")
        self.voice_toggle.setChecked(AgentConfig.VOICE_ENABLED)
        self.voice_toggle.stateChanged.connect(self.toggle_voice)
        layout.addWidget(self.voice_toggle)
        
        # TTS indicator
        self.tts_indicator = QLabel("🔊 TTS: OFF")
        layout.addWidget(self.tts_indicator)
        
        # STT indicator
        self.stt_indicator = QLabel("🎙️ STT: OFF")
        layout.addWidget(self.stt_indicator)
        
        layout.addStretch()
        
        # Microphone button
        self.mic_button = QPushButton("🎤 Speak Command")
        self.mic_button.setEnabled(False)
        self.mic_button.clicked.connect(self.activate_microphone)
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        layout.addWidget(self.mic_button)
        
        # Voice log
        self.voice_log = QTextEdit()
        self.voice_log.setReadOnly(True)
        self.voice_log.setMaximumHeight(100)
        self.voice_log.setPlaceholderText("Voice activity will appear here...")
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(QLabel("Voice Activity Log:"))
        main_layout.addWidget(self.voice_log)
        
        panel.setLayout(main_layout)
        return panel
    
    def create_tabs(self) -> QTabWidget:
        """Create tab widget."""
        tabs = QTabWidget()
        
        # Test execution tab
        test_tab = QWidget()
        test_layout = QVBoxLayout()
        
        # Test input
        input_group = QGroupBox("Test Selection")
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Test ID:"))
        self.test_id_input = QLineEdit()
        self.test_id_input.setPlaceholderText("e.g., NAID-24430")
        input_layout.addWidget(self.test_id_input)
        
        self.run_button = QPushButton("▶️ Run Test")
        self.run_button.clicked.connect(self.run_test)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        input_layout.addWidget(self.run_button)
        
        input_group.setLayout(input_layout)
        test_layout.addWidget(input_group)
        
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
        test_layout.addWidget(progress_group)
        
        # Results table
        results_group = QGroupBox("Test Results")
        results_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Step", "Description", "Expected", "Result"])
        results_layout.addWidget(self.results_table)
        
        results_group.setLayout(results_layout)
        test_layout.addWidget(results_group)
        
        test_tab.setLayout(test_layout)
        tabs.addTab(test_tab, "🧪 Test Execution")
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        # Voice settings group
        voice_settings = QGroupBox("🎤 Voice Settings")
        voice_layout = QVBoxLayout()
        
        # TTS settings
        tts_group = QGroupBox("Text-to-Speech (Agent Speaking)")
        tts_layout = QVBoxLayout()
        
        self.tts_enable = QCheckBox("Enable TTS (Agent narrates actions)")
        self.tts_enable.setChecked(True)
        tts_layout.addWidget(self.tts_enable)
        
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Speech Rate:"))
        self.tts_rate = QLineEdit("150")
        self.tts_rate.setPlaceholderText("Words per minute (100-200)")
        rate_layout.addWidget(self.tts_rate)
        tts_layout.addLayout(rate_layout)
        
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.tts_volume = QLineEdit("0.9")
        self.tts_volume.setPlaceholderText("0.0 to 1.0")
        volume_layout.addWidget(self.tts_volume)
        tts_layout.addLayout(volume_layout)
        
        tts_group.setLayout(tts_layout)
        voice_layout.addWidget(tts_group)
        
        # STT settings
        stt_group = QGroupBox("Speech-to-Text (Architect Speaking)")
        stt_layout = QVBoxLayout()
        
        self.stt_enable = QCheckBox("Enable STT (Voice commands)")
        self.stt_enable.setChecked(True)
        stt_layout.addWidget(self.stt_enable)
        
        timeout_layout = QHBoxLayout()
        timeout_layout.addWidget(QLabel("Listen Timeout:"))
        self.stt_timeout = QLineEdit("5")
        self.stt_timeout.setPlaceholderText("Seconds")
        timeout_layout.addWidget(self.stt_timeout)
        stt_layout.addLayout(timeout_layout)
        
        stt_group.setLayout(stt_layout)
        voice_layout.addWidget(stt_group)
        
        # Test voice button
        test_voice_btn = QPushButton("🔊 Test Voice")
        test_voice_btn.clicked.connect(self.test_voice)
        voice_layout.addWidget(test_voice_btn)
        
        voice_settings.setLayout(voice_layout)
        settings_layout.addWidget(voice_settings)
        
        # Agent settings
        agent_settings = QGroupBox("Agent Settings")
        agent_layout = QVBoxLayout()
        
        self.rag_toggle = QCheckBox("Enable RAG Mode")
        self.rag_toggle.setChecked(AgentConfig.RAG_ENABLED)
        agent_layout.addWidget(self.rag_toggle)
        
        retry_layout = QHBoxLayout()
        retry_layout.addWidget(QLabel("Max Retries:"))
        self.retry_spin = QLineEdit(str(AgentConfig.RETRY_SETTINGS['max_retries']))
        retry_layout.addWidget(self.retry_spin)
        agent_layout.addLayout(retry_layout)
        
        agent_settings.setLayout(agent_layout)
        settings_layout.addWidget(agent_settings)
        
        settings_layout.addStretch()
        
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings)
        settings_layout.addWidget(save_btn)
        
        settings_tab.setLayout(settings_layout)
        tabs.addTab(settings_tab, "⚙️ Settings")
        
        # Logs tab
        logs_tab = QWidget()
        logs_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        logs_layout.addWidget(self.log_text)
        
        logs_tab.setLayout(logs_layout)
        tabs.addTab(logs_tab, "📋 Logs")
        
        return tabs
    
    def init_voice(self):
        """Initialize voice interface."""
        if not VOICE_AVAILABLE:
            self.voice_status_label.setText("❌ Not Available")
            self.voice_status_label.setStyleSheet("color: red;")
            self.log("⚠️ Voice interface not available")
            return
        
        try:
            self.voice = VoiceInterface(
                tts_enabled=True,
                stt_enabled=True
            )
            self.voice_status_label.setText("✅ Ready")
            self.voice_status_label.setStyleSheet("color: green;")
            self.log("✅ Voice interface initialized")
        except Exception as e:
            self.voice_status_label.setText("❌ Error")
            self.voice_status_label.setStyleSheet("color: red;")
            self.log(f"❌ Voice init failed: {e}")
    
    def toggle_voice(self, state):
        """Toggle voice interface on/off."""
        self.voice_enabled = (state == Qt.CheckState.Checked.value)
        
        if self.voice_enabled:
            if self.voice:
                self.tts_indicator.setText("🔊 TTS: ON")
                self.tts_indicator.setStyleSheet("color: green;")
                self.stt_indicator.setText("🎙️ STT: ON")
                self.stt_indicator.setStyleSheet("color: green;")
                self.mic_button.setEnabled(True)
                
                self.voice_log_message("🎤 Voice interface ENABLED")
                
                if self.voice:
                    self.voice.speak("Voice interface enabled")
            else:
                QMessageBox.warning(
                    self,
                    "Voice Not Available",
                    "Voice interface is not available on this system.\n\n"
                    "Please install:\n"
                    "  pip install pyttsx3 SpeechRecognition pyaudio"
                )
                self.voice_toggle.setChecked(False)
        else:
            self.tts_indicator.setText("🔊 TTS: OFF")
            self.tts_indicator.setStyleSheet("color: gray;")
            self.stt_indicator.setText("🎙️ STT: OFF")
            self.stt_indicator.setStyleSheet("color: gray;")
            self.mic_button.setEnabled(False)
            
            self.voice_log_message("🔇 Voice interface DISABLED")
    
    def test_voice(self):
        """Test voice interface."""
        if not self.voice:
            QMessageBox.warning(self, "Error", "Voice not initialized")
            return
        
        # Test TTS
        if self.tts_enable.isChecked():
            self.voice_log_message("🔊 Testing TTS...")
            self.voice.speak("Voice interface is working correctly. Text to speech test.")
        
        # Test STT
        if self.stt_enable.isChecked():
            self.voice_log_message("🎙️ Testing STT... Please speak now")
            
            result = self.voice.listen_with_fallback(timeout=5)
            
            if result:
                self.voice_log_message(f"🎙️ Heard: {result}")
                self.voice.speak(f"I heard: {result}")
            else:
                self.voice_log_message("🎙️ No speech detected")
    
    def activate_microphone(self):
        """Activate microphone for voice command."""
        if not self.voice or not self.voice_enabled:
            return
        
        self.mic_button.setText("🎙️ Listening...")
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
            }
        """)
        
        self.voice_log_message("🎙️ Listening for command...")
        self.voice.speak("Listening for your command")
        
        # Listen for command
        command = self.voice.listen_with_fallback(timeout=10)
        
        if command:
            self.voice_log_message(f"✅ Command: {command}")
            self.voice.speak(f"Received: {command}")
            
            # Process command (can be extended)
            self.process_voice_command(command)
        else:
            self.voice_log_message("❌ No command detected")
            self.voice.speak("No command detected")
        
        # Reset button
        self.mic_button.setText("🎤 Speak Command")
        self.mic_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 14px;
            }
        """)
    
    def process_voice_command(self, command: str):
        """Process voice command from architect."""
        command_lower = command.lower()
        
        # Simple command processing
        if "run test" in command_lower:
            self.voice_log_message("📝 Executing: Run test")
            self.run_test()
        
        elif "stop" in command_lower or "cancel" in command_lower:
            self.voice_log_message("🛑 Executing: Stop test")
            # Stop test logic here
        
        else:
            self.voice_log_message(f"📝 Custom command: {command}")
            # Send to agent as solution
    
    def voice_log_message(self, message: str):
        """Add message to voice log."""
        timestamp = time.strftime("%H:%M:%S")
        self.voice_log.append(f"[{timestamp}] {message}")
        self.log(message)
    
    def check_system_status(self):
        """Check system status."""
        # Check RAG mode
        try:
            from automotive_prompts import is_rag_enabled
            if is_rag_enabled():
                self.mode_label.setText("RAG + LangChain")
                self.mode_label.setStyleSheet("color: green;")
            else:
                self.mode_label.setText("Traditional")
                self.mode_label.setStyleSheet("color: orange;")
        except:
            self.mode_label.setText("Unknown")
        
        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self.gpu_label.setText(f"✅ {gpu_name}")
                self.gpu_label.setStyleSheet("color: green;")
                self.log(f"GPU detected: {gpu_name}")
            else:
                self.gpu_label.setText("❌ CPU Mode")
                self.gpu_label.setStyleSheet("color: orange;")
        except:
            self.gpu_label.setText("❌ No PyTorch")
        
        # Initialize agent
        # DELAY: Use QTimer to initialize after event loop starts
        # This prevents blocking the GUI from showing if initialization fails
        QTimer.singleShot(100, self.initialize_agent)
    
    def initialize_agent(self):
        """Initialize agent with voice support."""
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
            
            self.log("Initializing agent...")
            
            # Get retry value safely
            try:
                max_retries = int(self.retry_spin.text())
            except:
                max_retries = 10  # Default
            
            self.agent = PromptDrivenAgent(
                enable_rag=self.rag_toggle.isChecked(),
                enable_voice=self.voice_enabled,
                max_retries=max_retries
            )
            
            self.log("✅ Agent initialized")
            
            if self.voice_enabled and self.voice:
                self.voice.speak("Agent initialized successfully")
                
        except Exception as e:
            import traceback
            error_msg = f"❌ Failed to initialize: {e}"
            
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
            except:
                print("Could not write to log widget")
            
            if self.voice_enabled and self.voice:
                try:
                    self.voice.speak("Agent initialization failed")
                except:
                    pass
            
            # Show error dialog (try/catch in case it fails)
            try:
                QMessageBox.critical(
                    self, 
                    "Agent Initialization Error", 
                    f"Failed to initialize agent:\n\n{type(e).__name__}: {e}\n\nCheck console for details."
                )
            except Exception as dialog_error:
                print(f"Could not show error dialog: {dialog_error}")
    
    def run_test(self):
        """Run test with voice narration."""
        test_id = self.test_id_input.text().strip()
        
        if not test_id:
            QMessageBox.warning(self, "Warning", "Please enter a Test ID")
            return
        
        if not self.agent:
            QMessageBox.critical(self, "Error", "Agent not initialized")
            return
        
        self.log(f"Starting test: {test_id}")
        
        if self.voice_enabled and self.voice:
            self.voice.speak(f"Starting test {test_id}")
        
        # Run test (simplified for demo)
        # In real implementation, this would be in a thread
        try:
            result = self.agent.run_test_by_id(test_id)
            
            if result.get('success'):
                self.log(f"✅ Test {test_id} PASSED")
                if self.voice_enabled and self.voice:
                    self.voice.speak(f"Test {test_id} passed successfully")
            else:
                self.log(f"❌ Test {test_id} FAILED")
                if self.voice_enabled and self.voice:
                    self.voice.speak(f"Test {test_id} failed")
                    
        except Exception as e:
            self.log(f"❌ Test error: {e}")
            if self.voice_enabled and self.voice:
                self.voice.speak(f"Test failed with error")
    
    def save_settings(self):
        """Save settings."""
        try:
            # Update voice settings
            if self.voice:
                AgentConfig.VOICE_SETTINGS['tts_rate'] = int(self.tts_rate.text())
                AgentConfig.VOICE_SETTINGS['tts_volume'] = float(self.tts_volume.text())
                AgentConfig.VOICE_SETTINGS['stt_timeout'] = int(self.stt_timeout.text())
            
            # Update agent settings
            AgentConfig.RAG_ENABLED = self.rag_toggle.isChecked()
            AgentConfig.VOICE_ENABLED = self.voice_enabled
            AgentConfig.RETRY_SETTINGS['max_retries'] = int(self.retry_spin.text())
            
            self.log("Settings saved")
            QMessageBox.information(self, "Success", "Settings saved successfully")
            
            if self.voice_enabled and self.voice:
                self.voice.speak("Settings saved")
            
            # Reinitialize
            self.initialize_agent()
            
        except Exception as e:
            self.log(f"Failed to save: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
    
    def log(self, message: str):
        """Add to log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def main():
    """Launch voice-enabled GUI."""
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
        app.setStyle('Fusion')
        
        print("Creating GUI window...")
        window = VoiceEnabledGUI()
        
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