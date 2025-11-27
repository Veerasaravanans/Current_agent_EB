"""
voice_interface.py - Voice I/O for AI Agent

Provides Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities.
The agent SPEAKS its thoughts and LISTENS to architect's solutions.

Features:
- TTS: Agent narrates actions (pyttsx3 - offline)
- STT: Architect speaks solutions (SpeechRecognition + Google API)
- Fallback to text if speech unclear
"""

import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Try to import TTS libraries
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("pyttsx3 not installed. Install with: pip install pyttsx3")

# Try to import STT libraries
try:
    import speech_recognition as sr
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    logger.warning("SpeechRecognition not installed. Install with: pip install SpeechRecognition")


class VoiceInterface:
    """
    Voice I/O interface for the AI agent.
    
    The agent can:
    - Speak its thoughts (TTS)
    - Listen to architect's instructions (STT)
    """
    
    def __init__(self, tts_enabled: bool = True, stt_enabled: bool = True):
        """
        Initialize voice interface.
        
        Args:
            tts_enabled: Enable text-to-speech
            stt_enabled: Enable speech-to-text
        """
        self.tts_enabled = tts_enabled and TTS_AVAILABLE
        self.stt_enabled = stt_enabled and STT_AVAILABLE
        
        # Initialize TTS engine
        self.tts_engine = None
        if self.tts_enabled:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure voice properties
                self.tts_engine.setProperty('rate', 150)  # Speed (words per minute)
                self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
                logger.info("✅ TTS initialized (pyttsx3)")
            except Exception as e:
                logger.error(f"Failed to initialize TTS: {e}")
                self.tts_enabled = False
        
        # Initialize STT recognizer
        self.stt_recognizer = None
        self.microphone = None
        if self.stt_enabled:
            try:
                self.stt_recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                # Adjust for ambient noise
                with self.microphone as source:
                    logger.info("Calibrating microphone for ambient noise...")
                    self.stt_recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("✅ STT initialized (SpeechRecognition)")
            except Exception as e:
                logger.error(f"Failed to initialize STT: {e}")
                self.stt_enabled = False
        
        # Speaking flag (to prevent multiple simultaneous speeches)
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        
        logger.info(f"Voice Interface: TTS={'ON' if self.tts_enabled else 'OFF'}, STT={'ON' if self.stt_enabled else 'OFF'}")
    
    def speak(self, text: str, async_mode: bool = False):
        """
        Speak text using TTS.
        
        Args:
            text: Text to speak
            async_mode: If True, speak in background (non-blocking)
        """
        if not self.tts_enabled or not self.tts_engine:
            logger.debug(f"TTS disabled, would say: {text}")
            return
        
        try:
            with self._speaking_lock:
                if self._is_speaking and not async_mode:
                    logger.warning("Already speaking, skipping...")
                    return
                
                self._is_speaking = True
            
            logger.info(f"🔊 Speaking: {text}")
            
            if async_mode:
                # Speak in background thread
                threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()
            else:
                # Speak synchronously
                self._speak_sync(text)
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            with self._speaking_lock:
                self._is_speaking = False
    
    def _speak_sync(self, text: str):
        """Internal synchronous speak method."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        finally:
            with self._speaking_lock:
                self._is_speaking = False
    
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Listen for speech input and convert to text.
        
        Args:
            timeout: Seconds to wait for speech to start
            phrase_time_limit: Maximum seconds for phrase
            
        Returns:
            Recognized text or None if failed/timeout
        """
        if not self.stt_enabled or not self.stt_recognizer or not self.microphone:
            logger.warning("STT not available")
            return None
        
        try:
            logger.info("🎤 Listening...")
            
            with self.microphone as source:
                # Listen for audio
                audio = self.stt_recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
            
            logger.info("Processing speech...")
            
            # Recognize speech using Google Speech Recognition
            text = self.stt_recognizer.recognize_google(audio)
            
            logger.info(f"✅ Recognized: {text}")
            return text
            
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out (no speech detected)")
            return None
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"STT service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected STT error: {e}")
            return None
    
    def listen_with_fallback(self, prompt_callback: Optional[Callable] = None) -> str:
        """
        Listen for speech with fallback to text input.
        
        Args:
            prompt_callback: Optional callback to prompt user for text input
                           Should return text string
        
        Returns:
            User input (from speech or text)
        """
        # Try speech first
        speech_text = self.listen()
        
        if speech_text:
            # Confirm transcription
            self.speak(f"I heard: {speech_text}. Is that correct?")
            # For now, assume correct. In GUI, user can confirm/retry
            return speech_text
        
        # Fallback to text
        logger.info("Speech recognition failed, falling back to text input")
        self.speak("I didn't understand. Please type your response.")
        
        if prompt_callback:
            return prompt_callback()
        else:
            # Console fallback
            return input("Your input: ")
    
    def narrate_action(self, action: str, details: str = ""):
        """
        Narrate an action the agent is taking.
        
        Args:
            action: Action being taken
            details: Optional details
        """
        if details:
            message = f"{action}. {details}"
        else:
            message = action
        
        self.speak(message, async_mode=True)
    
    def request_help(self, problem: str, attempts: int = 10):
        """
        Request help from architect after failed attempts.
        
        Args:
            problem: Description of the problem
            attempts: Number of attempts made
        """
        message = f"Attention Architect. I cannot solve this problem after {attempts} attempts. {problem}. Please provide guidance."
        self.speak(message, async_mode=False)  # Synchronous for important messages
    
    def confirm_solution(self, solution: str):
        """
        Confirm received solution from architect.
        
        Args:
            solution: Solution provided
        """
        message = f"Understood. I will try: {solution}"
        self.speak(message, async_mode=False)
    
    def set_voice_rate(self, rate: int):
        """
        Set speech rate.
        
        Args:
            rate: Words per minute (100-200 typical)
        """
        if self.tts_engine:
            self.tts_engine.setProperty('rate', rate)
            logger.info(f"Voice rate set to {rate} WPM")
    
    def set_voice_volume(self, volume: float):
        """
        Set speech volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        if self.tts_engine:
            self.tts_engine.setProperty('volume', max(0.0, min(1.0, volume)))
            logger.info(f"Voice volume set to {volume}")
    
    def stop_speaking(self):
        """Stop current speech (if speaking)."""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
                with self._speaking_lock:
                    self._is_speaking = False
            except Exception as e:
                logger.error(f"Error stopping speech: {e}")


# Convenience functions
def create_voice_interface(tts_enabled: bool = True, stt_enabled: bool = True) -> VoiceInterface:
    """
    Create a voice interface instance.
    
    Args:
        tts_enabled: Enable text-to-speech
        stt_enabled: Enable speech-to-text
        
    Returns:
        VoiceInterface instance
    """
    return VoiceInterface(tts_enabled=tts_enabled, stt_enabled=stt_enabled)


def main():
    """Test the voice interface."""
    import time
    
    print("=" * 60)
    print("Voice Interface Test")
    print("=" * 60)
    
    # Create interface
    vi = create_voice_interface(tts_enabled=True, stt_enabled=True)
    
    if not (vi.tts_enabled or vi.stt_enabled):
        print("❌ No voice capabilities available!")
        print("Install: pip install pyttsx3 SpeechRecognition pyaudio")
        return
    
    # Test TTS
    if vi.tts_enabled:
        print("\n1. Testing TTS...")
        vi.speak("Hello. I am the automotive AI agent created by Veera Saravanan.")
        time.sleep(1)
        
        vi.narrate_action("Searching for HVAC text using OCR")
        time.sleep(2)
        
        vi.narrate_action("Found AC button", "at coordinates 540, 300")
        time.sleep(2)
    
    # Test STT
    if vi.stt_enabled:
        print("\n2. Testing STT...")
        print("Say something (you have 5 seconds)...")
        
        text = vi.listen(timeout=5)
        
        if text:
            print(f"✅ You said: {text}")
            vi.speak(f"You said: {text}")
        else:
            print("❌ No speech detected or understood")
    
    # Test help request
    if vi.tts_enabled:
        print("\n3. Testing help request...")
        vi.request_help("Cannot find the AC button", attempts=10)
        time.sleep(3)
    
    print("\n✅ Voice interface test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
