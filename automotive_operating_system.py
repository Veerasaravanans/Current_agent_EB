"""
automotive_operating_system.py - ADB-based Android Control System

This module provides ADB commands for controlling Android automotive displays.
Enhanced with multi-directional gestures for the prompt-driven AI agent.

Compatible with the AI agent framework for autonomous automotive UI testing.
"""

import time
import subprocess
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomotiveOperatingSystem:
    """
    ADB-based operating system controller for Android automotive displays.
    
    Supported Operations:
    - tap: Tap at coordinates
    - double_tap: Double tap (for PACCAR media, etc.)
    - long_press: Long press at coordinates
    - swipe: Basic swipe gestures
    - swipe_multi_directional: Diagonal and multi-directional swipes
    - swipe_custom: Custom curved paths
    - input_text: Type text
    - press_key: Press Android keys
    """
    
    def __init__(self, device_serial: Optional[str] = None):
        """
        Initialize automotive OS controller.
        
        Args:
            device_serial: Optional ADB device serial number
        """
        self.device_serial = device_serial
        self.screen_width = 1920  # Default, will be detected
        self.screen_height = 1080  # Default, will be detected
        self._detect_screen_size()
        
        logger.info(f"Automotive OS initialized - Screen: {self.screen_width}x{self.screen_height}")
    
    def _detect_screen_size(self):
        """Detect the actual screen size of the Android device."""
        try:
            cmd = self._build_adb_command(['shell', 'wm', 'size'])
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse output like "Physical size: 1920x1080"
                output = result.stdout.strip()
                if 'x' in output:
                    size_str = output.split(':')[-1].strip()
                    width, height = size_str.split('x')
                    self.screen_width = int(width)
                    self.screen_height = int(height)
                    logger.info(f"Detected screen size: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            logger.warning(f"Could not detect screen size, using defaults: {e}")
    
    def _build_adb_command(self, args: List[str]) -> List[str]:
        """Build ADB command with optional device serial."""
        cmd = ['adb']
        if self.device_serial:
            cmd.extend(['-s', self.device_serial])
        cmd.extend(args)
        return cmd
    
    def _execute_adb(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Execute ADB command and return result."""
        cmd = self._build_adb_command(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                check=check
            )
            return result
        except subprocess.TimeoutExpired:
            logger.error(f"ADB command timed out: {' '.join(cmd)}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"ADB command failed: {e}")
            raise
    
    def tap(self, x: int, y: int, duration: float = 0.0) -> bool:
        """
        Tap at specified coordinates.
        
        Args:
            x: X coordinate (pixels)
            y: Y coordinate (pixels)
            duration: Tap duration in seconds (for long press)
            
        Returns:
            True if successful
        """
        try:
            logger.debug(f"Tapping at ({x}, {y})")
            
            if duration > 0:
                # Convert to milliseconds
                duration_ms = int(duration * 1000)
                cmd = ['shell', 'input', 'swipe', str(x), str(y), str(x), str(y), str(duration_ms)]
            else:
                cmd = ['shell', 'input', 'tap', str(x), str(y)]
            
            result = self._execute_adb(cmd)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to tap at ({x}, {y}): {e}")
            return False
    
    def tap_percent(self, x_percent: float, y_percent: float, duration: float = 0.0) -> bool:
        """
        Tap at coordinates specified as percentages of screen size.
        
        Args:
            x_percent: X position as percentage (0.0 to 1.0)
            y_percent: Y position as percentage (0.0 to 1.0)
            duration: Tap duration in seconds
            
        Returns:
            True if successful
        """
        x = int(self.screen_width * x_percent)
        y = int(self.screen_height * y_percent)
        return self.tap(x, y, duration)
    
    def double_tap(self, x: int, y: int, delay: int = 50) -> bool:
        """
        Double tap at coordinates (for PACCAR Media source access, etc.).
        
        Args:
            x: X coordinate
            y: Y coordinate
            delay: Delay between taps in milliseconds (default 50ms)
            
        Returns:
            True if successful
        """
        try:
            logger.debug(f"Double tapping at ({x}, {y}) with {delay}ms delay")
            self.tap(x, y)
            time.sleep(delay / 1000.0)
            self.tap(x, y)
            return True
        except Exception as e:
            logger.error(f"Failed to double tap: {e}")
            return False
    
    def long_press(self, x: int, y: int, duration: float = 1.0) -> bool:
        """
        Long press at coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Press duration in seconds (default 1.0)
            
        Returns:
            True if successful
        """
        logger.debug(f"Long pressing at ({x}, {y}) for {duration}s")
        return self.tap(x, y, duration)
    
    def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 300) -> bool:
        """
        Swipe from start to end coordinates.
        
        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Swipe duration in milliseconds (default 300)
            
        Returns:
            True if successful
        """
        try:
            logger.debug(f"Swiping from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            
            cmd = [
                'shell', 'input', 'swipe',
                str(start_x), str(start_y),
                str(end_x), str(end_y),
                str(duration)
            ]
            
            result = self._execute_adb(cmd)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to swipe: {e}")
            return False
    
    def swipe_direction(self, direction: str, distance_percent: float = 0.5) -> bool:
        """
        Swipe in a specified direction (basic: up/down/left/right).
        
        Args:
            direction: 'up', 'down', 'left', or 'right'
            distance_percent: Distance as percentage of screen (0.0 to 1.0)
            
        Returns:
            True if successful
        """
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        distance_x = int(self.screen_width * distance_percent)
        distance_y = int(self.screen_height * distance_percent)
        
        swipe_map = {
            'up': (center_x, center_y, center_x, center_y - distance_y),
            'down': (center_x, center_y, center_x, center_y + distance_y),
            'left': (center_x, center_y, center_x - distance_x, center_y),
            'right': (center_x, center_y, center_x + distance_x, center_y),
        }
        
        if direction.lower() not in swipe_map:
            logger.error(f"Invalid swipe direction: {direction}")
            return False
        
        start_x, start_y, end_x, end_y = swipe_map[direction.lower()]
        return self.swipe(start_x, start_y, end_x, end_y)
    
    def swipe_multi_directional(self, direction: str, distance: int = 200, speed: str = 'normal', duration: Optional[int] = None) -> bool:
        """
        Advanced swipe in any direction including diagonals.
        
        Args:
            direction: Direction to swipe:
                - Basic: 'up', 'down', 'left', 'right'
                - Diagonal: 'up_left', 'up_right', 'down_left', 'down_right'
            distance: Distance in pixels
            speed: 'slow' (800ms), 'normal' (300ms), 'fast' (100ms)
            duration: Optional manual duration override in milliseconds
        
        Returns:
            True if successful
        """
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        # Speed to duration mapping
        if duration is None:
            speed_map = {
                'slow': 800,
                'normal': 300,
                'fast': 100
            }
            duration = speed_map.get(speed.lower(), 300)
        
        # Direction to end coordinates mapping
        direction_map = {
            # Basic directions
            'up': (center_x, center_y, center_x, center_y - distance),
            'down': (center_x, center_y, center_x, center_y + distance),
            'left': (center_x, center_y, center_x - distance, center_y),
            'right': (center_x, center_y, center_x + distance, center_y),
            # Diagonal directions
            'up_left': (center_x, center_y, center_x - distance, center_y - distance),
            'up_right': (center_x, center_y, center_x + distance, center_y - distance),
            'down_left': (center_x, center_y, center_x - distance, center_y + distance),
            'down_right': (center_x, center_y, center_x + distance, center_y + distance),
        }
        
        if direction.lower() not in direction_map:
            logger.error(f"Invalid direction: {direction}")
            return False
        
        start_x, start_y, end_x, end_y = direction_map[direction.lower()]
        logger.debug(f"Multi-directional swipe {direction} with speed {speed}, duration {duration}ms")
        return self.swipe(start_x, start_y, end_x, end_y, duration)
    
    def swipe_custom(self, start_x: int, start_y: int, end_x: int, end_y: int, 
                     duration: int = 300, curve_points: Optional[List[Tuple[int, int]]] = None) -> bool:
        """
        Custom swipe with optional curve points for non-linear paths.
        
        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Total duration in milliseconds
            curve_points: Optional list of (x, y) points to create curved path
                         Example: [(mid_x, mid_y)] for a single curve point
        
        Returns:
            True if successful
        """
        if curve_points:
            # Execute swipe through multiple points
            points = [(start_x, start_y)] + curve_points + [(end_x, end_y)]
            segment_duration = duration // (len(points) - 1)
            
            for i in range(len(points) - 1):
                sx, sy = points[i]
                ex, ey = points[i + 1]
                if not self.swipe(sx, sy, ex, ey, segment_duration):
                    return False
                time.sleep(0.05)  # Brief pause between segments
            
            return True
        else:
            # Simple straight swipe
            return self.swipe(start_x, start_y, end_x, end_y, duration)
    
    def input_text(self, text: str) -> bool:
        """
        Input text via ADB.
        
        Args:
            text: Text to input
            
        Returns:
            True if successful
            
        Note: Special characters may need escaping
        """
        try:
            logger.debug(f"Inputting text: {text[:50]}...")
            
            # Replace spaces with %s for ADB
            text_escaped = text.replace(' ', '%s')
            
            # Handle special characters
            text_escaped = text_escaped.replace('&', '\\&')
            text_escaped = text_escaped.replace('(', '\\(')
            text_escaped = text_escaped.replace(')', '\\)')
            
            cmd = ['shell', 'input', 'text', text_escaped]
            result = self._execute_adb(cmd)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to input text: {e}")
            return False
    
    def press_key(self, keycode: str) -> bool:
        """
        Press an Android key.
        
        Args:
            keycode: Android keycode (e.g., 'KEYCODE_HOME', 'KEYCODE_BACK')
            
        Returns:
            True if successful
            
        Common keycodes:
        - KEYCODE_HOME: Home button
        - KEYCODE_BACK: Back button
        - KEYCODE_MENU: Menu button
        - KEYCODE_ENTER: Enter key
        - KEYCODE_DEL: Delete/Backspace
        - KEYCODE_VOLUME_UP: Volume up
        - KEYCODE_VOLUME_DOWN: Volume down
        """
        try:
            logger.debug(f"Pressing key: {keycode}")
            
            cmd = ['shell', 'input', 'keyevent', keycode]
            result = self._execute_adb(cmd)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Failed to press key {keycode}: {e}")
            return False
    
    def press_keys(self, keycodes: List[str]) -> bool:
        """
        Press multiple keys in sequence.
        
        Args:
            keycodes: List of Android keycodes
            
        Returns:
            True if all successful
        """
        for keycode in keycodes:
            if not self.press_key(keycode):
                return False
            time.sleep(0.1)
        return True
    
    def press_home(self) -> bool:
        """Press the home button."""
        return self.press_key('KEYCODE_HOME')
    
    def press_back(self) -> bool:
        """Press the back button."""
        return self.press_key('KEYCODE_BACK')
    
    def press_menu(self) -> bool:
        """Press the menu button."""
        return self.press_key('KEYCODE_MENU')
    
    def press_enter(self) -> bool:
        """Press the enter key."""
        return self.press_key('KEYCODE_ENTER')
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """
        Get screen dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.screen_width, self.screen_height)
    
    def is_device_connected(self) -> bool:
        """
        Check if ADB device is connected.
        
        Returns:
            True if device connected
        """
        try:
            result = self._execute_adb(['get-state'], check=False)
            return result.returncode == 0 and 'device' in result.stdout
        except Exception:
            return False


# Convenience function for backward compatibility
def create_operating_system(device_serial: Optional[str] = None) -> AutomotiveOperatingSystem:
    """
    Create an automotive operating system instance.
    
    Args:
        device_serial: Optional ADB device serial number
        
    Returns:
        AutomotiveOperatingSystem instance
    """
    return AutomotiveOperatingSystem(device_serial)


def main():
    """Test the automotive operating system."""
    import sys
    
    print("Testing Automotive Operating System")
    print("=" * 60)
    
    # Create OS instance
    aos = AutomotiveOperatingSystem()
    
    # Check connection
    if not aos.is_device_connected():
        print("❌ No ADB device connected!")
        sys.exit(1)
    
    print(f"✅ Device connected")
    print(f"📱 Screen size: {aos.screen_width}x{aos.screen_height}")
    
    # Test basic operations
    print("\nTesting operations:")
    print("-" * 60)
    
    # Test tap
    print("1. Testing tap at center...")
    aos.tap(aos.screen_width // 2, aos.screen_height // 2)
    time.sleep(1)
    
    # Test double tap
    print("2. Testing double tap...")
    aos.double_tap(aos.screen_width // 2, aos.screen_height // 2)
    time.sleep(1)
    
    # Test multi-directional swipe
    print("3. Testing diagonal swipe (up_right)...")
    aos.swipe_multi_directional('up_right', distance=150, speed='normal')
    time.sleep(1)
    
    # Test home button
    print("4. Testing home button...")
    aos.press_home()
    time.sleep(1)
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()