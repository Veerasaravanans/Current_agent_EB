"""
automotive_screenshot.py - ADB-based Screenshot Capture

This module captures screenshots from Android automotive displays via ADB.
Replaces desktop screenshot capture with Android device capture.
"""

import subprocess
import tempfile
import time
import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomotiveScreenshot:
    """
    Captures screenshots from Android automotive displays using ADB.
    
    Methods:
    - capture_screenshot: Capture and save screenshot
    - capture_screenshot_bytes: Capture screenshot as bytes
    - compress_screenshot: Compress screenshot for AI processing
    """
    
    def __init__(self, device_serial: Optional[str] = None):
        """
        Initialize screenshot capturer.
        
        Args:
            device_serial: Optional ADB device serial number
        """
        self.device_serial = device_serial
        self.temp_dir = Path(tempfile.gettempdir()) / "automotive_screenshots"
        self.temp_dir.mkdir(exist_ok=True)
        
        logger.info(f"Screenshot module initialized - Temp dir: {self.temp_dir}")
    
    def _build_adb_command(self, args: list) -> list:
        """Build ADB command with optional device serial."""
        cmd = ['adb']
        if self.device_serial:
            cmd.extend(['-s', self.device_serial])
        cmd.extend(args)
        return cmd
    
    def capture_screenshot(self, output_path: str) -> bool:
        """
        Capture screenshot and save to file.
        
        Args:
            output_path: Path where screenshot will be saved
            
        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.debug(f"Capturing screenshot to: {output_path}")
            
            # Method 1: Direct screencap (faster)
            cmd = self._build_adb_command(['exec-out', 'screencap', '-p'])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                with open(output_path, 'wb') as f:
                    f.write(result.stdout)
                
                logger.debug(f"✅ Screenshot saved: {output_path}")
                return True
            else:
                logger.warning("Direct screencap failed, trying alternative method...")
                return self._capture_screenshot_alternative(output_path)
                
        except subprocess.TimeoutExpired:
            logger.error("Screenshot capture timed out")
            return False
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return False
    
    def _capture_screenshot_alternative(self, output_path: Path) -> bool:
        """
        Alternative screenshot capture method (pull from device).
        
        Args:
            output_path: Path where screenshot will be saved
            
        Returns:
            True if successful
        """
        try:
            device_path = '/sdcard/screenshot_temp.png'
            
            # Capture on device
            cmd_capture = self._build_adb_command(['shell', 'screencap', '-p', device_path])
            result = subprocess.run(cmd_capture, timeout=10)
            
            if result.returncode != 0:
                logger.error("Failed to capture on device")
                return False
            
            # Pull from device
            cmd_pull = self._build_adb_command(['pull', device_path, str(output_path)])
            result = subprocess.run(cmd_pull, timeout=10, capture_output=True)
            
            if result.returncode != 0:
                logger.error("Failed to pull screenshot from device")
                return False
            
            # Clean up device
            cmd_rm = self._build_adb_command(['shell', 'rm', device_path])
            subprocess.run(cmd_rm, timeout=5)
            
            logger.debug(f"✅ Screenshot saved (alternative): {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Alternative capture failed: {e}")
            return False
    
    def capture_screenshot_bytes(self) -> Optional[bytes]:
        """
        Capture screenshot and return as bytes.
        
        Returns:
            Screenshot bytes or None if failed
        """
        try:
            cmd = self._build_adb_command(['exec-out', 'screencap', '-p'])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to capture screenshot bytes: {e}")
            return None
    
    def capture_screenshot_pil(self) -> Optional[Image.Image]:
        """
        Capture screenshot and return as PIL Image.
        
        Returns:
            PIL Image or None if failed
        """
        try:
            screenshot_bytes = self.capture_screenshot_bytes()
            
            if screenshot_bytes:
                return Image.open(io.BytesIO(screenshot_bytes))
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to capture PIL image: {e}")
            return None
    
    def compress_screenshot(
        self,
        input_path: str,
        output_path: str,
        quality: int = 85,
        max_size: Optional[tuple] = None
    ) -> bool:
        """
        Compress screenshot for AI processing.
        
        Args:
            input_path: Input screenshot path
            output_path: Output compressed path
            quality: JPEG quality (1-100, default 85)
            max_size: Optional maximum size as (width, height)
            
        Returns:
            True if successful
        """
        try:
            with Image.open(input_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if max_size specified
                if max_size:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save as JPEG
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
                
                logger.debug(f"✅ Compressed: {input_path} -> {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to compress screenshot: {e}")
            return False
    
    def capture_and_compress(
        self,
        output_path: str,
        quality: int = 85,
        max_size: Optional[tuple] = (1280, 720)
    ) -> bool:
        """
        Capture screenshot and compress in one operation.
        
        Args:
            output_path: Output path for compressed screenshot
            quality: JPEG quality (1-100, default 85)
            max_size: Maximum size (default 1280x720 for AI)
            
        Returns:
            True if successful
        """
        try:
            # Capture to temporary location
            temp_path = self.temp_dir / f"temp_screenshot_{int(time.time())}.png"
            
            if not self.capture_screenshot(str(temp_path)):
                return False
            
            # Compress
            success = self.compress_screenshot(
                str(temp_path),
                output_path,
                quality=quality,
                max_size=max_size
            )
            
            # Clean up
            try:
                temp_path.unlink()
            except:
                pass
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to capture and compress: {e}")
            return False
    
    def get_screen_dimensions(self) -> Optional[tuple]:
        """
        Get screen dimensions from device.
        
        Returns:
            Tuple of (width, height) or None if failed
        """
        try:
            cmd = self._build_adb_command(['shell', 'wm', 'size'])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse output like "Physical size: 1920x1080"
                output = result.stdout.strip()
                if 'x' in output:
                    size_str = output.split(':')[-1].strip()
                    width, height = map(int, size_str.split('x'))
                    return (width, height)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get screen dimensions: {e}")
            return None


def capture_screen_with_cursor(file_path: str, device_serial: Optional[str] = None) -> bool:
    """
    Capture screenshot (compatibility function for original framework).
    
    Note: Android doesn't show cursor in screenshots, but this function
    maintains API compatibility with the desktop version.
    
    Args:
        file_path: Path where screenshot will be saved
        device_serial: Optional ADB device serial
        
    Returns:
        True if successful
    """
    capturer = AutomotiveScreenshot(device_serial)
    return capturer.capture_screenshot(file_path)


def compress_screenshot(
    raw_screenshot_filename: str,
    screenshot_filename: str,
    quality: int = 85
) -> bool:
    """
    Compress screenshot (compatibility function).
    
    Args:
        raw_screenshot_filename: Input screenshot path
        screenshot_filename: Output compressed path
        quality: JPEG quality (1-100, default 85)
        
    Returns:
        True if successful
    """
    capturer = AutomotiveScreenshot()
    return capturer.compress_screenshot(
        raw_screenshot_filename,
        screenshot_filename,
        quality=quality
    )


def main():
    """Test screenshot capture."""
    import sys
    
    print("Testing Automotive Screenshot Capture")
    print("=" * 60)
    
    capturer = AutomotiveScreenshot()
    
    # Get screen dimensions
    dimensions = capturer.get_screen_dimensions()
    if dimensions:
        print(f"📱 Screen size: {dimensions[0]}x{dimensions[1]}")
    else:
        print("❌ Could not detect screen size")
        sys.exit(1)
    
    # Test basic capture
    print("\n1. Testing basic screenshot capture...")
    test_file = Path("test_screenshot.png")
    
    if capturer.capture_screenshot(str(test_file)):
        print(f"✅ Screenshot saved: {test_file}")
        print(f"   File size: {test_file.stat().st_size / 1024:.1f} KB")
    else:
        print("❌ Screenshot capture failed")
        sys.exit(1)
    
    # Test compression
    print("\n2. Testing screenshot compression...")
    compressed_file = Path("test_screenshot_compressed.jpg")
    
    if capturer.compress_screenshot(
        str(test_file),
        str(compressed_file),
        quality=85,
        max_size=(1280, 720)
    ):
        print(f"✅ Compressed screenshot saved: {compressed_file}")
        print(f"   Original: {test_file.stat().st_size / 1024:.1f} KB")
        print(f"   Compressed: {compressed_file.stat().st_size / 1024:.1f} KB")
    else:
        print("❌ Compression failed")
    
    # Test capture and compress
    print("\n3. Testing capture + compress in one operation...")
    combined_file = Path("test_screenshot_combined.jpg")
    
    if capturer.capture_and_compress(str(combined_file)):
        print(f"✅ Combined operation successful: {combined_file}")
        print(f"   File size: {combined_file.stat().st_size / 1024:.1f} KB")
    else:
        print("❌ Combined operation failed")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()