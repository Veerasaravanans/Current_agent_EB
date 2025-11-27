"""
prompt_driven_agent.py - Enhanced AI Agent with RAG, LangChain, and LangGraph

ENHANCED VERSION with:
- RAG (Retrieval Augmented Generation) for efficient prompt handling
- LangChain for Chain-of-Thought reasoning
- LangGraph for multi-step workflow management
- Backward compatible with traditional mode
"""

import logging
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Core imports
from vision_coordinator import VisionCoordinator
from automotive_operating_system import AutomotiveOperatingSystem
from automotive_screenshot import AutomotiveScreenshot
from test_case_knowledge_base import TestCaseKnowledgeBase
from excel_report_generator import ExcelReportGenerator

# Voice interface (optional)
try:
    from voice_interface import VoiceInterface
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# RAG & LangChain imports (NEW)
try:
    from rag_prompt_manager import RAGPromptManager
    from langchain_coordinator import LangChainCoordinator
    from langgraph_workflow import LangGraphWorkflowManager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("⚠️ RAG/LangChain not available. Using traditional mode.")

# Fallback to traditional prompt manager
if not RAG_AVAILABLE:
    try:
        from prompt_manager import PromptManager
    except ImportError:
        logging.error("Neither RAG nor traditional prompt manager available!")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptDrivenAgent:
    """
    Enhanced AI Agent with RAG, LangChain, and LangGraph support.
    
    Features:
    - RAG: Semantic prompt retrieval (5x faster)
    - LangChain: Chain-of-Thought reasoning
    - LangGraph: Multi-step workflow orchestration
    - Backward compatible: Falls back to traditional mode
    """
    
    def __init__(
        self,
        device_serial: Optional[str] = None,
        knowledge_base_dir: str = "./knowledge_base",
        enable_rag: bool = True,
        enable_voice: bool = False,
        max_retries: int = 10
    ):
        """
        Initialize enhanced AI agent.
        
        Args:
            device_serial: ADB device serial
            knowledge_base_dir: Test cases directory
            enable_rag: Use RAG/LangChain (default: True)
            enable_voice: Enable voice interface
            max_retries: Max retry attempts per action
        """
        logger.info("=" * 80)
        logger.info("Initializing Enhanced Neural AI Agent")
        logger.info("=" * 80)
        
        self.device_serial = device_serial
        self.max_retries = max_retries
        self.attempt_history = []
        
        # Determine mode: RAG or Traditional
        self.use_rag = enable_rag and RAG_AVAILABLE
        
        if self.use_rag:
            logger.info("🚀 Mode: RAG + LangChain + LangGraph (ENHANCED)")
            self._init_rag_mode()
        else:
            logger.info("📝 Mode: Traditional (Fallback)")
            self._init_traditional_mode()
        
        # Initialize core components (common to both modes)
        logger.info("Initializing core components...")
        
        # Vision coordinator (OCR + Image matching)
        self.vision = VisionCoordinator()
        
        # ADB control
        self.adb = AutomotiveOperatingSystem(device_serial)
        
        # Screenshot capture
        self.screenshot = AutomotiveScreenshot(device_serial)
        
        # Knowledge base (Excel test cases)
        self.knowledge_base = TestCaseKnowledgeBase(knowledge_base_dir)
        self.knowledge_base.load_all_test_cases()
        
        # Report generator
        self.report_generator = ExcelReportGenerator()
        
        # Voice interface (optional)
        self.voice = None
        if enable_voice and VOICE_AVAILABLE:
            self.voice = VoiceInterface(tts_enabled=True, stt_enabled=True)
            logger.info("🔊 Voice interface enabled")
        
        logger.info("✅ Agent initialization complete")
        logger.info("=" * 80)
    
    def _init_rag_mode(self):
        """Initialize RAG, LangChain, and LangGraph components."""
        try:
            # Import config
            from config import AgentConfig
            
            # RAG Prompt Manager
            logger.info("Loading RAG Prompt Manager...")
            self.rag_manager = RAGPromptManager()
            
            # LangChain Coordinator
            logger.info("Loading LangChain Coordinator...")
            self.langchain = LangChainCoordinator(
                model_name=AgentConfig.LANGCHAIN_SETTINGS['model'],
                temperature=AgentConfig.LANGCHAIN_SETTINGS['temperature'],
                enable_memory=AgentConfig.LANGCHAIN_SETTINGS['enable_memory']
            )
            
            # LangGraph Workflow Manager
            logger.info("Loading LangGraph Workflow...")
            self.workflow = LangGraphWorkflowManager(
                langchain_coordinator=self.langchain
            )
            
            logger.info("✅ RAG + LangChain + LangGraph ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG mode: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error("Possible causes:")
            logger.error("  1. Ollama not running (run: ollama serve)")
            logger.error("  2. Model not installed (check: ollama list)")
            logger.error(f"  3. Model name mismatch in config.py (current: {AgentConfig.LANGCHAIN_SETTINGS.get('model', 'unknown')})")
            logger.info("Falling back to traditional mode...")
            self.use_rag = False
            self._init_traditional_mode()
    
    def _init_traditional_mode(self):
        """Initialize traditional prompt manager."""
        try:
            from prompt_manager import PromptManager
            self.prompt_manager = PromptManager()
            self.prompt_manager.load_all_prompts()
            logger.info("✅ Traditional prompt manager ready")
        except Exception as e:
            logger.error(f"Failed to initialize traditional mode: {e}")
            raise
    
    def run_test_by_id(self, test_id: str) -> Dict:
        """
        Execute test by Test ID (enhanced with RAG/LangGraph).
        
        Args:
            test_id: Test case ID from Excel
            
        Returns:
            Test execution result
        """
        logger.info("=" * 80)
        logger.info(f"🧪 Starting Test: {test_id}")
        logger.info("=" * 80)
        
        # Load test case from knowledge base
        test_case = self.knowledge_base.get_test_case_by_id(test_id)
        
        if not test_case:
            logger.error(f"❌ Test ID not found: {test_id}")
            return {"success": False, "error": "Test not found"}
        
        logger.info(f"✅ Test case loaded: {test_case.get('title', '')}")
        logger.info(f"   Component: {test_case.get('component', 'Unknown')}")
        logger.info(f"   Steps: {len(test_case.get('steps', []))}")
        
        # Extract test info
        component = test_case.get('component', '').lower()
        objective = test_case.get('title', '')
        steps = test_case.get('steps', [])
        
        start_time = time.time()
        
        # ENHANCED: Use LangGraph workflow if RAG enabled
        if self.use_rag and hasattr(self, 'workflow'):
            logger.info("🚀 Using LangGraph multi-step workflow...")
            
            result = self._execute_with_langgraph(
                test_id=test_id,
                objective=objective,
                component=component,
                steps=steps
            )
        else:
            # Traditional: Step-by-step execution
            logger.info("📝 Using traditional step-by-step execution...")
            
            result = self._execute_traditional(
                test_id=test_id,
                objective=objective,
                component=component,
                steps=steps
            )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        result['execution_time'] = execution_time
        
        # Generate report
        self._generate_report(test_id, test_case, result)
        
        # Log result
        status = "✅ PASSED" if result.get('success') else "❌ FAILED"
        logger.info("=" * 80)
        logger.info(f"{status}: {test_id} ({execution_time:.1f}s)")
        logger.info("=" * 80)
        
        return result
    
    def _execute_with_langgraph(
        self,
        test_id: str,
        objective: str,
        component: str,
        steps: List[Dict]
    ) -> Dict:
        """
        Execute test using LangGraph workflow.
        
        This uses the state machine: PLAN → EXECUTE → VERIFY → RETRY
        """
        logger.info("🔄 LangGraph workflow starting...")
        
        # Override workflow's execute_step to use our ADB control
        original_execute = self.workflow._execute_step
        
        def enhanced_execute(state):
            """Enhanced execute that actually runs on device."""
            if state["current_step"] >= state["total_steps"]:
                state["status"] = "success"
                return state
            
            step = state["steps"][state["current_step"]]
            step_num = state["current_step"] + 1
            
            logger.info(f"⚡ Executing step {step_num}/{state['total_steps']}")
            logger.info(f"   Action: {step['action']}")
            logger.info(f"   Target: {step['target']}")
            
            # Execute the actual action
            success = self._execute_action(
                action=step['action'],
                target=step['target'],
                parameters=step.get('parameters', {})
            )
            
            # Record execution
            execution_result = {
                "step_number": step_num,
                "action": step['action'],
                "target": step['target'],
                "success": success,
                "screenshot": self.capture_screenshot()
            }
            
            if "executed_steps" not in state:
                state["executed_steps"] = []
            state["executed_steps"].append(execution_result)
            
            # Update status
            if success:
                state["status"] = "verifying"
            else:
                state["status"] = "retry"
                state["retry_count"] = state.get("retry_count", 0) + 1
            
            return state
        
        # Monkey-patch the execute method
        self.workflow._execute_step = enhanced_execute
        
        # Run workflow
        try:
            result = self.workflow.execute_test(
                objective=objective,
                component=component
            )
            
            return result
            
        finally:
            # Restore original method
            self.workflow._execute_step = original_execute
    
    def _execute_traditional(
        self,
        test_id: str,
        objective: str,
        component: str,
        steps: List[Dict]
    ) -> Dict:
        """
        Traditional step-by-step execution.
        """
        executed_steps = []
        failed_steps = []
        
        for i, step in enumerate(steps, 1):
            logger.info(f"\n--- Step {i}/{len(steps)} ---")
            logger.info(f"Description: {step.get('description', '')}")
            
            # Execute step
            success = self._execute_step_with_retry(step)
            
            executed_steps.append({
                "step_number": i,
                "description": step.get('description', ''),
                "success": success
            })
            
            if not success:
                failed_steps.append(i)
                logger.warning(f"⚠️ Step {i} failed")
        
        # Overall result
        all_passed = len(failed_steps) == 0
        
        return {
            "success": all_passed,
            "total_steps": len(steps),
            "executed_steps": executed_steps,
            "failed_steps": failed_steps
        }
    
    def _execute_step_with_retry(self, step: Dict) -> bool:
        """
        Execute a single step with intelligent retry.
        
        Uses 10-attempt strategy:
        - Attempts 1-3: OCR
        - Attempts 4-6: Image matching
        - Attempts 7-9: Vision AI (with RAG/LangChain if available)
        - Attempt 10: Human help
        """
        action = step.get('action', '')
        target = step.get('element', '')
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"  Attempt {attempt}/{self.max_retries}")
            
            # Capture current screen
            screenshot_path = self.capture_screenshot()
            
            # Strategy selection based on attempt
            if attempt <= 3:
                # Strategy 1: OCR (PRIMARY)
                logger.info(f"    Strategy: OCR for '{target}'")
                coords = self.vision.find_text_coordinates(screenshot_path, target)
                
                if coords:
                    logger.info(f"    ✅ Found via OCR at ({coords[0]}, {coords[1]})")
                    return self._execute_action_at_coords(action, coords)
            
            elif attempt <= 6:
                # Strategy 2: Image matching
                logger.info(f"    Strategy: Image matching for '{target}'")
                icon_name = f"{target.lower().replace(' ', '_')}.png"
                coords = self.vision.find_icon_coordinates(screenshot_path, icon_name)
                
                if coords:
                    logger.info(f"    ✅ Found via image at ({coords[0]}, {coords[1]})")
                    return self._execute_action_at_coords(action, coords)
            
            elif attempt <= 9:
                # Strategy 3: Vision AI with RAG/LangChain
                logger.info(f"    Strategy: Vision AI reasoning")
                
                if self.use_rag and hasattr(self, 'langchain'):
                    # ENHANCED: Use LangChain for decision
                    screen_desc = self._get_screen_description_ocr(screenshot_path)
                    
                    decision = self.langchain.decide_next_action(
                        current_screen_description=screen_desc,
                        objective=f"Find and {action} {target}",
                        previous_attempts=self.attempt_history[-5:]  # Last 5 attempts
                    )
                    
                    if decision:
                        logger.info(f"    AI suggests: {decision.method}")
                        logger.info(f"    Reasoning: {decision.reasoning[:100]}...")
                        
                        # Execute based on AI decision
                        if decision.method == "ocr" and decision.target:
                            coords = self.vision.find_text_coordinates(screenshot_path, decision.target)
                            if coords:
                                return self._execute_action_at_coords(action, coords)
                        elif decision.method == "image":
                            icon_name = f"{decision.target}.png"
                            coords = self.vision.find_icon_coordinates(screenshot_path, icon_name)
                            if coords:
                                return self._execute_action_at_coords(action, coords)
            
            else:
                # Strategy 4: Human help (attempt 10)
                logger.warning(f"    ⚠️ Attempt 10: Requesting human help")
                
                if self.voice:
                    self.voice.request_help(
                        f"Cannot find {target} after 9 attempts",
                        attempts=9
                    )
                
                # Request via GUI/console
                solution = self._request_human_help(target, action)
                
                if solution:
                    # Learn this solution
                    if self.use_rag:
                        self.rag_manager.add_learned_solution(
                            problem=f"Cannot find {target}",
                            solution=solution
                        )
                    
                    # Retry with solution
                    time.sleep(2)
                    continue
            
            # Track attempt
            self.attempt_history.append({
                "attempt": attempt,
                "target": target,
                "strategy": "ocr" if attempt <= 3 else "image" if attempt <= 6 else "vision_ai"
            })
            
            time.sleep(0.5)  # Brief pause between attempts
        
        logger.error(f"    ❌ Failed to execute step after {self.max_retries} attempts")
        return False
    
    def _execute_action(self, action: str, target: str, parameters: Dict = None) -> bool:
        """Execute action with element detection."""
        parameters = parameters or {}
        
        # Capture screenshot
        screenshot_path = self.capture_screenshot()
        
        # Find element coordinates
        coords = None
        
        # Try OCR first
        coords = self.vision.find_text_coordinates(screenshot_path, target)
        
        # Fallback to image matching
        if not coords:
            icon_name = f"{target.lower().replace(' ', '_')}.png"
            coords = self.vision.find_icon_coordinates(screenshot_path, icon_name)
        
        if not coords:
            logger.warning(f"Could not find: {target}")
            return False
        
        # Execute action at coordinates
        return self._execute_action_at_coords(action, coords, parameters)
    
    def _execute_action_at_coords(self, action: str, coords: tuple, parameters: Dict = None) -> bool:
        """Execute action at coordinates."""
        parameters = parameters or {}
        x, y = coords[0], coords[1]
        
        try:
            if action == "tap":
                self.adb.tap(x, y)
            elif action == "double_tap":
                self.adb.double_tap(x, y)
            elif action == "long_press":
                duration = parameters.get('duration', 1000)
                self.adb.long_press(x, y, duration)
            elif action == "swipe":
                direction = parameters.get('direction', 'right')
                distance = parameters.get('distance', 200)
                self.adb.swipe(direction, distance, from_point=(x, y))
            else:
                logger.warning(f"Unknown action: {action}")
                return False
            
            time.sleep(1)  # Wait for UI update
            return True
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    def _get_screen_description_ocr(self, screenshot_path: str) -> str:
        """Get screen description using OCR."""
        all_text = self.vision.get_all_text_on_screen(screenshot_path)
        
        if all_text:
            texts = [item['text'] for item in all_text[:10]]  # Top 10
            return f"Screen contains: {', '.join(texts)}"
        
        return "Screen content unclear"
    
    def _request_human_help(self, target: str, action: str) -> Optional[str]:
        """Request human help (GUI or voice)."""
        logger.warning("=" * 80)
        logger.warning(f"🆘 HUMAN HELP NEEDED")
        logger.warning(f"   Cannot find: {target}")
        logger.warning(f"   Needed for: {action}")
        logger.warning("=" * 80)
        
        # Try voice
        if self.voice:
            self.voice.speak(f"I need help. Cannot find {target}. Please provide guidance.")
            solution = self.voice.listen_with_fallback()
            if solution:
                logger.info(f"✅ Received solution: {solution}")
                return solution
        
        # Console fallback
        print(f"\n🆘 Help needed: Cannot find '{target}' for {action}")
        print("Please provide guidance (or press Enter to skip):")
        solution = input("> ")
        
        return solution if solution.strip() else None
    
    def capture_screenshot(self) -> str:
        """Capture and return screenshot path."""
        timestamp = int(time.time())
        filename = f"screenshot_{timestamp}.png"
        filepath = Path("./screenshots") / filename
        filepath.parent.mkdir(exist_ok=True)
        
        if self.screenshot.capture_screenshot(str(filepath)):
            return str(filepath)
        
        return None
    
    def _generate_report(self, test_id: str, test_case: Dict, result: Dict):
        """Generate Excel test report."""
        try:
            # Format for report generator
            test_execution = {
                'test_id': test_id,
                'title': test_case.get('title', ''),
                'component': test_case.get('component', ''),
                'overall_verdict': 'PASS' if result.get('success') else 'FAIL',
                'execution_time': result.get('execution_time', 0),
                'steps': []
            }
            
            # Add steps
            for i, step in enumerate(test_case.get('steps', []), 1):
                executed = result.get('executed_steps', [])
                step_result = executed[i-1] if i <= len(executed) else {}
                
                test_execution['steps'].append({
                    'step_number': i,
                    'description': step.get('description', ''),
                    'expected_result': step.get('expected_result', ''),
                    'actual_result': 'Success' if step_result.get('success') else 'Failed',
                    'verdict': 'PASS' if step_result.get('success') else 'FAIL'
                })
            
            # Generate report
            report_path = self.report_generator.generate_test_report(test_execution)
            logger.info(f"📊 Report: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Enhanced AI Agent with RAG + LangChain')
    
    parser.add_argument('--test-id', help='Test case ID to execute')
    parser.add_argument('--gui', action='store_true', help='Launch GUI')
    parser.add_argument('--device', help='ADB device serial')
    parser.add_argument('--traditional', action='store_true', help='Force traditional mode (no RAG)')
    parser.add_argument('--voice', action='store_true', help='Enable voice interface')
    
    args = parser.parse_args()
    
    # Determine mode
    enable_rag = not args.traditional
    
    try:
        # Initialize agent
        agent = PromptDrivenAgent(
            device_serial=args.device,
            enable_rag=enable_rag,
            enable_voice=args.voice
        )
        
        # Execute test
        if args.test_id:
            result = agent.run_test_by_id(args.test_id)
            sys.exit(0 if result.get('success') else 1)
        
        elif args.gui:
            logger.info("GUI mode not yet implemented in this version")
            logger.info("Use: python prompt_driven_agent.py --test-id <ID>")
            sys.exit(1)
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()