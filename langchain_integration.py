"""
langchain_integration.py - LangChain Integration for Heavy Lifting

This module uses LangChain to:
- Orchestrate complex multi-step testing workflows
- Chain together vision, OCR, and action decisions
- Memory management across test steps
- Tool calling and agent loops
- Error recovery with learned solutions

LangChain handles the complexity so llava:7b can focus on vision!
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

# LangChain core
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain_community.llms import Ollama

# LangChain tools
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ActionDecision(BaseModel):
    """Structured action decision output."""
    action_type: str = Field(description="Type of action: tap, swipe, double_tap, etc.")
    target: str = Field(description="Target element to interact with")
    coordinates: Optional[Tuple[int, int]] = Field(default=None, description="X,Y coordinates if known")
    method: str = Field(description="Detection method: ocr, image, vision_ai")
    confidence: float = Field(description="Confidence score 0.0-1.0")
    reasoning: str = Field(description="Chain-of-thought reasoning for decision")


class ActionOutputParser(BaseOutputParser[ActionDecision]):
    """Parse LLM output into structured ActionDecision."""
    
    def parse(self, text: str) -> ActionDecision:
        """Parse text output into ActionDecision."""
        # Simple parsing - can be enhanced with JSON parsing
        lines = text.strip().split('\n')
        
        action_type = "tap"
        target = "unknown"
        method = "ocr"
        confidence = 0.5
        reasoning = text
        
        for line in lines:
            if "action:" in line.lower():
                action_type = line.split(":")[-1].strip().lower()
            elif "target:" in line.lower():
                target = line.split(":")[-1].strip()
            elif "method:" in line.lower():
                method = line.split(":")[-1].strip().lower()
            elif "confidence:" in line.lower():
                try:
                    confidence = float(line.split(":")[-1].strip())
                except:
                    confidence = 0.5
        
        return ActionDecision(
            action_type=action_type,
            target=target,
            method=method,
            confidence=confidence,
            reasoning=reasoning
        )


class LangChainTestOrchestrator:
    """
    LangChain-powered test orchestrator.
    
    Uses chains and agents to orchestrate complex testing workflows.
    """
    
    def __init__(
        self,
        model_name: str = "llava:7b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize LangChain orchestrator.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
        """
        self.model_name = model_name
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.1  # Low temperature for consistent behavior
        )
        
        # Memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Action parser
        self.action_parser = ActionOutputParser()
        
        logger.info(f"LangChain orchestrator initialized with {model_name}")
    
    def create_action_decision_chain(self) -> LLMChain:
        """
        Create chain for deciding next action.
        
        This chain uses Chain-of-Thought prompting.
        """
        template = """You are an automotive testing AI agent. Use chain-of-thought reasoning.

Current Test Objective: {objective}
Current Screen State: {screen_state}
Previous Actions: {previous_actions}
Attempt Number: {attempt_number}

Think step-by-step:

1. UNDERSTAND: What needs to be done?
2. ANALYZE: What's visible on screen?
3. PLAN: What's the best approach?
   - Priority 1: Vision AI (for complex visual analysis)
   - Priority 2: OCR (if target has text)
   - Priority 3: Image matching (if target is icon)
4. DECIDE: Choose action

Output your decision in this format:
Action: [tap/swipe/double_tap/long_press]
Target: [element name or description]
Method: [ocr/image/vision_ai]
Confidence: [0.0-1.0]
Reasoning: [your step-by-step thinking]

Decision:"""
        
        prompt = PromptTemplate(
            input_variables=["objective", "screen_state", "previous_actions", "attempt_number"],
            template=template
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=self.action_parser
        )
        
        return chain
    
    def create_verification_chain(self) -> LLMChain:
        """Create chain for result verification."""
        template = """You are verifying a test action result.

Action Taken: {action_taken}
Expected Result: {expected_result}
Observed Screen State: {observed_state}

Compare expected vs observed. Think carefully:

1. Does the observed state match expectations?
2. Are there any indicators of success (color changes, text, icons)?
3. Could there be a delay - should we wait longer?
4. Is there an error or unexpected state?

Output:
Verification: [PASS/FAIL/UNCERTAIN]
Confidence: [0.0-1.0]
Reasoning: [explain your decision]

Analysis:"""
        
        prompt = PromptTemplate(
            input_variables=["action_taken", "expected_result", "observed_state"],
            template=template
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        return chain
    
    def create_error_recovery_chain(self) -> LLMChain:
        """Create chain for error recovery suggestions."""
        template = """An action has failed. Suggest recovery strategy.

Failed Action: {failed_action}
Failure Reason: {failure_reason}
Attempts So Far: {attempts}
Known Solutions: {known_solutions}

Think through alternatives:

1. What might be wrong?
2. What variations can we try?
3. Which recovery strategy is most likely to work?

Suggest next approach:
Strategy: [describe alternative approach]
Confidence: [0.0-1.0]
Reasoning: [why this should work]

Recovery Plan:"""
        
        prompt = PromptTemplate(
            input_variables=["failed_action", "failure_reason", "attempts", "known_solutions"],
            template=template
        )
        
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt
        )
        
        return chain
    
    def decide_action(
        self,
        objective: str,
        screen_state: str,
        previous_actions: List[str],
        attempt_number: int
    ) -> ActionDecision:
        """
        Decide next action using LangChain reasoning.
        
        Args:
            objective: Test objective
            screen_state: Current screen description
            previous_actions: List of previous action descriptions
            attempt_number: Current attempt number
            
        Returns:
            ActionDecision with reasoning
        """
        try:
            chain = self.create_action_decision_chain()
            
            result = chain.invoke({
                "objective": objective,
                "screen_state": screen_state,
                "previous_actions": "\n".join(previous_actions) if previous_actions else "None",
                "attempt_number": attempt_number
            })
            
            if isinstance(result, dict):
                return result.get('text', result)
            return result
            
        except Exception as e:
            logger.error(f"Action decision failed: {e}")
            # Fallback decision
            return ActionDecision(
                action_type="tap",
                target="unknown",
                method="ocr",
                confidence=0.3,
                reasoning=f"Fallback due to error: {e}"
            )
    
    def verify_result(
        self,
        action_taken: str,
        expected_result: str,
        observed_state: str
    ) -> Dict[str, Any]:
        """
        Verify action result using LangChain.
        
        Args:
            action_taken: Description of action
            expected_result: Expected outcome
            observed_state: Current observed state
            
        Returns:
            Verification result dictionary
        """
        try:
            chain = self.create_verification_chain()
            
            result = chain.invoke({
                "action_taken": action_taken,
                "expected_result": expected_result,
                "observed_state": observed_state
            })
            
            # Parse result
            output = result.get('text', str(result))
            
            verification = "UNCERTAIN"
            confidence = 0.5
            reasoning = output
            
            if "PASS" in output.upper():
                verification = "PASS"
                confidence = 0.8
            elif "FAIL" in output.upper():
                verification = "FAIL"
                confidence = 0.7
            
            return {
                'verification': verification,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'verification': "UNCERTAIN",
                'confidence': 0.3,
                'reasoning': f"Verification error: {e}"
            }
    
    def suggest_recovery(
        self,
        failed_action: str,
        failure_reason: str,
        attempts: int,
        known_solutions: List[str]
    ) -> Dict[str, Any]:
        """
        Suggest error recovery strategy.
        
        Args:
            failed_action: Action that failed
            failure_reason: Why it failed
            attempts: Number of attempts so far
            known_solutions: List of known solution descriptions
            
        Returns:
            Recovery suggestion dictionary
        """
        try:
            chain = self.create_error_recovery_chain()
            
            result = chain.invoke({
                "failed_action": failed_action,
                "failure_reason": failure_reason,
                "attempts": attempts,
                "known_solutions": "\n".join(known_solutions) if known_solutions else "None available"
            })
            
            output = result.get('text', str(result))
            
            return {
                'strategy': output,
                'confidence': 0.6,
                'reasoning': output
            }
            
        except Exception as e:
            logger.error(f"Recovery suggestion failed: {e}")
            return {
                'strategy': "Try OCR with case-insensitive search",
                'confidence': 0.3,
                'reasoning': f"Fallback strategy due to error: {e}"
            }


class ScreenAnalysisTool(BaseTool):
    """LangChain tool for screen analysis."""
    
    name = "analyze_screen"
    description = "Analyzes current screen state using OCR and vision. Returns description of visible elements."
    
    def _run(self, screenshot_path: str) -> str:
        """Run screen analysis."""
        # This would integrate with vision_coordinator
        return f"Screen analysis for {screenshot_path}: HVAC controls visible with AC button at top-left"
    
    async def _arun(self, screenshot_path: str) -> str:
        """Async version."""
        return self._run(screenshot_path)


class TapElementTool(BaseTool):
    """LangChain tool for tapping elements."""
    
    name = "tap_element"
    description = "Taps on an element by finding it via OCR or image matching. Input should be element name or description."
    
    def _run(self, element: str) -> str:
        """Execute tap."""
        # This would integrate with automotive_operating_system
        return f"Tapped element: {element} at coordinates (540, 300)"
    
    async def _arun(self, element: str) -> str:
        """Async version."""
        return self._run(element)


def create_testing_agent(llm, tools: List[BaseTool]) -> AgentExecutor:
    """
    Create ReAct agent for testing.
    
    The agent can reason about which tools to use and when.
    """
    from langchain.agents import AgentType, initialize_agent
    
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=10
    )
    
    return agent


def test_langchain_integration():
    """Test LangChain integration."""
    print("=" * 80)
    print("LangChain Integration Test")
    print("=" * 80)
    
    orchestrator = LangChainTestOrchestrator()
    
    # Test action decision
    print("\n1. Testing action decision chain...")
    decision = orchestrator.decide_action(
        objective="Turn on AC",
        screen_state="HVAC controls visible, AC button at top-left",
        previous_actions=[],
        attempt_number=1
    )
    print(f"Decision: {decision}")
    
    # Test verification
    print("\n2. Testing verification chain...")
    verification = orchestrator.verify_result(
        action_taken="Tapped AC button at (540, 300)",
        expected_result="AC indicator should turn ON",
        observed_state="AC indicator shows blue color, text changed to 'AC ON'"
    )
    print(f"Verification: {verification}")
    
    # Test recovery suggestion
    print("\n3. Testing error recovery...")
    recovery = orchestrator.suggest_recovery(
        failed_action="Tap AC button",
        failure_reason="Element not found via OCR",
        attempts=3,
        known_solutions=["Try image matching", "Check for case sensitivity"]
    )
    print(f"Recovery: {recovery['strategy'][:200]}...")
    
    print("\n✅ LangChain integration test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_langchain_integration()
