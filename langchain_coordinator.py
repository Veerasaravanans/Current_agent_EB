"""
langchain_coordinator.py - LangChain Integration for Heavy Lifting

Uses LangChain for:
- Chain-of-Thought reasoning
- Multi-step task decomposition
- Structured output parsing
- Memory management
- Error correction loops

This handles the complex AI reasoning that was previously manual.
"""

import logging
from typing import List, Dict, Optional, Any
import json

from rag_prompt_manager import RAGPromptManager

logger = logging.getLogger(__name__)

try:
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
    
    # Try importing ConversationBufferMemory (backward compatible)
    MEMORY_AVAILABLE = False
    try:
        # For LangChain >= 0.2.0
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_core.chat_history import BaseChatMessageHistory
        MEMORY_AVAILABLE = True
        MEMORY_TYPE = "new"
    except ImportError:
        try:
            # For LangChain < 0.2.0
            from langchain.memory import ConversationBufferMemory
            MEMORY_AVAILABLE = True
            MEMORY_TYPE = "old"
        except ImportError:
            # Memory not available, will use simple list fallback
            MEMORY_AVAILABLE = False
            MEMORY_TYPE = "fallback"
            logger.warning("Memory module not available, using simple list fallback")
    
    # ==================================================================================
    # STRUCTURED OUTPUT MODELS (Pydantic) - Only defined if imports succeed
    # ==================================================================================
    
    class ActionStep(BaseModel):
        """Single action step."""
        action_type: str = Field(description="Type of action: tap, swipe, input_text, verify, etc.")
        target: str = Field(description="Target element: text or icon name")
        parameters: Dict[str, Any] = Field(default={}, description="Additional parameters")
        expected_result: str = Field(description="Expected result after action")
        reasoning: str = Field(description="Why this action is needed")


    class TestPlan(BaseModel):
        """Test execution plan with chain-of-thought."""
        objective: str = Field(description="Overall test objective")
        thought_process: str = Field(description="Step-by-step reasoning")
        steps: List[ActionStep] = Field(description="List of actions to execute")
        component: str = Field(description="Component being tested: hvac, media, navigation")
        estimated_difficulty: str = Field(description="easy, medium, hard")


    class ActionDecision(BaseModel):
        """Decision for next action."""
        action: str = Field(description="Action type")
        target: str = Field(description="Target element")
        method: str = Field(description="Detection method: ocr, image, vision_ai")
        confidence: float = Field(description="Confidence 0.0-1.0")
        reasoning: str = Field(description="Chain-of-thought explanation")
        fallback: Optional[str] = Field(default=None, description="Fallback method if this fails")


    class VerificationResult(BaseModel):
        """Screen verification result."""
        verified: bool = Field(description="Whether verification passed")
        observed_state: str = Field(description="What was actually observed")
        expected_state: str = Field(description="What was expected")
        confidence: float = Field(description="Confidence 0.0-1.0")
        reasoning: str = Field(description="Explanation of decision")
    
    LANGCHAIN_AVAILABLE = True
    
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    logger.warning(f"LangChain not installed. Install with: pip install langchain-ollama langgraph (Error: {e})")


# ==================================================================================
# LANGCHAIN COORDINATOR
# ==================================================================================

class LangChainCoordinator:
    """
    Coordinates LangChain operations for the AI agent.
    
    Features:
    - Chain-of-Thought: Explicit reasoning steps
    - Few-Shot Learning: Examples guide behavior
    - Structured Outputs: Parsed JSON responses
    - Memory: Tracks conversation context
    - Error Recovery: Self-correcting loops
    """
    
    def __init__(
        self,
        model_name: str = "llava:7b",
        temperature: float = 0.3,
        enable_memory: bool = True
    ):
        """
        Initialize LangChain coordinator.
        
        Args:
            model_name: Ollama model name
            temperature: Model temperature (0.0-1.0)
            enable_memory: Enable conversation memory
        """
        if not LANGCHAIN_AVAILABLE:
            logger.error("❌ LangChain not available")
            self.available = False
            return
        
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize RAG prompt manager
        self.rag_manager = RAGPromptManager()
        
        # Initialize Ollama LLM via LangChain
        logger.info(f"Initializing LangChain with model: {model_name}...")
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
        )
        
        # Initialize memory with backward compatibility
        self.memory = None
        if enable_memory:
            if MEMORY_AVAILABLE:
                if MEMORY_TYPE == "old":
                    # Use old ConversationBufferMemory
                    self.memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    logger.info("Using ConversationBufferMemory (LangChain < 0.2)")
                elif MEMORY_TYPE == "new":
                    # Use new ChatMessageHistory
                    self.memory = ChatMessageHistory()
                    logger.info("Using ChatMessageHistory (LangChain >= 0.2)")
            else:
                # Fallback: simple list-based memory
                self.memory = []
                logger.info("Using simple list-based memory (fallback)")
        
        # Output parsers
        self.action_parser = PydanticOutputParser(pydantic_object=ActionDecision)
        self.plan_parser = PydanticOutputParser(pydantic_object=TestPlan)
        self.verification_parser = PydanticOutputParser(pydantic_object=VerificationResult)
        
        self.available = True
        logger.info("✅ LangChain coordinator initialized")
    
    def generate_test_plan(
        self,
        objective: str,
        component: Optional[str] = None
    ) -> Optional['TestPlan']:
        """
        Generate test execution plan with Chain-of-Thought.
        
        Args:
            objective: Test objective
            component: Optional component hint
            
        Returns:
            TestPlan with reasoning and steps
        """
        if not self.available:
            logger.error("LangChain not available")
            return None
        
        try:
            logger.info(f"🧠 Generating test plan with Chain-of-Thought: {objective}")
            
            # Get relevant context from RAG
            context = self.rag_manager.get_system_prompt_for_task(
                task_description=objective,
                component=component,
                max_context_size=2000
            )
            
            # Chain-of-Thought prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert automotive testing AI agent. 

Your task: Create a detailed test execution plan using Chain-of-Thought reasoning.

Relevant Knowledge:
{context}

{format_instructions}

Think step-by-step:
1. Understand the objective
2. Identify the component and UI elements involved
3. Plan the sequence of actions
4. Consider potential issues
5. Output structured plan"""),
                ("human", "Objective: {objective}")
            ])
            
            # Create chain
            chain = prompt_template | self.llm
            
            # Execute
            response = chain.invoke({
                "objective": objective,
                "context": context,
                "format_instructions": self.plan_parser.get_format_instructions()
            })
            
            # Parse response
            content = response.content
            
            # Clean JSON from markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            plan = self.plan_parser.parse(content)
            
            logger.info(f"✅ Generated plan with {len(plan.steps)} steps")
            logger.debug(f"Thought process: {plan.thought_process[:200]}...")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to generate test plan: {e}")
            return None
    
    def decide_next_action(
        self,
        current_screen_description: str,
        objective: str,
        previous_attempts: List[str] = None
    ) -> Optional['ActionDecision']:
        """
        Decide next action with Chain-of-Thought reasoning.
        
        Args:
            current_screen_description: What's on screen (from OCR/vision)
            objective: Current sub-objective
            previous_attempts: List of previous failed attempts
            
        Returns:
            ActionDecision with reasoning
        """
        if not self.available:
            return None
        
        try:
            logger.info(f"🧠 Deciding next action with CoT reasoning")
            
            # Get context
            context = self.rag_manager.get_context_for_action(
                action_type="decide",
                target=objective
            )
            
            # Build prompt with few-shot examples
            few_shot_examples = [
                {
                    "screen": "Main dashboard with AC, Temperature, Fan controls visible",
                    "objective": "Turn on AC",
                    "decision": {
                        "action": "tap",
                        "target": "AC",
                        "method": "ocr",
                        "confidence": 0.95,
                        "reasoning": "AC text is visible on screen. OCR is fastest and most accurate for text detection. Will search for 'AC' text and tap center coordinates.",
                        "fallback": "image_match with ac_button.png if OCR fails"
                    }
                },
                {
                    "screen": "Media player showing current station",
                    "objective": "Switch to FM radio (PACCAR system)",
                    "decision": {
                        "action": "double_tap",
                        "target": "Media",
                        "method": "ocr",
                        "confidence": 0.90,
                        "reasoning": "PACCAR systems require double-tap on Media icon to open source selection. Single tap only opens player. Must use double_tap() with 50ms delay.",
                        "fallback": "single tap and look for Source button"
                    }
                }
            ]
            
            # Create few-shot prompt
            example_prompt = ChatPromptTemplate.from_messages([
                ("human", "Screen: {screen}\nObjective: {objective}"),
                ("assistant", "{decision}")
            ])
            
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=few_shot_examples,
                input_variables=["screen", "objective", "previous_attempts"]
            )
            
            # Full prompt
            full_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AI agent deciding the best action.

Use Chain-of-Thought reasoning:
1. Analyze current screen state
2. Consider previous failed attempts
3. Choose best detection method (OCR > Image Match > Vision AI)
4. Explain your reasoning
5. Provide fallback strategy

Context:
{context}

{format_instructions}"""),
                few_shot_prompt,
                ("human", "Screen: {screen}\nObjective: {objective}\nPrevious attempts: {previous_attempts}\n\nDecide the BEST next action:")
            ])
            
            # Create chain
            chain = full_prompt | self.llm
            
            # Execute
            response = chain.invoke({
                "context": context[:1000],  # Limit context
                "screen": current_screen_description,
                "objective": objective,
                "previous_attempts": str(previous_attempts) if previous_attempts else "None",
                "format_instructions": self.action_parser.get_format_instructions()
            })
            
            # Parse
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            decision = self.action_parser.parse(content)
            
            logger.info(f"✅ Decision: {decision.action} on {decision.target} via {decision.method}")
            logger.debug(f"Reasoning: {decision.reasoning[:150]}...")
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to decide action: {e}")
            return None
    
    def verify_screen_state(
        self,
        screen_description: str,
        expected_state: str
    ) -> Optional['VerificationResult']:
        """
        Verify screen state with reasoning.
        
        Args:
            screen_description: Current screen (from OCR/vision)
            expected_state: Expected state
            
        Returns:
            VerificationResult with reasoning
        """
        if not self.available:
            return None
        
        try:
            logger.info(f"🔍 Verifying: {expected_state}")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are verifying screen state.

Analyze carefully and explain your reasoning.

{format_instructions}"""),
                ("human", """Current screen: {screen}

Expected state: {expected}

Does the current screen match the expected state? Explain your reasoning.""")
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "screen": screen_description,
                "expected": expected_state,
                "format_instructions": self.verification_parser.get_format_instructions()
            })
            
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            result = self.verification_parser.parse(content)
            
            logger.info(f"✅ Verification: {result.verified} (confidence: {result.confidence})")
            
            return result
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return None
    
    def analyze_error_and_suggest_fix(
        self,
        error_description: str,
        attempts_made: List[str]
    ) -> str:
        """
        Analyze error and suggest fix using RAG + CoT.
        
        Args:
            error_description: Description of error
            attempts_made: What was already tried
            
        Returns:
            Suggested solution
        """
        if not self.available:
            return ""
        
        try:
            logger.info(f"🔧 Analyzing error: {error_description[:50]}...")
            
            # Get relevant solutions from RAG
            solutions = self.rag_manager.get_error_solution(error_description)
            
            # CoT prompt for error analysis
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert troubleshooter.

Analyze the error and suggest a solution using:
1. Known solutions from knowledge base
2. Chain-of-Thought reasoning
3. Previous attempts (avoid repeating)

Known Solutions:
{solutions}"""),
                ("human", """Error: {error}

Attempts made:
{attempts}

Think step-by-step and suggest the BEST next approach:""")
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "error": error_description,
                "attempts": "\n".join(f"- {a}" for a in attempts_made),
                "solutions": solutions
            })
            
            suggestion = response.content
            
            logger.info(f"✅ Suggested fix: {suggestion[:100]}...")
            
            return suggestion
            
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return ""
    
    def get_memory_context(self) -> str:
        """Get conversation memory context."""
        if not self.memory:
            return ""
        
        try:
            if isinstance(self.memory, list):
                # Fallback list-based memory
                return "\n".join(str(m) for m in self.memory[-10:])  # Last 10 messages
            elif MEMORY_TYPE == "old":
                # Old ConversationBufferMemory
                return str(self.memory.load_memory_variables({}))
            elif MEMORY_TYPE == "new":
                # New ChatMessageHistory
                messages = self.memory.messages
                return "\n".join([f"{m.type}: {m.content}" for m in messages[-10:]])
            else:
                return ""
        except:
            return ""
    
    def clear_memory(self):
        """Clear conversation memory."""
        if self.memory:
            if isinstance(self.memory, list):
                self.memory.clear()
            elif MEMORY_TYPE == "old":
                self.memory.clear()
            elif MEMORY_TYPE == "new":
                self.memory.clear()
            logger.info("Memory cleared")


def main():
    """Test LangChain coordinator."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Testing LangChain Coordinator")
    print("=" * 80)
    
    # Initialize
    coordinator = LangChainCoordinator()
    
    if not coordinator.available:
        print("❌ LangChain not available")
        return
    
    # Test 1: Generate test plan
    print("\n1. Testing test plan generation (Chain-of-Thought)...")
    plan = coordinator.generate_test_plan(
        objective="Turn on AC and set temperature to 72 degrees",
        component="hvac"
    )
    
    if plan:
        print(f"   Objective: {plan.objective}")
        print(f"   Component: {plan.component}")
        print(f"   Difficulty: {plan.estimated_difficulty}")
        print(f"   Steps: {len(plan.steps)}")
        print(f"\n   Thought Process:\n   {plan.thought_process[:200]}...")
        print(f"\n   First step: {plan.steps[0].action_type} {plan.steps[0].target}")
    
    # Test 2: Action decision
    print("\n2. Testing action decision (Few-Shot + CoT)...")
    decision = coordinator.decide_next_action(
        current_screen_description="Dashboard showing AC button, temperature controls, fan controls",
        objective="Turn on AC",
        previous_attempts=[]
    )
    
    if decision:
        print(f"   Action: {decision.action}")
        print(f"   Target: {decision.target}")
        print(f"   Method: {decision.method}")
        print(f"   Confidence: {decision.confidence}")
        print(f"\n   Reasoning: {decision.reasoning[:150]}...")
    
    # Test 3: Error analysis
    print("\n3. Testing error analysis...")
    solution = coordinator.analyze_error_and_suggest_fix(
        error_description="Temperature slider not responding to swipe gestures",
        attempts_made=["Horizontal swipe 300ms", "Vertical swipe", "Faster swipe"]
    )
    
    print(f"   Solution: {solution[:200]}...")
    
    print("\n✅ LangChain coordinator test complete!")


if __name__ == "__main__":
    main()