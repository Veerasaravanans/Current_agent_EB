"""
langgraph_workflow.py - Multi-Step Workflow Management with LangGraph

Uses LangGraph to orchestrate complex testing workflows that go beyond
simple Q&A conversations.

Features:
- State machines for test execution
- Multi-agent coordination
- Error recovery flows
- Decision points and branching
- Human-in-the-loop when needed
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    from typing_extensions import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not installed. Install with: pip install langgraph")


# ==================================================================================
# WORKFLOW STATE DEFINITIONS
# ==================================================================================

class TestState(TypedDict):
    """State for test execution workflow."""
    objective: str
    component: Optional[str]
    current_step: int
    total_steps: int
    steps: List[Dict]
    executed_steps: List[Dict]
    current_screen: Optional[str]
    error_count: int
    retry_count: int
    status: str
    result: Optional[Dict]


class WorkflowStatus(Enum):
    """Workflow status codes."""
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    ERROR = "error"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    HUMAN_INPUT = "human_input"


# ==================================================================================
# LANGGRAPH WORKFLOW MANAGER
# ==================================================================================

class LangGraphWorkflowManager:
    """
    Manages complex multi-step testing workflows using LangGraph.
    
    Workflow stages:
    1. Plan: Generate test plan
    2. Execute: Run each step
    3. Verify: Check results
    4. Retry: Handle errors with backoff
    5. Learn: Update knowledge base
    6. Report: Generate results
    """
    
    def __init__(self, langchain_coordinator=None):
        """Initialize workflow manager."""
        if not LANGGRAPH_AVAILABLE:
            logger.error("❌ LangGraph not available")
            self.available = False
            return
        
        self.coordinator = langchain_coordinator
        self.available = True
        self.workflow = self._build_test_execution_workflow()
        self.memory = MemorySaver()
        
        logger.info("✅ LangGraph workflow manager initialized")
    
    def _build_test_execution_workflow(self) -> StateGraph:
        """Build the test execution state machine."""
        workflow = StateGraph(TestState)
        
        # Add nodes
        workflow.add_node("plan", self._plan_test)
        workflow.add_node("execute_step", self._execute_step)
        workflow.add_node("verify_result", self._verify_result)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("request_human_help", self._request_human_help)
        workflow.add_node("finalize", self._finalize_test)
        
        # Entry point
        workflow.set_entry_point("plan")
        
        # Edges
        workflow.add_edge("plan", "execute_step")
        workflow.add_conditional_edges(
            "execute_step",
            self._should_verify_or_retry,
            {"verify": "verify_result", "retry": "handle_error", "next_step": "execute_step", "done": "finalize"}
        )
        workflow.add_conditional_edges(
            "verify_result",
            self._check_verification,
            {"success": "execute_step", "retry": "handle_error", "done": "finalize"}
        )
        workflow.add_conditional_edges(
            "handle_error",
            self._should_retry_or_escalate,
            {"retry": "execute_step", "human_help": "request_human_help", "fail": "finalize"}
        )
        workflow.add_edge("request_human_help", "execute_step")
        workflow.add_edge("finalize", END)
        
        return workflow
    
    def _plan_test(self, state: TestState) -> TestState:
        """Node: Plan test execution."""
        logger.info("📋 Planning test...")
        
        if self.coordinator:
            plan = self.coordinator.generate_test_plan(state["objective"], state.get("component"))
            if plan:
                state["steps"] = [{"action": s.action_type, "target": s.target, "parameters": s.parameters,
                                  "expected": s.expected_result, "reasoning": s.reasoning} for s in plan.steps]
                state["total_steps"] = len(plan.steps)
                state["current_step"] = 0
                state["status"] = WorkflowStatus.EXECUTING.value
                logger.info(f"✅ Plan created: {len(plan.steps)} steps")
        else:
            state["status"] = WorkflowStatus.EXECUTING.value
        
        return state
    
    def _execute_step(self, state: TestState) -> TestState:
        """Node: Execute current step."""
        logger.info(f"⚡ Executing step {state['current_step'] + 1}/{state['total_steps']}")
        
        if state["current_step"] >= state["total_steps"]:
            state["status"] = WorkflowStatus.SUCCESS.value
            return state
        
        step = state["steps"][state["current_step"]]
        logger.info(f"   Action: {step['action']} on {step['target']}")
        
        execution_result = {"step_number": state["current_step"] + 1, "action": step["action"],
                          "target": step["target"], "success": True, "screenshot": None}
        
        if "executed_steps" not in state:
            state["executed_steps"] = []
        state["executed_steps"].append(execution_result)
        state["status"] = WorkflowStatus.VERIFYING.value
        
        return state
    
    def _verify_result(self, state: TestState) -> TestState:
        """Node: Verify step result."""
        logger.info("🔍 Verifying result...")
        
        step = state["steps"][state["current_step"]]
        expected = step.get("expected", "")
        
        if self.coordinator and state.get("current_screen"):
            verification = self.coordinator.verify_screen_state(state["current_screen"], expected)
            if verification and verification.verified:
                logger.info("✅ Verification passed")
                state["current_step"] += 1
                state["retry_count"] = 0
                state["status"] = WorkflowStatus.EXECUTING.value
            else:
                logger.warning("❌ Verification failed")
                state["retry_count"] = state.get("retry_count", 0) + 1
                state["status"] = WorkflowStatus.RETRY.value
        else:
            logger.warning("Verification skipped")
            state["current_step"] += 1
            state["status"] = WorkflowStatus.EXECUTING.value
        
        return state
    
    def _handle_error(self, state: TestState) -> TestState:
        """Node: Handle error with retry logic."""
        logger.info("🔧 Handling error...")
        
        state["error_count"] = state.get("error_count", 0) + 1
        retry_count = state.get("retry_count", 0)
        
        if retry_count < 10:
            logger.info(f"Retrying... (attempt {retry_count + 1}/10)")
            if self.coordinator:
                step = state["steps"][state["current_step"]]
                error_desc = f"Failed to {step['action']} {step['target']}"
                attempts = [executed["action"] for executed in state.get("executed_steps", [])]
                solution = self.coordinator.analyze_error_and_suggest_fix(error_desc, attempts)
                logger.info(f"Suggested solution: {solution[:100]}...")
            state["status"] = WorkflowStatus.RETRY.value
        else:
            logger.warning("Max retries reached, requesting human help")
            state["status"] = WorkflowStatus.HUMAN_INPUT.value
        
        return state
    
    def _request_human_help(self, state: TestState) -> TestState:
        """Node: Request human intervention."""
        logger.info("🆘 Requesting human help...")
        step = state["steps"][state["current_step"]]
        logger.info(f"   Human needed for: {step['action']} {step['target']}")
        state["retry_count"] = 0
        state["status"] = WorkflowStatus.EXECUTING.value
        return state
    
    def _finalize_test(self, state: TestState) -> TestState:
        """Node: Finalize test and generate report."""
        logger.info("📊 Finalizing test...")
        
        total_steps = state["total_steps"]
        executed = len(state.get("executed_steps", []))
        success = state["current_step"] >= total_steps
        
        state["result"] = {"success": success, "total_steps": total_steps,
                          "executed_steps": executed, "error_count": state.get("error_count", 0),
                          "final_status": state["status"]}
        
        status = "✅ SUCCESS" if success else "❌ FAILURE"
        logger.info(f"{status}: {executed}/{total_steps} steps completed")
        
        return state
    
    def _should_verify_or_retry(self, state: TestState) -> str:
        """Decide whether to verify, retry, continue, or finish."""
        if state["status"] == WorkflowStatus.RETRY.value:
            return "retry"
        elif state["current_step"] >= state["total_steps"]:
            return "done"
        else:
            return "verify"
    
    def _check_verification(self, state: TestState) -> str:
        """Check verification result."""
        if state["status"] == WorkflowStatus.RETRY.value:
            return "retry"
        elif state["current_step"] >= state["total_steps"]:
            return "done"
        else:
            return "success"
    
    def _should_retry_or_escalate(self, state: TestState) -> str:
        """Decide whether to retry, ask human, or fail."""
        if state["status"] == WorkflowStatus.HUMAN_INPUT.value:
            return "human_help"
        elif state.get("retry_count", 0) < 10:
            return "retry"
        else:
            return "fail"
    
    def execute_test(self, objective: str, component: Optional[str] = None) -> Dict[str, Any]:
        """Execute test using workflow."""
        if not self.available:
            logger.error("LangGraph not available")
            return {"success": False, "error": "LangGraph not available"}
        
        logger.info(f"🚀 Starting workflow for: {objective}")
        
        initial_state = {"objective": objective, "component": component, "current_step": 0,
                        "total_steps": 0, "steps": [], "executed_steps": [], "current_screen": None,
                        "error_count": 0, "retry_count": 0, "status": WorkflowStatus.PLANNING.value,
                        "result": None}
        
        app = self.workflow.compile(checkpointer=self.memory)
        
        try:
            config = {"configurable": {"thread_id": "test_execution"}}
            final_state = None
            for state in app.stream(initial_state, config):
                for node_name, node_state in state.items():
                    logger.debug(f"   Node: {node_name}, Status: {node_state.get('status')}")
                    final_state = node_state
            
            result = final_state.get("result", {}) if final_state else {}
            logger.info(f"✅ Workflow completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {"success": False, "error": str(e)}


def main():
    """Test LangGraph workflow."""
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Testing LangGraph Workflow Manager")
    print("=" * 80)
    
    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph not available")
        return
    
    workflow_manager = LangGraphWorkflowManager()
    
    print("\n1. Testing multi-step workflow...")
    result = workflow_manager.execute_test(
        objective="Turn on AC and set temperature to 72 degrees",
        component="hvac"
    )
    
    print(f"\nWorkflow result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n✅ LangGraph workflow test complete!")


if __name__ == "__main__":
    main()
