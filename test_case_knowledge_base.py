"""
test_case_knowledge_base.py - Test Case Knowledge Base System

This module loads, indexes, and manages test cases from Excel files.
The AI agent learns from these test cases to:
1. Understand test patterns
2. Generate similar tests
3. Execute tests correctly
4. Verify results accurately

The more test cases added, the smarter the AI becomes!
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCaseKnowledgeBase:
    """
    Manages test case knowledge for the AI agent.
    
    Features:
    - Load test cases from multiple Excel files
    - Index by component, action, feature
    - Provide examples to AI for learning
    - Support pattern recognition
    - Continuous learning (add new cases)
    """
    
    def __init__(self, knowledge_base_dir: str = "test_case_knowledge"):
        """
        Initialize knowledge base.
        
        Args:
            knowledge_base_dir: Directory containing test case Excel files
        """
        self.kb_dir = Path(knowledge_base_dir)
        self.test_cases = []
        self.index = {
            'by_component': defaultdict(list),
            'by_action': defaultdict(list),
            'by_feature': defaultdict(list),
            'by_id': {}
        }
        
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Knowledge base initialized: {self.kb_dir}")
    
    def load_test_cases_from_excel(self, excel_path: str) -> int:
        """
        Load test cases from an Excel file.
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            Number of test cases loaded
        """
        try:
            excel_path = Path(excel_path)
            logger.info(f"Loading test cases from: {excel_path.name}")
            
            # Read Excel file
            xls = pd.ExcelFile(excel_path)
            
            # Try common sheet names
            sheet_name = None
            for name in ['Sheet1', xls.sheet_names[0]]:
                if name in xls.sheet_names:
                    sheet_name = name
                    break
            
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Parse test cases
            count = self._parse_test_cases_from_dataframe(df, excel_path.stem)
            
            logger.info(f"✅ Loaded {count} test cases from {excel_path.name}")
            return count
            
        except Exception as e:
            logger.error(f"Failed to load {excel_path}: {e}")
            return 0
    
    def _parse_test_cases_from_dataframe(
        self, 
        df: pd.DataFrame, 
        source: str
    ) -> int:
        """
        Parse test cases from DataFrame.
        
        Handles the multi-step test case format where:
        - First row has ID and Title
        - Subsequent rows are steps
        """
        count = 0
        current_test_case = None
        
        for _, row in df.iterrows():
            # Check if this is a new test case (has ID)
            if pd.notna(row.get('ID')):
                # Save previous test case
                if current_test_case and current_test_case['steps']:
                    self._add_test_case(current_test_case)
                    count += 1
                
                # Start new test case
                current_test_case = {
                    'id': str(row['ID']).strip(),
                    'title': str(row['Title']).strip() if pd.notna(row.get('Title')) else '',
                    'source': source,
                    'steps': [],
                    'component': self._extract_component(row),
                    'feature': self._extract_feature(row),
                }
            
            # Add step to current test case
            if current_test_case and pd.notna(row.get('Description')):
                step = {
                    'step_number': int(row['Step']) if pd.notna(row.get('Step')) else len(current_test_case['steps']) + 1,
                    'description': str(row['Description']).strip(),
                    'expected_result': str(row['Expected Result']).strip() if pd.notna(row.get('Expected Result')) else '',
                }
                current_test_case['steps'].append(step)
        
        # Add last test case
        if current_test_case and current_test_case['steps']:
            self._add_test_case(current_test_case)
            count += 1
        
        return count
    
    def _extract_component(self, row: pd.Series) -> str:
        """Extract component from test case (HVAC, Media, etc)."""
        title = str(row.get('Title', '')).lower()
        
        if 'hvac' in title or 'climate' in title:
            return 'HVAC'
        elif 'media' in title or 'audio' in title or 'radio' in title:
            return 'Media'
        elif 'navigation' in title or 'map' in title:
            return 'Navigation'
        elif 'phone' in title or 'call' in title:
            return 'Phone'
        elif 'settings' in title:
            return 'Settings'
        else:
            return 'General'
    
    def _extract_feature(self, row: pd.Series) -> str:
        """Extract specific feature from test case."""
        title = str(row.get('Title', ''))
        
        # Try to extract feature from title (e.g., "HVAC: Fan Speed" -> "Fan Speed")
        if ':' in title:
            return title.split(':', 1)[1].strip()
        
        return title
    
    def _add_test_case(self, test_case: Dict):
        """Add test case to knowledge base and indices."""
        self.test_cases.append(test_case)
        
        # Index by component
        self.index['by_component'][test_case['component']].append(test_case)
        
        # Index by feature
        self.index['by_feature'][test_case['feature']].append(test_case)
        
        # Index by ID
        self.index['by_id'][test_case['id']] = test_case
        
        # Index by actions (extract from steps)
        for step in test_case['steps']:
            actions = self._extract_actions(step['description'])
            for action in actions:
                self.index['by_action'][action].append(test_case)
    
    def _extract_actions(self, description: str) -> List[str]:
        """Extract action keywords from step description."""
        actions = []
        desc_lower = description.lower()
        
        # Common actions
        action_keywords = [
            'click', 'tap', 'press', 'swipe', 'scroll',
            'open', 'close', 'launch', 'navigate',
            'enter', 'type', 'input',
            'verify', 'check', 'observe',
            'select', 'choose',
            'increase', 'decrease', 'adjust',
            'play', 'pause', 'stop',
            'minimize', 'maximize'
        ]
        
        for keyword in action_keywords:
            if keyword in desc_lower:
                actions.append(keyword)
        
        return actions
    
    def load_all_test_cases(self) -> int:
        """
        Load all Excel files from knowledge base directory.
        
        Returns:
            Total number of test cases loaded
        """
        total = 0
        
        for excel_file in self.kb_dir.glob("*.xlsx"):
            count = self.load_test_cases_from_excel(str(excel_file))
            total += count
        
        logger.info(f"📚 Total test cases in knowledge base: {total}")
        return total
    
    def get_examples_for_component(
        self, 
        component: str, 
        limit: int = 5
    ) -> List[Dict]:
        """
        Get example test cases for a component.
        
        Args:
            component: Component name (HVAC, Media, etc)
            limit: Maximum number of examples
            
        Returns:
            List of test case examples
        """
        examples = self.index['by_component'].get(component, [])
        return examples[:limit]
    
    def get_examples_for_action(
        self, 
        action: str, 
        limit: int = 3
    ) -> List[Dict]:
        """
        Get example test cases for an action.
        
        Args:
            action: Action keyword (click, verify, etc)
            limit: Maximum number of examples
            
        Returns:
            List of test case examples
        """
        examples = self.index['by_action'].get(action.lower(), [])
        return examples[:limit]
    
    def get_test_case_by_id(self, test_id: str) -> Optional[Dict]:
        """Get a specific test case by ID."""
        return self.index['by_id'].get(test_id)
    
    def generate_knowledge_summary(self) -> str:
        """
        Generate a summary of knowledge base for AI context.
        
        Returns:
            Formatted summary string
        """
        summary = f"""
TEST CASE KNOWLEDGE BASE SUMMARY
================================

Total Test Cases: {len(self.test_cases)}

Components Covered:
"""
        
        for component, cases in self.index['by_component'].items():
            summary += f"  - {component}: {len(cases)} test cases\n"
        
        summary += "\nCommon Actions:\n"
        top_actions = sorted(
            self.index['by_action'].items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:10]
        
        for action, cases in top_actions:
            summary += f"  - {action}: {len(cases)} occurrences\n"
        
        return summary
    
    def get_ai_context_for_component(
        self, 
        component: str, 
        max_examples: int = 3
    ) -> str:
        """
        Generate AI context with examples for a specific component.
        
        Args:
            component: Component name
            max_examples: Maximum examples to include
            
        Returns:
            Formatted context string for AI prompts
        """
        examples = self.get_examples_for_component(component, max_examples)
        
        if not examples:
            return f"No test case examples found for {component}"
        
        context = f"\n{'='*70}\n"
        context += f"TEST CASE EXAMPLES FOR {component.upper()}\n"
        context += f"{'='*70}\n\n"
        context += "Learn from these examples to understand how to test this component:\n\n"
        
        for i, test_case in enumerate(examples, 1):
            context += f"EXAMPLE {i}: {test_case['title']} (ID: {test_case['id']})\n"
            context += "-" * 70 + "\n"
            
            for step in test_case['steps']:
                context += f"Step {step['step_number']}: {step['description']}\n"
                context += f"Expected: {step['expected_result']}\n\n"
            
            context += "\n"
        
        return context
    
    def save_knowledge_base_summary(self, output_path: str):
        """Save knowledge base summary to JSON file."""
        summary = {
            'total_test_cases': len(self.test_cases),
            'components': {
                comp: len(cases) 
                for comp, cases in self.index['by_component'].items()
            },
            'features': {
                feat: len(cases) 
                for feat, cases in self.index['by_feature'].items()
            },
            'test_case_ids': list(self.index['by_id'].keys())
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Knowledge base summary saved: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            'total_test_cases': len(self.test_cases),
            'components': dict(
                (comp, len(cases)) 
                for comp, cases in self.index['by_component'].items()
            ),
            'total_steps': sum(
                len(tc['steps']) for tc in self.test_cases
            ),
            'actions_covered': len(self.index['by_action'])
        }


def main():
    """Test the knowledge base system."""
    print("Testing Test Case Knowledge Base System")
    print("=" * 80)
    
    # Create knowledge base
    kb = TestCaseKnowledgeBase("test_case_knowledge")
    
    # Load test cases
    test_files = [
        "/mnt/user-data/uploads/1763476774309_Paccar_App_component_Automation_test_cases.xlsx",
        "/mnt/user-data/uploads/1763476774315_Paccar_AOSP_component_Automation_test_cases.xlsx",
    ]
    
    for file in test_files:
        if Path(file).exists():
            kb.load_test_cases_from_excel(file)
    
    # Show statistics
    print("\n" + kb.generate_knowledge_summary())
    
    # Show examples
    print("\n" + "=" * 80)
    print("EXAMPLE: Getting HVAC test cases")
    print("=" * 80)
    
    hvac_examples = kb.get_examples_for_component("HVAC", limit=2)
    for example in hvac_examples:
        print(f"\n{example['title']} (ID: {example['id']})")
        for step in example['steps'][:2]:
            print(f"  Step {step['step_number']}: {step['description'][:60]}...")
    
    print("\n✅ Knowledge base system working!")


if __name__ == "__main__":
    main()