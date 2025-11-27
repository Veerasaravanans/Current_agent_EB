"""
excel_report_generator.py - Professional Excel Test Report Generator

Generates detailed test execution reports in Excel format matching
the industry standard format from PACCAR test cases.

Output Format Matches:
- Test Case ID and Title
- Step-by-step execution details
- Expected vs Actual results
- Step verdicts (Pass/Fail)
- Overall test verdict
- Defect tracking
- Screenshots/Evidence
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

try:
    import xlsxwriter
    XLSXWRITER_AVAILABLE = True
except:
    XLSXWRITER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExcelReportGenerator:
    """
    Generates professional Excel test reports.
    
    Features:
    - Industry-standard format
    - Step-by-step details
    - Pass/Fail verdicts
    - Defect tracking
    - Screenshots links
    - Summary statistics
    """
    
    def __init__(self, output_dir: str = "test_reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory for saving reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Report generator initialized: {self.output_dir}")
    
    def generate_test_report(
        self,
        test_execution: Dict,
        report_name: Optional[str] = None
    ) -> str:
        """
        Generate Excel report for test execution.
        
        Args:
            test_execution: Test execution dictionary with results
            report_name: Optional report filename
            
        Returns:
            Path to generated report
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_id = test_execution.get('test_id', 'TEST')
            report_name = f"{test_id}_Report_{timestamp}.xlsx"
        
        report_path = self.output_dir / report_name
        
        # Create DataFrame
        df = self._create_report_dataframe(test_execution)
        
        # Write to Excel with formatting
        if XLSXWRITER_AVAILABLE:
            self._write_formatted_excel(df, report_path, test_execution)
        else:
            df.to_excel(report_path, index=False)
        
        logger.info(f"✅ Report generated: {report_path}")
        return str(report_path)
    
    def _create_report_dataframe(self, test_execution: Dict) -> pd.DataFrame:
        """
        Create DataFrame from test execution data.
        
        Format matches PACCAR test case structure:
        - ID, Title, #, Step, Description, Expected Result, Actual Result,
          Step Verdict, Test Case Verdict, Related Jira Defect, etc.
        """
        rows = []
        
        test_id = test_execution.get('test_id', '')
        title = test_execution.get('title', '')
        overall_verdict = test_execution.get('overall_verdict', '')
        steps = test_execution.get('steps', [])
        
        # First row with test case header
        if steps:
            first_step = steps[0]
            rows.append({
                'ID': test_id,
                'Title': title,
                '#': 1,
                'Step': 1,
                'Description': first_step.get('description', ''),
                'Expected Result': first_step.get('expected_result', ''),
                'Actual Result': first_step.get('actual_result', ''),
                'Step Verdict': first_step.get('verdict', ''),
                'Test Case Verdict': overall_verdict,
                'Related Jira Defect': first_step.get('defect_id', ''),
                'Non EB Failure': first_step.get('non_eb_failure', 'No'),
                'Test Comment': test_execution.get('comment', ''),
                'Type': 'Test Case',
                '_polarion': f"PACCAR_NAID/{test_id}"
            })
            
            # Subsequent steps
            for i, step in enumerate(steps[1:], start=2):
                rows.append({
                    'ID': '',  # Empty for subsequent steps
                    'Title': '',
                    '#': i,
                    'Step': i,
                    'Description': step.get('description', ''),
                    'Expected Result': step.get('expected_result', ''),
                    'Actual Result': step.get('actual_result', ''),
                    'Step Verdict': step.get('verdict', ''),
                    'Test Case Verdict': '',
                    'Related Jira Defect': step.get('defect_id', ''),
                    'Non EB Failure': step.get('non_eb_failure', ''),
                    'Test Comment': '',
                    'Type': '',
                    '_polarion': f"PACCAR_NAID/{test_id}"
                })
        
        return pd.DataFrame(rows)
    
    def _write_formatted_excel(
        self,
        df: pd.DataFrame,
        filepath: Path,
        test_execution: Dict
    ):
        """Write Excel with professional formatting."""
        workbook = xlsxwriter.Workbook(str(filepath))
        worksheet = workbook.add_worksheet('Test Results')
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4472C4',
            'font_color': 'white',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        pass_format = workbook.add_format({
            'bg_color': '#C6EFCE',
            'font_color': '#006100',
            'border': 1
        })
        
        fail_format = workbook.add_format({
            'bg_color': '#FFC7CE',
            'font_color': '#9C0006',
            'border': 1
        })
        
        cell_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'top',
            'text_wrap': True
        })
        
        # Write headers
        for col_num, column in enumerate(df.columns):
            worksheet.write(0, col_num, column, header_format)
        
        # Write data with formatting
        for row_num, row_data in enumerate(df.values, start=1):
            for col_num, cell_value in enumerate(row_data):
                column_name = df.columns[col_num]
                
                # Apply conditional formatting for verdicts
                if column_name in ['Step Verdict', 'Test Case Verdict']:
                    if str(cell_value).upper() == 'PASS':
                        worksheet.write(row_num, col_num, cell_value, pass_format)
                    elif str(cell_value).upper() == 'FAIL':
                        worksheet.write(row_num, col_num, cell_value, fail_format)
                    else:
                        worksheet.write(row_num, col_num, cell_value, cell_format)
                else:
                    worksheet.write(row_num, col_num, cell_value, cell_format)
        
        # Set column widths
        worksheet.set_column('A:A', 15)  # ID
        worksheet.set_column('B:B', 30)  # Title
        worksheet.set_column('C:D', 8)   # #, Step
        worksheet.set_column('E:E', 40)  # Description
        worksheet.set_column('F:F', 40)  # Expected Result
        worksheet.set_column('G:G', 40)  # Actual Result
        worksheet.set_column('H:I', 15)  # Verdicts
        worksheet.set_column('J:M', 20)  # Other columns
        
        workbook.close()
    
    def generate_summary_report(
        self,
        test_executions: List[Dict],
        report_name: str = "Test_Summary"
    ) -> str:
        """
        Generate summary report for multiple test executions.
        
        Args:
            test_executions: List of test execution dictionaries
            report_name: Summary report filename
            
        Returns:
            Path to generated summary report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{report_name}_{timestamp}.xlsx"
        
        # Calculate statistics
        total_tests = len(test_executions)
        passed_tests = sum(
            1 for te in test_executions 
            if te.get('overall_verdict', '').upper() == 'PASS'
        )
        failed_tests = total_tests - passed_tests
        
        total_steps = sum(
            len(te.get('steps', [])) for te in test_executions
        )
        passed_steps = sum(
            sum(1 for step in te.get('steps', []) 
                if step.get('verdict', '').upper() == 'PASS')
            for te in test_executions
        )
        failed_steps = total_steps - passed_steps
        
        # Create summary data
        summary_data = {
            'Metric': [
                'Total Test Cases',
                'Passed Test Cases',
                'Failed Test Cases',
                'Pass Rate (%)',
                '',
                'Total Steps',
                'Passed Steps',
                'Failed Steps',
                'Step Pass Rate (%)'
            ],
            'Value': [
                total_tests,
                passed_tests,
                failed_tests,
                round(passed_tests / total_tests * 100, 2) if total_tests > 0 else 0,
                '',
                total_steps,
                passed_steps,
                failed_steps,
                round(passed_steps / total_steps * 100, 2) if total_steps > 0 else 0
            ]
        }
        
        # Create detailed results
        results_data = []
        for te in test_executions:
            results_data.append({
                'Test ID': te.get('test_id', ''),
                'Title': te.get('title', ''),
                'Component': te.get('component', ''),
                'Total Steps': len(te.get('steps', [])),
                'Verdict': te.get('overall_verdict', ''),
                'Execution Time': te.get('execution_time', ''),
                'Defects': len(te.get('defects', []))
            })
        
        # Write to Excel
        if XLSXWRITER_AVAILABLE:
            workbook = xlsxwriter.Workbook(str(report_path))
            
            # Summary sheet
            summary_ws = workbook.add_worksheet('Summary')
            summary_df = pd.DataFrame(summary_data)
            
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#4472C4',
                'font_color': 'white', 'border': 1
            })
            
            for col_num, col_name in enumerate(summary_df.columns):
                summary_ws.write(0, col_num, col_name, header_format)
                for row_num, value in enumerate(summary_df[col_name], start=1):
                    summary_ws.write(row_num, col_num, value)
            
            # Results sheet
            results_ws = workbook.add_worksheet('Detailed Results')
            results_df = pd.DataFrame(results_data)
            
            for col_num, col_name in enumerate(results_df.columns):
                results_ws.write(0, col_num, col_name, header_format)
                for row_num, value in enumerate(results_df[col_name], start=1):
                    results_ws.write(row_num, col_num, value)
            
            workbook.close()
        else:
            with pd.ExcelWriter(report_path) as writer:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                pd.DataFrame(results_data).to_excel(writer, sheet_name='Detailed Results', index=False)
        
        logger.info(f"✅ Summary report generated: {report_path}")
        return str(report_path)


def main():
    """Test report generator."""
    print("Testing Excel Report Generator")
    print("=" * 80)
    
    # Create sample test execution
    test_execution = {
        'test_id': 'NAID-24430',
        'title': 'HVAC: Fan Speed Test',
        'component': 'HVAC',
        'overall_verdict': 'PASS',
        'execution_time': '45 seconds',
        'comment': 'Test executed successfully',
        'defects': [],
        'steps': [
            {
                'step_number': 1,
                'description': 'Go to System UI HVAC section',
                'expected_result': 'Fan speed widget should be shown',
                'actual_result': 'Fan speed widget displayed correctly',
                'verdict': 'PASS',
                'defect_id': '',
                'non_eb_failure': 'No'
            },
            {
                'step_number': 2,
                'description': 'Change fan speed value',
                'expected_result': 'Widget should display new fan speed',
                'actual_result': 'Fan speed updated to level 5',
                'verdict': 'PASS',
                'defect_id': '',
                'non_eb_failure': 'No'
            }
        ]
    }
    
    # Generate report
    generator = ExcelReportGenerator("test_reports")
    report_path = generator.generate_test_report(test_execution)
    
    print(f"\n✅ Test report generated: {report_path}")
    print("\nReport includes:")
    print("  - Test case ID and title")
    print("  - Step-by-step execution details")
    print("  - Expected vs Actual results")
    print("  - Pass/Fail verdicts")
    print("  - Professional formatting")


if __name__ == "__main__":
    main()