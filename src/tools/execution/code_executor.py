from langchain_core.tools import tool
from typing import Dict, Any
import subprocess
import tempfile
import json
from pathlib import Path
from config.settings import settings


@tool
def execute_python_code(code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.
    
    Args:
        code: Python code to execute
        context: Optional context variables to inject
        
    Returns:
        Dictionary with stdout, stderr, and return value
    """
    
    # Prepare code with context
    if context:
        context_code = "# Injected context\n"
        for key, value in context.items():
            context_code += f"{key} = {repr(value)}\n"
        full_code = context_code + "\n" + code
    else:
        full_code = code
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        temp_file = f.name
    
    try:
        # Execute with timeout
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=settings.code_execution_timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out",
            "return_code": -1
        }
    
    finally:
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)


@tool
def execute_statistical_analysis(
    test_type: str,
    data: Dict[str, list],
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Execute statistical analysis on provided data.
    
    Args:
        test_type: Type of test (t_test, anova, regression, correlation)
        data: Dictionary of data arrays
        parameters: Additional test parameters
        
    Returns:
        Dictionary with test results
    """
    
    import numpy as np
    from scipy import stats
    
    parameters = parameters or {}
    
    if test_type == "t_test":
        group1 = np.array(data.get("group1", []))
        group2 = np.array(data.get("group2", []))
        statistic, pvalue = stats.ttest_ind(group1, group2)
        
        return {
            "test": "Independent t-test",
            "statistic": float(statistic),
            "p_value": float(pvalue),
            "significant": bool(pvalue < parameters.get("alpha", 0.05)),
            "effect_size": float((np.mean(group1) - np.mean(group2)) / 
                                np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2))
        }
    
    elif test_type == "correlation":
        x = np.array(data.get("x", []))
        y = np.array(data.get("y", []))
        r, pvalue = stats.pearsonr(x, y)
        
        return {
            "test": "Pearson correlation",
            "correlation": float(r),
            "p_value": float(pvalue),
            "significant": bool(pvalue < parameters.get("alpha", 0.05))
        }
    
    elif test_type == "anova":
        groups = [np.array(data[key]) for key in data.keys()]
        statistic, pvalue = stats.f_oneway(*groups)
        
        return {
            "test": "One-way ANOVA",
            "f_statistic": float(statistic),
            "p_value": float(pvalue),
            "significant": bool(pvalue < parameters.get("alpha", 0.05))
        }
    
    else:
        return {
            "error": f"Unsupported test type: {test_type}"
        }
