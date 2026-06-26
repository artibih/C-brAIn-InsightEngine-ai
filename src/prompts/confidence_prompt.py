CONFIDENCE_PROMPT = """
Score confidence from 0 to 1 using this rubric:

0.0-0.2: speculative or no data  
0.2-0.4: weak evidence  
0.4-0.6: moderate mixed evidence  
0.6-0.8: strong convergent evidence  
0.8-1.0: replicated causal evidence  
Return only a float.
"""
