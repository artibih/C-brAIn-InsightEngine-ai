from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from config.global_config import CONFIG

def retry_logic(func):
    retry_config = CONFIG.get('retry_logic', {})
    max_attempts = retry_config.get('max_attempts', 5)
    multiplier = retry_config.get('multiplier', 1)
    min_wait = retry_config.get('min_wait', 1)
    max_wait = retry_config.get('max_wait', 5)
    exceptions = retry_config.get('exceptions', ["Exception"])

    @retry(
        stop=stop_after_attempt(max_attempts), 
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception)  
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
