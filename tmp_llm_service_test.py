from app.config import get_settings
from app.services.llm_service import LLMService
import logging
settings = get_settings()
settings.llm_provider = 'ollama'
settings.llm_mode = 'auto'
llm = LLMService(settings=settings, logger=logging.getLogger('test'))
print('provider', llm.provider)
print('client set', llm.client is not None)
print('check_ollama_available', llm.check_ollama_available())
print('generate_general_answer:')
answer, used = llm.generate_general_answer('¿Qué es comercio ambulatorio?')
print('used', used)
print(answer)
