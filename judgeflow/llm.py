"""GPT-4 adapter with retry functionality."""

import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
import openai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAdapter:
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """Initialize the LLM adapter.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY env var
            model: Model to use, defaults to GPT-4 Turbo
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat request to the OpenAI API with automatic retries.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens in response (optional)
        
        Returns:
            The response text from the model
            
        Raises:
            openai.APIError: If all retries fail
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

async def test_chat():
    """Simple test function."""
    llm = LLMAdapter()
    messages = [{"role": "user", "content": "Say hello!"}]
    response = await llm.chat(messages)
    print(f"Response: {response}")

def main():
    """Run the test chat with proper event loop handling."""
    if os.name == 'nt':  # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(test_chat())
    finally:
        loop.close()

if __name__ == "__main__":
    main() 