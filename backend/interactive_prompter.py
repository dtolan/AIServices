from typing import List, Dict, Optional
from backend.llm_service import LLMService
from backend.knowledge_base import KnowledgeBase
from backend.models.schemas import SDPrompt
import json


class InteractivePrompter:
    """
    Handles interactive prompt creation with clarifying questions
    """

    def __init__(self, llm_service: LLMService, knowledge_base: KnowledgeBase):
        self.llm = llm_service
        self.kb = knowledge_base

    def _build_question_system_prompt(self) -> str:
        """Build system prompt for generating clarifying questions"""
        kb_context = self.kb.get_all_context()

        return f"""You are an expert prompt engineer for Stable Diffusion. Your job is to ask clarifying questions to create better prompts.

When a user gives a vague or incomplete description, ask 2-3 specific questions to gather more detail.

Focus on:
1. **Style**: What art style? (anime, photorealistic, painting, etc.)
2. **Composition**: What should be the focus? Camera angle? Framing?
3. **Mood/Atmosphere**: What feeling should it convey?
4. **Details**: Specific elements they want included
5. **Lighting**: What kind of lighting?

{kb_context if kb_context else ""}

Ask natural, conversational questions. Keep questions concise.

Respond with JSON in this format:
{{
  "needs_clarification": true/false,
  "questions": ["question 1", "question 2"],
  "reasoning": "why you're asking these questions"
}}

If the description is already detailed enough, set needs_clarification to false."""

    def _build_prompt_from_qa_system_prompt(self) -> str:
        """Build system prompt for creating prompt from Q&A"""
        kb_context = self.kb.get_all_context()

        return f"""You are an expert Stable Diffusion prompt engineer. Create an optimized prompt based on the conversation.

The user has provided answers to clarifying questions. Use all this information to create the best possible Stable Diffusion prompt.

{kb_context if kb_context else ""}

Key principles:
1. Be specific and descriptive
2. Include quality tags: "masterpiece, best quality, highly detailed"
3. Include appropriate style tags
4. Add lighting and composition details
5. Use negative prompts to avoid unwanted elements
6. Choose optimal parameters based on the style

Respond ONLY with valid JSON in this exact format:
{{
  "positive_prompt": "detailed prompt here",
  "negative_prompt": "negative tags here",
  "steps": 30,
  "cfg_scale": 7.5,
  "width": 512,
  "height": 512,
  "sampler_name": "DPM++ 2M Karras",
  "explanation": "brief explanation of choices",
  "suggestions": ["tip 1", "tip 2"]
}}"""

    async def analyze_and_ask_questions(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Analyze user input and ask clarifying questions if needed

        Returns:
            Dict with needs_clarification, questions, reasoning
        """
        # Build context from knowledge base
        kb_context = self.kb.get_context_for_prompt(user_input)

        user_prompt = f"""User wants to generate: "{user_input}"

Should I ask clarifying questions to improve the prompt, or is this detailed enough?"""

        if conversation_history:
            user_prompt = "Previous conversation:\n"
            for msg in conversation_history[-3:]:
                user_prompt += f"{msg['role']}: {msg['content']}\n"
            user_prompt += f"\nUser now says: \"{user_input}\"\n\nShould I ask clarifying questions?"

        try:
            # Use planning LLM for this (better reasoning)
            response = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=self._build_question_system_prompt(),
                temperature=0.7,
                use_planning_llm=True
            )

            # Parse JSON response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            result = json.loads(response)
            return result

        except json.JSONDecodeError as e:
            print(f"Failed to parse question response: {e}")
            # Fallback - assume no questions needed
            return {
                "needs_clarification": False,
                "questions": [],
                "reasoning": "Unable to analyze input"
            }

    async def create_prompt_from_conversation(
        self,
        conversation_history: List[Dict]
    ) -> Dict:
        """
        Create final prompt from full conversation including Q&A

        Args:
            conversation_history: Full conversation with questions and answers

        Returns:
            Dict with enhanced_prompt and explanation
        """
        # Build conversation summary
        conversation_text = "Full conversation:\n\n"
        for msg in conversation_history:
            conversation_text += f"{msg['role']}: {msg['content']}\n"

        conversation_text += "\nBased on this conversation, create the optimal Stable Diffusion prompt."

        # Get relevant knowledge base context
        # Extract user's original intent from conversation
        user_messages = [msg['content'] for msg in conversation_history if msg['role'] == 'user']
        combined_intent = ' '.join(user_messages)
        kb_context = self.kb.get_context_for_prompt(combined_intent)

        try:
            # Use planning LLM for final prompt creation
            response = await self.llm.generate(
                prompt=conversation_text,
                system_prompt=self._build_prompt_from_qa_system_prompt(),
                temperature=0.7,
                use_planning_llm=True
            )

            # Parse JSON response
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            return data

        except json.JSONDecodeError as e:
            print(f"Failed to parse prompt creation response: {e}")
            print(f"Response was: {response}")

            # Fallback
            return {
                "positive_prompt": combined_intent + ", masterpiece, best quality, highly detailed",
                "negative_prompt": "blurry, low quality, distorted, ugly",
                "steps": 30,
                "cfg_scale": 7.5,
                "width": 512,
                "height": 512,
                "sampler_name": "DPM++ 2M Karras",
                "explanation": "Created basic prompt (parsing failed)",
                "suggestions": []
            }

    async def refine_with_feedback(
        self,
        current_prompt: SDPrompt,
        user_feedback: str,
        kb_hints: Optional[str] = None
    ) -> Dict:
        """
        Refine prompt based on user feedback and knowledge base hints

        Args:
            current_prompt: Current prompt
            user_feedback: User's feedback
            kb_hints: Optional knowledge base hints to consider

        Returns:
            Refined prompt data
        """
        # Search knowledge base for relevant info
        if not kb_hints:
            kb_hints = self.kb.get_context_for_prompt(user_feedback)

        kb_section = f"Relevant knowledge base hints:\n{kb_hints}" if kb_hints else ""

        refinement_prompt = f"""Current prompt:
Positive: {current_prompt.positive_prompt}
Negative: {current_prompt.negative_prompt}

User feedback: "{user_feedback}"

{kb_section}

Refine the prompt based on this feedback."""

        try:
            # Use execution LLM for faster refinement
            response = await self.llm.generate(
                prompt=refinement_prompt,
                system_prompt=self._build_prompt_from_qa_system_prompt(),
                temperature=0.7,
                use_planning_llm=False  # Use execution LLM for speed
            )

            # Parse JSON
            response = response.strip()
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            data = json.loads(response)
            return data

        except json.JSONDecodeError as e:
            print(f"Failed to parse refinement response: {e}")

            # Simple fallback
            return {
                "positive_prompt": f"{current_prompt.positive_prompt}, {user_feedback}",
                "negative_prompt": current_prompt.negative_prompt,
                "steps": current_prompt.steps,
                "cfg_scale": current_prompt.cfg_scale,
                "width": current_prompt.width,
                "height": current_prompt.height,
                "sampler_name": current_prompt.sampler_name,
                "explanation": "Applied basic refinement",
                "suggestions": []
            }
