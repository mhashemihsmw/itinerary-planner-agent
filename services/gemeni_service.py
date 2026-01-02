"""
Google Gemini Service for AI Trip Planner
Implements two-step AI pipeline using Google Generative AI (Gemini) via Google ADK:
1. Extract travel facts from caption text
2. Generate structured day-by-day itinerary
"""
import json
import os
from typing import Optional, Dict, Any

from google.adk.models import Gemini

from models.schemas import Itinerary, TravelFacts, UserPreferences


class GeminiService:
    """Service for interacting with Google Gemini API using Google ADK"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini service

        Args:
            api_key: Google AI Studio API key
            model: Gemini model to use (default: gemini-2.0-flash)
        """
        if not api_key:
            raise ValueError("Google API key is required")

        # Set environment variables required by Google ADK
        os.environ.setdefault("GOOGLE_API_KEY", api_key)
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

        self.model_name = model
        self.client = Gemini(model=model)

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Safely pull text from Gemini response."""
        # Check for direct text attribute
        if hasattr(response, "text") and response.text:
            return response.text
        
        # Check for candidates list
        candidates = getattr(response, "candidates", [])
        if candidates:
            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                return content.parts[0].text
                
        raise ValueError(f"Unexpected response format from Gemini: {response}")

    async def _generate_json(self, prompt: str, temperature: float) -> Dict[str, Any]:
        """Run Gemini with JSON response contract and parse the result."""
        try:
            # Google ADK's Gemini wrapper supports generate_content_async
            response = await self.client.generate_content_async(
                contents=prompt,
                generation_config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )
            
            text_content = self._extract_text(response)
            return json.loads(text_content)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Gemini JSON response: {str(e)}")
        except Exception as e:
            raise ValueError(f"Gemini request failed: {str(e)}")

    async def extract_travel_facts(
        self,
        caption_text: str,
        preferences: Optional[UserPreferences] = None,
    ) -> TravelFacts:
        """
        Step 1: Extract factual travel data from caption text.

        Args:
            caption_text: Full caption describing the trip
            preferences: Optional user preferences to override extraction

        Returns:
            TravelFacts object with extracted information
        """
        prefs = preferences or UserPreferences()

        user_inputs = []
        if prefs.start_date and prefs.end_date:
            user_inputs.append(
                f"User trip dates: {prefs.start_date} to {prefs.end_date} (override video suggestions)"
            )
        elif prefs.flexible_month:
            flex = f"User flexible month: {prefs.flexible_month}"
            if prefs.flexible_days:
                flex += f" for {prefs.flexible_days} days"
            user_inputs.append(flex)
        if prefs.budget:
            user_inputs.append(f"Budget: {prefs.budget}")
        if prefs.pace:
            user_inputs.append(f"Pace: {prefs.pace}")
        if prefs.food_preference:
            user_inputs.append(
                f"Food preference: {prefs.food_preference} (be flexible; assume most restaurants can accommodate)"
            )
        if prefs.extra_notes:
            user_inputs.append(f"Notes: {prefs.extra_notes}")

        preference_block = "\n".join(user_inputs) if user_inputs else "None provided"

        prompt = f"""You are a travel data extraction expert. 
        Analyze the following travel video caption and extract factual information.

Caption Text:
{caption_text}

User-provided overrides (these win over the caption if they conflict):
{preference_block}

Extract and return ONLY a JSON object with these exact fields:
- destination: string (primary destination name)
- best_season: string (best time to visit, if mentioned; otherwise infer from context)
- recommended_duration_days: number (trip duration in days, if mentioned; otherwise suggest based on activities)
- attractions: array of strings (key attractions, landmarks, or places mentioned)
- food_suggestions: array of strings (local foods, dishes, or restaurants mentioned)
- travel_style: string (one of: adventure, relaxed, cultural, luxury, budget, family, romantic)

Rules:
- Extract ONLY information present or clearly implied in the text
- If specific duration isn't mentioned, estimate based on activities described
- If best season isn't mentioned, provide general guidance for that destination
- Be specific with attraction names when provided
- If the user provided dates/month, prefer those dates over any season/duration mentioned in the caption when you estimate duration
- Return valid JSON only, no explanation

Return the JSON object now:"""

        facts_json = await self._generate_json(prompt, temperature=0.3)
        return TravelFacts(**facts_json)

    async def generate_itinerary_from_facts(
        self,
        facts: TravelFacts,
        preferences: Optional[UserPreferences] = None,
    ) -> Itinerary:
        """
        Step 2: Generate day-by-day itinerary from extracted facts.

        Args:
            facts: Extracted travel facts
            preferences: Optional user preferences

        Returns:
            Complete Itinerary object
        """
        prefs = preferences or UserPreferences()

        pref_lines = []
        if prefs.start_date and prefs.end_date:
            pref_lines.append(
                f"Trip dates: {prefs.start_date} to {prefs.end_date} (must honor even if season differs)"
            )
        elif prefs.flexible_month:
            flex = f"Flexible month: {prefs.flexible_month}"
            if prefs.flexible_days:
                flex += f" for {prefs.flexible_days} days"
            pref_lines.append(flex)
        if prefs.budget:
            pref_lines.append(f"Budget level: {prefs.budget}")
        if prefs.pace:
            pref_lines.append(f"Pace: {prefs.pace}")
        if prefs.food_preference:
            pref_lines.append(
                f"Food preference: {prefs.food_preference} (be accommodating; assume restaurants offer vegan/vegetarian options)"
            )
        if prefs.extra_notes:
            pref_lines.append(f"Notes: {prefs.extra_notes}")

        preference_block = "\n".join(pref_lines) if pref_lines else "None provided"

        prompt = f"""You are an expert travel itinerary planner. Create a detailed day-by-day itinerary.

Travel Facts:
- Destination: {facts.destination}
- Duration: {facts.recommended_duration_days} days
- Best Season: {facts.best_season}
- Travel Style: {facts.travel_style}
- Key Attractions: {', '.join(facts.attractions)}
- Food Suggestions: {', '.join(facts.food_suggestions)}

User constraints (override caption suggestions where needed):
{preference_block}

Create a {facts.recommended_duration_days}-day itinerary that:
1. Distributes attractions logically across days
2. Balances activities (mix of sightseeing, food, rest)
3. Considers travel time and practicality
4. Incorporates mentioned attractions and food suggestions
5. Matches the "{facts.travel_style}" travel style
6. Respects user-provided dates/month, pace, budget, and food preferences (be flexible on food options)
7. Includes globally recognized must-see landmarks for the destination even if not in the caption, adapting choices to the user's budget
8. Treat the video as personalization, not limitation: completeness comes from the destination; however, if the trip is too short to fit all must-see landmarks, prioritize the video’s suggestions first
9. For every activity, prepend a realistic time window (e.g., "10:00-12:00: Museum…") that accounts for pace and typical dwell time
10. Estimate travel time between consecutive activities; if exact data is unavailable, use reasonable intra-city estimates (walk/transit/rideshare) and keep sequencing realistic
11. Identify activities that require booking; pick the best time of day using video cues first, then general knowledge; include a booking URL when commonly available
12. Check (or assume) opening hours/access constraints for the planned dates; if a spot is likely closed or restricted, swap in a similar open alternative and mention the fallback
13. Make all URLs clickable (use full https links) for bookings and transport tickets
14. Estimate inter-activity travel using Google Maps-style durations; prefer walking if under 30 minutes, otherwise pick transit; suggest taxi/rideshare only for luxury budget or when clearly fastest
15. Between activities, show the travel leg with mode + emoji (e.g., "🚶 12 min" or "🚇 18 min") and place it before the next activity time block
16. Add an emoji to each day title reflecting the vibe (e.g., 🌄, 🏛️, 🌊, 🍜)
17. When transit tickets are commonly bought online, include a clickable link for the likely ticket/pass
18. Avoid generic food/restaurant/cafe suggestions; prefer specific, locally relevant places 

Return ONLY a JSON object with these exact fields:
- destination: string (same as input)
- best_season: string (same as input)
- duration_days: number (same as input)
- daily_plan: array of objects, each with:
  - day: number (1, 2, 3, etc.)
  - title: string (catchy title for the day, e.g., "Temple Trail & Culture")
  - activities: array of strings (3-5 specific activities/experiences)
- key_attractions: array of strings (same as input attractions)
- food_suggestions: array of strings (same as input food suggestions)
- start_date: string | null (include if provided by user)
- end_date: string | null (include if provided by user)
- flexible_month: string | null (include if provided by user)
- flexible_days: number | null (include if provided by user)
- budget: string | null (value | balanced | luxury)
- pace: string | null (relaxed | active)
- food_preference: string | null (vegan | vegetarian | all-eater)
- extra_notes: string | null (user notes to respect)

Make the itinerary realistic, engaging, and well-paced.

Return the JSON object now:"""

        itinerary_json = await self._generate_json(prompt, temperature=0.5)
        return Itinerary(**itinerary_json)

    async def generate_itinerary(
        self,
        caption_text: str,
        video_url: Optional[str] = None,
        preferences: Optional[UserPreferences] = None,
    ) -> Itinerary:
        """
        Complete two-step pipeline: Extract facts, then generate itinerary.

        Args:
            caption_text: Full caption describing the trip
            video_url: Video URL (for reference, not used in processing)
            preferences: Optional user preferences

        Returns:
            Complete Itinerary object
        """
        prefs = preferences or UserPreferences()

        # Step 1: Extract travel facts
        facts = await self.extract_travel_facts(caption_text, prefs)

        # If user provided dates or flexible days, override duration for planning
        if prefs.start_date and prefs.end_date:
            facts.recommended_duration_days = (prefs.end_date - prefs.start_date).days + 1
        elif prefs.flexible_days:
            facts.recommended_duration_days = prefs.flexible_days

        # Step 2: Generate itinerary from facts
        itinerary = await self.generate_itinerary_from_facts(facts, prefs)

        return itinerary
