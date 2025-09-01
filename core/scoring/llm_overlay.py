import os
import json
from typing import Optional, List
from ..schemas.models import ParsedResume, ParsedJD, MatchResult


class LLMOverlay:
    def __init__(self, provider: str = "openai"):
        """Initialize the LLM overlay with specified provider."""
        self.provider = provider.lower()
        self.api_key = os.getenv("LLM_API_KEY")
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is required")
        
        # Initialize the appropriate client
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def _init_gemini(self):
        """Initialize Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
    
    def generate_rationale(self, resume: ParsedResume, jd: ParsedJD, match_result: MatchResult) -> str:
        """Generate rationale explaining the match score and gaps."""
        prompt = self._build_rationale_prompt(resume, jd, match_result)
        
        try:
            if self.provider == "openai":
                return self._call_openai(prompt)
            elif self.provider == "gemini":
                return self._call_gemini(prompt)
        except Exception as e:
            return f"Unable to generate rationale: {str(e)}"
    
    def generate_suggestions(self, resume: ParsedResume, jd: ParsedJD, match_result: MatchResult) -> List[str]:
        """Generate suggestions for improving the resume."""
        prompt = self._build_suggestions_prompt(resume, jd, match_result)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "gemini":
                response = self._call_gemini(prompt)
            
            # Parse suggestions from response
            return self._parse_suggestions(response)
        except Exception as e:
            return [f"Unable to generate suggestions: {str(e)}"]
    
    def _build_rationale_prompt(self, resume: ParsedResume, jd: ParsedJD, match_result: MatchResult) -> str:
        """Build prompt for generating rationale."""
        return f"""
You are an expert recruiter analyzing a candidate's fit for a job position. Based on the following information, provide a concise and professional rationale explaining the match score and highlighting key strengths and gaps.

JOB DESCRIPTION:
Title: {jd.title or 'Not specified'}
Must-have skills: {', '.join(jd.must_haves_raw) if jd.must_haves_raw else 'None specified'}
Nice-to-have skills: {', '.join(jd.nice_to_haves_raw) if jd.nice_to_haves_raw else 'None specified'}

CANDIDATE PROFILE:
Name: {resume.name or 'Not specified'}
Skills: {', '.join(resume.skills_norm) if resume.skills_norm else 'None detected'}

MATCH RESULTS:
Score: {match_result.baseline_score:.2f} ({match_result.baseline_score * 100:.1f}%)
Matched skills: {', '.join(match_result.matched_skills) if match_result.matched_skills else 'None'}
Missing skills: {', '.join(match_result.missing_skills) if match_result.missing_skills else 'None'}

Please provide a 2-3 sentence rationale that:
1. Explains the overall match score
2. Highlights the candidate's key strengths
3. Identifies the most critical gaps
4. Uses professional, objective language

Focus on the most important points and keep it concise.
"""
    
    def _build_suggestions_prompt(self, resume: ParsedResume, jd: ParsedJD, match_result: MatchResult) -> str:
        """Build prompt for generating suggestions."""
        return f"""
You are an expert resume writer helping a candidate improve their resume for a specific job application. Based on the following information, provide 2-3 specific, actionable suggestions.

JOB DESCRIPTION:
Title: {jd.title or 'Not specified'}
Must-have skills: {', '.join(jd.must_haves_raw) if jd.must_haves_raw else 'None specified'}
Nice-to-have skills: {', '.join(jd.nice_to_haves_raw) if jd.nice_to_haves_raw else 'None specified'}

CANDIDATE PROFILE:
Skills: {', '.join(resume.skills_norm) if resume.skills_norm else 'None detected'}

MISSING SKILLS:
{', '.join(match_result.missing_skills) if match_result.missing_skills else 'None'}

Please provide 2-3 specific suggestions that:
1. Focus on the most critical missing skills
2. Are actionable and specific (e.g., "Add a bullet point about Docker deployment experience")
3. Include learning resources or quick wins where appropriate
4. Are realistic and achievable

Format each suggestion as a separate bullet point starting with "• ".
Keep suggestions concise and practical.
"""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides professional, objective analysis and suggestions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse suggestions from LLM response."""
        suggestions = []
        
        # Split by bullet points or numbered items
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '→', '▶', '1.', '2.', '3.')):
                # Remove the bullet point marker
                suggestion = line.lstrip('•-*→▶').strip()
                suggestion = suggestion.lstrip('0123456789.').strip()
                if suggestion:
                    suggestions.append(suggestion)
            elif line and len(suggestions) > 0:
                # This might be a continuation of the previous suggestion
                suggestions[-1] += " " + line
        
        # If no bullet points found, split by sentences
        if not suggestions and response:
            sentences = response.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:  # Minimum length for a meaningful suggestion
                    suggestions.append(sentence + '.')
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        try:
            if self.provider == "openai":
                # Test with a simple prompt
                test_response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                )
                return True
            elif self.provider == "gemini":
                # Test with a simple prompt
                test_response = self.model.generate_content("Hello")
                return True
        except Exception:
            return False
        
        return False
