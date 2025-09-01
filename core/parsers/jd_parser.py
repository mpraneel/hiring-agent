import re
from typing import List, Tuple
from ..schemas.models import ParsedJD
from ..normalize.normalizer import SkillNormalizer


class JDParser:
    def __init__(self):
        """Initialize the JD parser with skill normalizer."""
        self.normalizer = SkillNormalizer()
        
        # Patterns for detecting must-have requirements
        self.must_have_patterns = [
            r'must\s+have',
            r'required',
            r'minimum',
            r'essential',
            r'necessary',
            r'mandatory',
            r'prerequisite',
            r'qualification',
            r'requirement',
            r'experience\s+with',
            r'proficient\s+in',
            r'expertise\s+in',
            r'knowledge\s+of',
            r'familiarity\s+with',
            r'understanding\s+of',
            r'background\s+in',
            r'degree\s+in',
            r'certification\s+in',
            r'years?\s+of\s+experience',
            r'experience\s+in',
        ]
        
        # Patterns for detecting nice-to-have requirements
        self.nice_to_have_patterns = [
            r'nice\s+to\s+have',
            r'preferred',
            r'bonus',
            r'plus',
            r'advantage',
            r'desired',
            r'ideal',
            r'would\s+be\s+great',
            r'helpful',
            r'beneficial',
            r'familiarity\s+with',
            r'knowledge\s+of',
            r'exposure\s+to',
            r'understanding\s+of',
            r'background\s+in',
            r'experience\s+with',
            r'proficient\s+in',
            r'expertise\s+in',
        ]
        
        # Patterns for detecting job titles
        self.title_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Developer|Engineer|Architect|Manager|Lead|Specialist|Analyst|Consultant|Coordinator|Director|Head|Officer|Representative|Assistant|Associate|Senior|Junior|Principal|Staff)',
            r'(?:Senior|Junior|Principal|Staff|Lead)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Developer|Engineer|Architect|Manager|Lead|Specialist|Analyst|Consultant|Coordinator|Director|Head|Officer|Representative|Assistant|Associate)',
            r'(?:Software|Web|Mobile|Frontend|Backend|Full\s+Stack|DevOps|Data|Machine\s+Learning|AI|Cloud|Security|QA|Test|Product|Project|Business|Systems|Network|Database|UI|UX|Design|Marketing|Sales|Customer|Support|Operations|Administrative|Executive|Management)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
    
    def parse_jd(self, jd_text: str) -> ParsedJD:
        """Parse job description text and return structured data."""
        # Clean and normalize the text
        cleaned_text = self._clean_text(jd_text)
        
        # Initialize parsed JD
        parsed_jd = ParsedJD()
        
        # Extract job title
        parsed_jd.title = self._extract_job_title(cleaned_text)
        
        # Extract must-have and nice-to-have requirements
        must_haves, nice_to_haves = self._extract_requirements(cleaned_text)
        parsed_jd.must_haves_raw = must_haves
        parsed_jd.nice_to_haves_raw = nice_to_haves
        
        # Normalize skills
        all_skills = must_haves + nice_to_haves
        parsed_jd.skills_norm = self.normalizer.normalize_skills(all_skills)
        
        return parsed_jd
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize the job description text."""
        # Convert to lowercase for easier processing
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common HTML-like tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove bullet points and list markers
        text = re.sub(r'^[\s•\-\*→▶]+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from the job description."""
        # Look for job title in the first few lines
        lines = text.split('\n')[:10]
        
        for line in lines:
            for pattern in self.title_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Basic validation
                    if len(title.split()) <= 6:  # Reasonable title length
                        return title
        
        # If no specific title found, try to extract from common patterns
        title_keywords = [
            'position', 'role', 'job', 'opportunity', 'opening',
            'we are looking for', 'seeking', 'hiring', 'recruiting'
        ]
        
        for line in lines:
            for keyword in title_keywords:
                if keyword in line:
                    # Extract text after the keyword
                    parts = line.split(keyword, 1)
                    if len(parts) > 1:
                        potential_title = parts[1].strip()
                        # Clean up the title
                        potential_title = re.sub(r'[^\w\s]', '', potential_title)
                        if len(potential_title.split()) <= 6:
                            return potential_title
        
        return None
    
    def _extract_requirements(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract must-have and nice-to-have requirements from text."""
        must_haves = []
        nice_to_haves = []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if this sentence contains must-have requirements
            if self._is_must_have_sentence(sentence):
                skills = self._extract_skills_from_sentence(sentence)
                must_haves.extend(skills)
            
            # Check if this sentence contains nice-to-have requirements
            elif self._is_nice_to_have_sentence(sentence):
                skills = self._extract_skills_from_sentence(sentence)
                nice_to_haves.extend(skills)
        
        # Also look for skills in bullet points or lists
        bullet_skills = self._extract_skills_from_bullets(text)
        must_haves.extend(bullet_skills)
        
        # Remove duplicates and clean up
        must_haves = list(set(must_haves))
        nice_to_haves = list(set(nice_to_haves))
        
        return must_haves, nice_to_haves
    
    def _is_must_have_sentence(self, sentence: str) -> bool:
        """Check if a sentence contains must-have requirements."""
        sentence_lower = sentence.lower()
        
        for pattern in self.must_have_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _is_nice_to_have_sentence(self, sentence: str) -> bool:
        """Check if a sentence contains nice-to-have requirements."""
        sentence_lower = sentence.lower()
        
        for pattern in self.nice_to_have_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _extract_skills_from_sentence(self, sentence: str) -> List[str]:
        """Extract skills from a single sentence."""
        skills = []
        
        # Get all canonical skills from the ontology
        all_canonical_skills = self.normalizer.get_all_canonical_skills()
        
        # Check for each canonical skill in the sentence
        for skill in all_canonical_skills:
            if self._skill_in_sentence(skill, sentence):
                skills.append(skill)
        
        # Also check for skill variants
        for canonical_skill, variants in self.normalizer.ontology.items():
            for variant in variants:
                if self._skill_in_sentence(variant, sentence):
                    skills.append(canonical_skill)
                    break  # Only add canonical skill once
        
        return skills
    
    def _skill_in_sentence(self, skill: str, sentence: str) -> bool:
        """Check if a skill appears in a sentence."""
        sentence_lower = sentence.lower()
        skill_lower = skill.lower()
        
        # Check for exact word boundaries
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        return bool(re.search(pattern, sentence_lower))
    
    def _extract_skills_from_bullets(self, text: str) -> List[str]:
        """Extract skills from bullet points or list items."""
        skills = []
        
        # Look for bullet points
        bullet_patterns = [
            r'^[\s•\-\*→▶]+\s*(.+)',
            r'^\d+\.\s*(.+)',
            r'^[a-z]\)\s*(.+)',
        ]
        
        lines = text.split('\n')
        for line in lines:
            for pattern in bullet_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    bullet_text = match.group(1).strip()
                    bullet_skills = self._extract_skills_from_sentence(bullet_text)
                    skills.extend(bullet_skills)
        
        return skills
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract all skills from text without categorizing them."""
        all_skills = []
        
        # Get all canonical skills from the ontology
        all_canonical_skills = self.normalizer.get_all_canonical_skills()
        
        # Check for each canonical skill in the text
        for skill in all_canonical_skills:
            if self._skill_in_text(skill, text):
                all_skills.append(skill)
        
        # Also check for skill variants
        for canonical_skill, variants in self.normalizer.ontology.items():
            for variant in variants:
                if self._skill_in_text(variant, text):
                    if canonical_skill not in all_skills:
                        all_skills.append(canonical_skill)
                    break  # Only add canonical skill once
        
        return all_skills
    
    def _skill_in_text(self, skill: str, text: str) -> bool:
        """Check if a skill appears in text."""
        text_lower = text.lower()
        skill_lower = skill.lower()
        
        # Check for exact word boundaries
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        return bool(re.search(pattern, text_lower))
