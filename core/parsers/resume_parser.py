import re
import pypdf
from typing import List, Optional
from pathlib import Path

from ..schemas.models import ParsedResume, CandidateExperience
from ..normalize.normalizer import SkillNormalizer


class ResumeParser:
    def __init__(self):
        """Initialize the resume parser with skill normalizer."""
        self.normalizer = SkillNormalizer()
        
        # Common section headers
        self.section_headers = [
            r'skills?',
            r'experience',
            r'work\s+history',
            r'employment',
            r'education',
            r'academic',
            r'projects?',
            r'achievements?',
            r'certifications?',
            r'languages?',
            r'interests?',
            r'contact',
            r'personal',
            r'profile',
            r'summary',
            r'objective'
        ]
        
        # Patterns for extracting information
        self.name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-|]',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*resume',
        ]
        
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        
        # Date patterns
        self.date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}',
            r'(\d{4})\s*[-–—]\s*(present|current|now|\d{4})',
        ]
    
    def parse_pdf(self, pdf_path: str) -> ParsedResume:
        """Parse a PDF resume and return structured data."""
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            
            # Parse the extracted text
            return self._parse_text(text)
            
        except Exception as e:
            raise Exception(f"Failed to parse PDF: {str(e)}")
    
    def parse_text(self, text: str) -> ParsedResume:
        """Parse resume text and return structured data."""
        return self._parse_text(text)
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        
        except Exception as e:
            raise Exception(f"Failed to read PDF file: {str(e)}")
        
        return text
    
    def _parse_text(self, text: str) -> ParsedResume:
        """Parse resume text into structured data."""
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        # Initialize parsed resume
        parsed_resume = ParsedResume()
        
        # Extract basic information
        parsed_resume.name = self._extract_name(cleaned_lines)
        parsed_resume.email = self._extract_email(text)
        parsed_resume.phone = self._extract_phone(text)
        
        # Extract skills
        skills_section = self._find_section(cleaned_lines, ['skills', 'technical skills', 'technologies'])
        if skills_section:
            parsed_resume.skills_raw = self._extract_skills(skills_section)
            parsed_resume.skills_norm = self.normalizer.normalize_skills(parsed_resume.skills_raw)
        
        # Extract education
        education_section = self._find_section(cleaned_lines, ['education', 'academic'])
        if education_section:
            parsed_resume.education = self._extract_education(education_section)
        
        # Extract experiences
        experience_section = self._find_section(cleaned_lines, ['experience', 'work history', 'employment'])
        if experience_section:
            parsed_resume.experiences = self._extract_experiences(experience_section)
        
        return parsed_resume
    
    def _extract_name(self, lines: List[str]) -> Optional[str]:
        """Extract candidate name from resume."""
        for line in lines[:10]:  # Check first 10 lines
            for pattern in self.name_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    name = match.group(1).strip()
                    # Basic validation - name should be reasonable length
                    if 2 <= len(name.split()) <= 4:
                        return name
        return None
    
    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text."""
        match = re.search(self.email_pattern, text)
        return match.group(0) if match else None
    
    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text."""
        match = re.search(self.phone_pattern, text)
        if match:
            # Reconstruct phone number
            parts = match.groups()
            phone = ''.join(part for part in parts if part)
            return phone
        return None
    
    def _find_section(self, lines: List[str], section_names: List[str]) -> List[str]:
        """Find and extract a specific section from the resume."""
        section_lines = []
        in_section = False
        
        for line in lines:
            # Check if this line starts a new section
            if self._is_section_header(line, section_names):
                if in_section:
                    break  # Found next section, stop here
                in_section = True
                continue
            
            # Check if we've hit another major section
            if in_section and self._is_major_section_header(line):
                break
            
            if in_section:
                section_lines.append(line)
        
        return section_lines
    
    def _is_section_header(self, line: str, target_sections: List[str]) -> bool:
        """Check if a line is a section header."""
        line_lower = line.lower().strip()
        
        # Check for exact matches
        for section in target_sections:
            if line_lower == section.lower():
                return True
        
        # Check for patterns
        for pattern in self.section_headers:
            if re.search(pattern, line_lower):
                return True
        
        return False
    
    def _is_major_section_header(self, line: str) -> bool:
        """Check if a line is a major section header (not a subsection)."""
        line_lower = line.lower().strip()
        
        major_sections = [
            'experience', 'work history', 'employment',
            'education', 'academic',
            'skills', 'technical skills',
            'projects', 'achievements',
            'contact', 'personal'
        ]
        
        for section in major_sections:
            if line_lower == section or line_lower.startswith(section):
                return True
        
        return False
    
    def _extract_skills(self, skills_lines: List[str]) -> List[str]:
        """Extract skills from skills section."""
        skills = []
        
        for line in skills_lines:
            # Split by common delimiters
            line_skills = re.split(r'[,;•\|\-\n]', line)
            
            for skill in line_skills:
                skill = skill.strip()
                if skill and len(skill) > 1:  # Filter out empty or single character skills
                    # Remove common prefixes/suffixes
                    skill = re.sub(r'^(proficient\s+in|experience\s+with|knowledge\s+of|expertise\s+in)\s*', '', skill, flags=re.IGNORECASE)
                    skill = re.sub(r'\s*(years?|yrs?|experience|proficient|expert|intermediate|beginner|advanced)$', '', skill, flags=re.IGNORECASE)
                    
                    if skill and len(skill) > 1:
                        skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def _extract_education(self, education_lines: List[str]) -> List[str]:
        """Extract education information."""
        education = []
        current_education = []
        
        for line in education_lines:
            # Look for degree patterns
            if re.search(r'(bachelor|master|phd|associate|diploma|certificate)', line, re.IGNORECASE):
                if current_education:
                    education.append(' '.join(current_education))
                    current_education = []
                current_education = [line]
            else:
                current_education.append(line)
        
        if current_education:
            education.append(' '.join(current_education))
        
        return education
    
    def _extract_experiences(self, experience_lines: List[str]) -> List[CandidateExperience]:
        """Extract work experiences from experience section."""
        experiences = []
        current_experience = None
        current_bullets = []
        
        for line in experience_lines:
            # Look for job title patterns
            title_match = self._extract_job_title(line)
            if title_match:
                # Save previous experience if exists
                if current_experience:
                    current_experience.bullets = current_bullets
                    experiences.append(current_experience)
                
                # Start new experience
                current_experience = CandidateExperience(title=title_match)
                current_bullets = []
            else:
                # Check if this is a bullet point
                if line.strip().startswith(('•', '-', '*', '→', '▶')):
                    bullet = line.strip()[1:].strip()
                    if bullet:
                        current_bullets.append(bullet)
                elif current_experience and line.strip():
                    # Might be additional context for current experience
                    current_bullets.append(line.strip())
        
        # Add the last experience
        if current_experience:
            current_experience.bullets = current_bullets
            experiences.append(current_experience)
        
        return experiences
    
    def _extract_job_title(self, line: str) -> Optional[str]:
        """Extract job title from a line."""
        # Common patterns for job titles
        patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:at|@|for)\s+([A-Z][a-zA-Z\s&]+)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-|]\s*([A-Z][a-zA-Z\s&]+)',
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*[-|]\s*([A-Z][a-zA-Z\s&]+)\s*[-|]\s*(\d{4})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                title = match.group(1).strip()
                # Basic validation
                if len(title.split()) <= 5:  # Reasonable title length
                    return title
        
        return None
