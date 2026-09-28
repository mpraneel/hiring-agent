#!/usr/bin/env python3
"""
Basic test script for the Hiring Agent application.
Run this to verify that the core components are working correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from core.schemas.models import ParsedResume, ParsedJD, MatchResult
        print("✓ Data models imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data models: {e}")
        return False
    
    try:
        from core.normalize.normalizer import SkillNormalizer
        print("✓ Skill normalizer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import skill normalizer: {e}")
        return False
    
    try:
        from core.parsers.resume_parser import ResumeParser
        print("✓ Resume parser imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import resume parser: {e}")
        return False
    
    try:
        from core.parsers.jd_parser import JDParser
        print("✓ JD parser imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import JD parser: {e}")
        return False
    
    try:
        from core.scoring.baseline_scorer import BaselineScorer
        print("✓ Baseline scorer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import baseline scorer: {e}")
        return False
    
    try:
        from core.scoring.aggregate import MatchAggregator
        print("✓ Match aggregator imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import match aggregator: {e}")
        return False
    
    return True

def test_skill_normalization():
    """Test skill normalization functionality."""
    print("\nTesting skill normalization...")
    
    try:
        from core.normalize.normalizer import SkillNormalizer
        
        normalizer = SkillNormalizer()
        
        # Test basic normalization
        test_skills = ["javascript", "JS", "Python", "PY", "React.js", "react"]
        normalized = normalizer.normalize_skills(test_skills)
        
        print(f"Original skills: {test_skills}")
        print(f"Normalized skills: {normalized}")
        
        # Check that we get some normalized results
        if len(normalized) > 0:
            print("✓ Skill normalization working")
            return True
        else:
            print("✗ No skills were normalized")
            return False
            
    except Exception as e:
        print(f"✗ Skill normalization failed: {e}")
        return False

def test_jd_parsing():
    """Test job description parsing."""
    print("\nTesting JD parsing...")
    
    try:
        from core.parsers.jd_parser import JDParser
        
        jd_parser = JDParser()
        
        # Sample job description
        sample_jd = """
        Senior Software Engineer
        
        We are looking for a Senior Software Engineer with the following requirements:
        
        Must have:
        - 5+ years of experience with Python
        - Strong knowledge of React and JavaScript
        - Experience with Docker and AWS
        
        Nice to have:
        - Kubernetes experience
        - Machine learning background
        - Agile methodology experience
        """
        
        parsed_jd = jd_parser.parse_jd(sample_jd)
        
        print(f"Job title: {parsed_jd.title}")
        print(f"Must-have skills: {parsed_jd.must_have_skills}")
        print(f"Nice-to-have skills: {parsed_jd.nice_to_have_skills}")
        print(f"Normalized skills: {parsed_jd.skills_norm}")
        
        if parsed_jd.skills_norm:
            print("✓ JD parsing working")
            return True
        else:
            print("✗ No skills extracted from JD")
            return False
            
    except Exception as e:
        print(f"✗ JD parsing failed: {e}")
        return False

def test_scoring():
    """Test the scoring functionality."""
    print("\nTesting scoring...")
    
    try:
        from core.schemas.models import ParsedResume, ParsedJD
        from core.scoring.baseline_scorer import BaselineScorer
        
        # Create sample data
        resume = ParsedResume(
            name="John Doe",
            skills_norm=["python", "react", "docker", "git"]
        )
        
        jd = ParsedJD(
            title="Senior Software Engineer",
            must_have_skills=["python", "javascript"],
            nice_to_have_skills=["docker", "aws", "kubernetes"]
        )
        
        scorer = BaselineScorer()
        score, matched, missing, nice = scorer.calculate_score(resume, jd)
        
        print(f"Match score: {score:.2f} ({score*100:.1f}%)")
        print(f"Matched skills: {matched}")
        print(f"Missing skills: {missing}")
        print(f"Nice matches: {nice}")
        
        if 0 <= score <= 1:
            print("✓ Scoring working")
            return True
        else:
            print("✗ Invalid score range")
            return False
            
    except Exception as e:
        print(f"✗ Scoring failed: {e}")
        return False

def test_aggregator():
    """Test the match aggregator."""
    print("\nTesting match aggregator...")
    
    try:
        from core.schemas.models import ParsedResume, ParsedJD
        from core.scoring.aggregate import MatchAggregator
        
        # Create sample data
        resume = ParsedResume(
            name="Jane Smith",
            skills_norm=["python", "react", "docker", "git", "aws"]
        )
        
        jd = ParsedJD(
            title="Full Stack Developer",
            must_have_skills=["python", "javascript", "react"],
            nice_to_have_skills=["docker", "aws", "kubernetes", "git"]
        )
        
        # Test without LLM (to avoid API key requirement)
        aggregator = MatchAggregator(enable_llm=False)
        result = aggregator.match_resume_to_jd(resume, jd)
        
        print(f"Match score: {result.match_score:.2f}")
        print(f"Matched skills: {result.matched_skills}")
        print(f"Missing skills: {result.missing_skills}")
        print(f"Nice matches: {result.nice_matches}")
        
        if result.match_score > 0:
            print("✓ Match aggregator working")
            return True
        else:
            print("✗ No match score calculated")
            return False
            
    except Exception as e:
        print(f"✗ Match aggregator failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Hiring Agent - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_skill_normalization,
        test_jd_parsing,
        test_scoring,
        test_aggregator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
