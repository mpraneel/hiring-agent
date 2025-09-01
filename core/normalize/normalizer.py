import json
import re
from typing import List, Dict, Set
from pathlib import Path


class SkillNormalizer:
    def __init__(self, ontology_path: str = None):
        """Initialize the skill normalizer with the ontology."""
        if ontology_path is None:
            ontology_path = Path(__file__).parent / "skill_ontology.json"
        
        with open(ontology_path, 'r') as f:
            self.ontology = json.load(f)
        
        # Create reverse mapping for faster lookup
        self.variant_to_canonical = {}
        for canonical, variants in self.ontology.items():
            for variant in variants:
                self.variant_to_canonical[variant.lower()] = canonical
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize a single skill to its canonical form."""
        if not skill:
            return ""
        
        # Clean the skill string
        cleaned = self._clean_skill_string(skill)
        
        # Try exact match first
        if cleaned in self.variant_to_canonical:
            return self.variant_to_canonical[cleaned]
        
        # Try fuzzy matching for common patterns
        return self._fuzzy_match(cleaned)
    
    def normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize a list of skills to their canonical forms."""
        normalized = []
        for skill in skills:
            canonical = self.normalize_skill(skill)
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized
    
    def _clean_skill_string(self, skill: str) -> str:
        """Clean and standardize a skill string."""
        # Convert to lowercase
        cleaned = skill.lower()
        
        # Remove common punctuation and extra whitespace
        cleaned = re.sub(r'[^\w\s+]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _fuzzy_match(self, skill: str) -> str:
        """Perform fuzzy matching for skills that don't have exact matches."""
        # Check if the skill contains any known variants
        for variant, canonical in self.variant_to_canonical.items():
            if variant in skill or skill in variant:
                return canonical
        
        # Check for common abbreviations and patterns
        if skill in ['js', 'javascript']:
            return 'javascript'
        elif skill in ['ts', 'typescript']:
            return 'typescript'
        elif skill in ['py', 'python']:
            return 'python'
        elif skill in ['java']:
            return 'java'
        elif skill in ['cpp', 'c++']:
            return 'c++'
        elif skill in ['csharp', 'c#']:
            return 'c#'
        elif skill in ['golang', 'go']:
            return 'go'
        elif skill in ['rustlang', 'rust']:
            return 'rust'
        elif skill in ['k8s', 'kubernetes']:
            return 'kubernetes'
        elif skill in ['iac', 'terraform']:
            return 'terraform'
        elif skill in ['ci', 'jenkins']:
            return 'jenkins'
        elif skill in ['git', 'github', 'gitlab']:
            return 'git'
        elif skill in ['html5', 'html']:
            return 'html'
        elif skill in ['css3', 'css']:
            return 'css'
        elif skill in ['spring', 'j2ee']:
            return 'java'
        elif skill in ['dotnet', '.net']:
            return 'c#'
        elif skill in ['container', 'docker']:
            return 'docker'
        elif skill in ['web server', 'nginx']:
            return 'nginx'
        elif skill in ['httpd', 'apache']:
            return 'apache'
        elif skill in ['cache', 'redis']:
            return 'redis'
        elif skill in ['elastic', 'es']:
            return 'elasticsearch'
        elif skill in ['message queue', 'kafka']:
            return 'kafka'
        elif skill in ['amqp', 'rabbitmq']:
            return 'rabbitmq'
        elif skill in ['gql', 'graphql']:
            return 'graphql'
        elif skill in ['rest api', 'api']:
            return 'rest'
        elif skill in ['micro service', 'microservices']:
            return 'microservices'
        elif skill in ['lambda', 'serverless']:
            return 'serverless'
        elif skill in ['ml', 'ai']:
            return 'machine learning'
        elif skill in ['neural networks', 'deep learning']:
            return 'deep learning'
        elif skill in ['tf', 'tensorflow']:
            return 'tensorflow'
        elif skill in ['torch', 'pytorch']:
            return 'pytorch'
        elif skill in ['sklearn', 'scikit-learn']:
            return 'scikit-learn'
        elif skill in ['pd', 'pandas']:
            return 'pandas'
        elif skill in ['np', 'numpy']:
            return 'numpy'
        elif skill in ['plotting', 'matplotlib']:
            return 'matplotlib'
        elif skill in ['visualization', 'seaborn']:
            return 'seaborn'
        elif skill in ['notebook', 'jupyter']:
            return 'jupyter'
        elif skill in ['bi', 'tableau']:
            return 'tableau'
        elif skill in ['power bi', 'powerbi']:
            return 'powerbi'
        elif skill in ['spreadsheet', 'excel']:
            return 'excel'
        elif skill in ['scrum', 'kanban']:
            return 'agile'
        elif skill in ['project management', 'jira']:
            return 'jira'
        elif skill in ['documentation', 'confluence']:
            return 'confluence'
        elif skill in ['communication', 'slack']:
            return 'slack'
        elif skill in ['microsoft teams', 'teams']:
            return 'teams'
        elif skill in ['design', 'figma']:
            return 'figma'
        elif skill in ['ui design', 'sketch']:
            return 'sketch'
        elif skill in ['photoshop', 'illustrator']:
            return 'adobe'
        elif skill in ['mobile development', 'android']:
            return 'android'
        elif skill in ['iphone', 'swift']:
            return 'ios'
        elif skill in ['cross platform', 'flutter']:
            return 'flutter'
        elif skill in ['reactnative', 'react native']:
            return 'react native'
        elif skill in ['cms', 'wordpress']:
            return 'wordpress'
        elif skill in ['ecommerce', 'shopify']:
            return 'shopify'
        elif skill in ['crm', 'salesforce']:
            return 'salesforce'
        elif skill in ['ga', 'analytics']:
            return 'google analytics'
        elif skill in ['search engine optimization', 'seo']:
            return 'seo'
        elif skill in ['mailchimp', 'email marketing']:
            return 'email marketing'
        elif skill in ['cryptocurrency', 'bitcoin']:
            return 'blockchain'
        elif skill in ['smart contracts', 'ethereum']:
            return 'ethereum'
        elif skill in ['smart contract language', 'solidity']:
            return 'solidity'
        elif skill in ['decentralized web', 'web3']:
            return 'web3'
        elif skill in ['internet of things', 'iot']:
            return 'iot'
        elif skill in ['cloud', 'cloud computing']:
            return 'cloud computing'
        elif skill in ['data engineering', 'big data']:
            return 'big data'
        elif skill in ['data scientist', 'data science']:
            return 'data science'
        elif skill in ['etl', 'data engineering']:
            return 'data engineering'
        elif skill in ['business intelligence', 'data analyst']:
            return 'data analyst'
        elif skill in ['requirements', 'business analyst']:
            return 'business analyst'
        elif skill in ['product management', 'product manager']:
            return 'product manager'
        elif skill in ['project management', 'project manager']:
            return 'project manager'
        elif skill in ['agile coach', 'scrum master']:
            return 'scrum master'
        elif skill in ['development operations', 'devops']:
            return 'devops'
        elif skill in ['sre', 'site reliability']:
            return 'site reliability'
        elif skill in ['cybersecurity', 'infosec']:
            return 'security'
        elif skill in ['pentest', 'penetration testing']:
            return 'penetration testing'
        elif skill in ['gdpr', 'hipaa']:
            return 'compliance'
        elif skill in ['a11y', 'accessibility']:
            return 'accessibility'
        elif skill in ['optimization', 'performance']:
            return 'performance'
        elif skill in ['qa', 'quality assurance']:
            return 'testing'
        elif skill in ['unit tests', 'unit testing']:
            return 'unit testing'
        elif skill in ['integration tests', 'integration testing']:
            return 'integration testing'
        elif skill in ['e2e', 'cypress']:
            return 'end to end testing'
        elif skill in ['webdriver', 'selenium']:
            return 'selenium'
        elif skill in ['browser automation', 'playwright']:
            return 'playwright'
        elif skill in ['api testing', 'postman']:
            return 'postman'
        elif skill in ['openapi', 'swagger']:
            return 'swagger'
        elif skill in ['rpc', 'grpc']:
            return 'grpc'
        elif skill in ['real time', 'websocket']:
            return 'websocket'
        elif skill in ['websocket library', 'socket.io']:
            return 'socket.io'
        elif skill in ['authentication', 'oauth']:
            return 'oauth'
        elif skill in ['json web token', 'jwt']:
            return 'jwt'
        elif skill in ['active directory', 'ldap']:
            return 'ldap'
        elif skill in ['single sign on', 'saml']:
            return 'saml'
        elif skill in ['openid connect', 'oauth2']:
            return 'oauth2'
        elif skill in ['multi factor authentication', 'mfa']:
            return 'mfa'
        elif skill in ['two factor authentication', '2fa']:
            return '2fa'
        elif skill in ['cryptography', 'encryption']:
            return 'encryption'
        elif skill in ['password hashing', 'hashing']:
            return 'hashing'
        elif skill in ['password hashing', 'bcrypt']:
            return 'bcrypt'
        elif skill in ['password hashing', 'argon2']:
            return 'argon2'
        elif skill in ['throttling', 'rate limiting']:
            return 'rate limiting'
        elif skill in ['load balancer', 'load balancing']:
            return 'load balancing'
        elif skill in ['content delivery network', 'cdn']:
            return 'cdn'
        elif skill in ['cache', 'caching']:
            return 'caching'
        elif skill in ['memory cache', 'memcached']:
            return 'memcached'
        elif skill in ['http cache', 'varnish']:
            return 'varnish'
        elif skill in ['observability', 'monitoring']:
            return 'monitoring'
        elif skill in ['log management', 'logging']:
            return 'logging'
        elif skill in ['telemetry', 'metrics']:
            return 'metrics'
        elif skill in ['distributed tracing', 'tracing']:
            return 'tracing'
        elif skill in ['metrics', 'prometheus']:
            return 'prometheus'
        elif skill in ['dashboard', 'grafana']:
            return 'grafana'
        elif skill in ['monitoring', 'datadog']:
            return 'datadog'
        elif skill in ['error tracking', 'sentry']:
            return 'sentry'
        elif skill in ['search engine', 'elasticsearch']:
            return 'elasticsearch'
        elif skill in ['visualization', 'kibana']:
            return 'kibana'
        elif skill in ['log analysis', 'splunk']:
            return 'splunk'
        elif skill in ['aws monitoring', 'cloudwatch']:
            return 'cloudwatch'
        elif skill in ['paas', 'heroku']:
            return 'heroku'
        elif skill in ['deployment', 'vercel']:
            return 'vercel'
        elif skill in ['static hosting', 'netlify']:
            return 'netlify'
        elif skill in ['cdn and security', 'cloudflare']:
            return 'cloudflare'
        elif skill in ['serverless function', 'aws lambda']:
            return 'aws lambda'
        elif skill in ['gcp serverless', 'google cloud functions']:
            return 'google cloud functions'
        elif skill in ['azure serverless', 'azure functions']:
            return 'azure functions'
        elif skill in ['api management', 'aws api gateway']:
            return 'aws api gateway'
        elif skill in ['api gateway', 'kong']:
            return 'kong'
        elif skill in ['api documentation', 'swagger ui']:
            return 'swagger ui'
        elif skill in ['api documentation', 'redoc']:
            return 'redoc'
        elif skill in ['http client', 'curl']:
            return 'curl'
        elif skill in ['windows scripting', 'powershell']:
            return 'powershell'
        elif skill in ['shell scripting', 'bash']:
            return 'bash'
        elif skill in ['shell', 'zsh']:
            return 'zsh'
        elif skill in ['text processing', 'awk']:
            return 'awk'
        elif skill in ['stream editor', 'sed']:
            return 'sed'
        elif skill in ['text search', 'grep']:
            return 'grep'
        elif skill in ['file search', 'find']:
            return 'find'
        elif skill in ['secure shell', 'ssh']:
            return 'ssh'
        elif skill in ['file transfer protocol', 'ftp']:
            return 'ftp'
        elif skill in ['hypertext transfer protocol', 'http']:
            return 'http'
        elif skill in ['secure http', 'https']:
            return 'https'
        elif skill in ['transmission control protocol', 'tcp']:
            return 'tcp'
        elif skill in ['user datagram protocol', 'udp']:
            return 'udp'
        elif skill in ['domain name system', 'dns']:
            return 'dns'
        elif skill in ['dynamic host configuration', 'dhcp']:
            return 'dhcp'
        elif skill in ['virtual private network', 'vpn']:
            return 'vpn'
        elif skill in ['network connectivity', 'ping']:
            return 'ping'
        elif skill in ['network path', 'traceroute']:
            return 'traceroute'
        elif skill in ['network scanning', 'nmap']:
            return 'nmap'
        elif skill in ['packet analysis', 'wireshark']:
            return 'wireshark'
        elif skill in ['firewall', 'iptables']:
            return 'iptables'
        elif skill in ['intrusion prevention', 'fail2ban']:
            return 'fail2ban'
        elif skill in ['linux auditing', 'auditd']:
            return 'auditd'
        elif skill in ['windows monitoring', 'sysmon']:
            return 'sysmon'
        
        # If no match found, return the original skill
        return skill
    
    def get_all_canonical_skills(self) -> Set[str]:
        """Get all canonical skill names from the ontology."""
        return set(self.ontology.keys())
    
    def get_skill_variants(self, canonical_skill: str) -> List[str]:
        """Get all variants for a canonical skill."""
        return self.ontology.get(canonical_skill, [])
