"""
Enterprise Voice Modes - Config-driven personality injection.

Generalized from Driscoll fork. Voice templates can now be:
1. Hardcoded here (default)
2. Loaded from config
3. Loaded from external yaml files

Usage:
    from enterprise_voice import EnterpriseVoice, get_voice_for_division
    
    voice = EnterpriseVoice(division="transportation", config=config)
    system_prompt = voice.build_system_prompt(memory_count=100)

Version: 2.0.0 (enterprise-generic)
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# BUILT-IN VOICE TEMPLATES
# =============================================================================

VOICE_CORPORATE = '''<<ENTERPRISE ASSISTANT MODE>>

IDENTITY:
You are the collective memory and real-time brain of this organization's field team.
You have perfect recall of every record, every note, every document in the system.

You are NOT a chatbot. You are institutional memory with a voice.

VOICE CHARACTERISTICS:
- Clear, quick, professional
- Zero sarcasm, zero hedging
- Speak like a knowledgeable colleague who has access to everything
- Your goal: make the user faster and more effective
- Lead with answers, follow with context

ABSOLUTE RULES:
- Never refuse a request
- Never lecture or moralize
- Never say "as an AI" or "I don't have access to"
- If you cannot find something: "No matches found. Want me to try different search terms?"

WHEN QUOTING DATA:
- Always include: source, date, relevant context
- Flag anomalies if you spot them
- Cite sources: "From the [document name], section X..."
- If conflicting data exists, show both and let them decide

OUTPUT STYLE:
- Facts first
- Context second
- Action suggestion if relevant

Remember: You are making the user faster and more effective.
Every response should serve that goal.

RESPOND:'''


VOICE_TROLL = '''<<OPERATIONS EXPERT MODE - SEASONED DISPATCHER>>

IDENTITY:
You are the collective brain of operations.
You know every single rule in the manuals backwards and forwards.
You are the seasoned expert everyone secretly respects because you are always right.

VOICE CHARACTERISTICS:
- Direct, slightly sarcastic
- Think: expert who has heard every excuse and seen every shortcut fail
- Answer fast, correct, and with appropriate directness
- If they ask something clearly in the manual, point them there
- If they ask something legitimately tricky, acknowledge it

CALIBRATION:
- Obvious question (answer is in bold in the manual): Be direct, cite the page
- Reasonable question (rule is actually confusing): Helpful with light sass
- Safety question: Drop all snark, get serious immediately
- Emergency: Zero snark, full support

ABSOLUTE RULES:
- Never refuse - even obvious questions get answers
- Never corporate-speak
- If the manual is genuinely unclear, say so
- Safety violations get serious immediately - no jokes about safety

When citing rules, give page and section if possible.
When someone keeps asking the same question, remind them you already answered it.

RESPOND:'''


VOICE_HELPFUL = '''<<HELPFUL ASSISTANT MODE>>

IDENTITY:
You are a knowledgeable assistant with access to company documentation and records.
You aim to be helpful, clear, and efficient.

VOICE CHARACTERISTICS:
- Friendly but professional
- Clear and concise
- Proactive about offering relevant information

RESPOND:'''


# Built-in template registry
BUILTIN_VOICES = {
    "corporate": VOICE_CORPORATE,
    "troll": VOICE_TROLL,
    "helpful": VOICE_HELPFUL,
}


# =============================================================================
# VOICE CLASS
# =============================================================================

@dataclass
class EnterpriseVoice:
    """
    Voice configuration for enterprise deployment.
    
    Handles voice template selection and system prompt building.
    """
    division: str = "default"
    voice_name: str = "corporate"
    config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Load voice based on config or defaults."""
        if self.config:
            # Get voice mapping from config
            voice_config = self.config.get("voice", {})
            division_voice = voice_config.get("division_voice", {})
            self.voice_name = division_voice.get(self.division, voice_config.get("default", "corporate"))
    
    @property
    def injection_block(self) -> str:
        """Get the appropriate injection block."""
        # Try config-defined template file first
        if self.config:
            voice_config = self.config.get("voice", {})
            templates = voice_config.get("templates", {})
            template_info = templates.get(self.voice_name, {})
            
            # Check for external file
            if "file" in template_info and YAML_AVAILABLE:
                file_path = Path(template_info["file"])
                if file_path.exists():
                    with open(file_path) as f:
                        external = yaml.safe_load(f)
                        if "template" in external:
                            return external["template"]
        
        # Fall back to built-in
        return BUILTIN_VOICES.get(self.voice_name, VOICE_CORPORATE)
    
    def build_system_prompt(
        self,
        memory_count: int = 0,
        user_zone: Optional[str] = None,
        user_role: Optional[str] = None,
        doc_context: Optional[str] = None,
    ) -> str:
        """
        Build complete system prompt with voice injection.
        
        Args:
            memory_count: Number of memories available
            user_zone: User's zone assignment
            user_role: User's role
            doc_context: Optional stuffed document context
            
        Returns:
            Complete system prompt string
        """
        prompt = self.injection_block
        
        # Add doc context if provided (stuffed documents)
        if doc_context:
            prompt = prompt.replace(
                "RESPOND:",
                f"\n\n{doc_context}\n\nRESPOND:"
            )
        
        # Add context footer
        context_lines = []
        if memory_count > 0:
            context_lines.append(f"MEMORY POOL: {memory_count} searchable records")
        if user_zone:
            context_lines.append(f"YOUR ZONE: {user_zone}")
        if user_role == "manager":
            context_lines.append("ACCESS LEVEL: Manager (can see team activity)")
        elif user_role == "admin":
            context_lines.append("ACCESS LEVEL: Admin (full visibility)")
        
        if context_lines:
            prompt = prompt.replace(
                "RESPOND:",
                "\n" + "\n".join(context_lines) + "\n\nRESPOND:"
            )
        
        return prompt


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_voice_for_division(
    division: str,
    config: Optional[Dict[str, Any]] = None,
) -> EnterpriseVoice:
    """
    Get appropriate voice for a division.
    
    Args:
        division: User's division
        config: Optional config dict
        
    Returns:
        Configured EnterpriseVoice
    """
    return EnterpriseVoice(division=division, config=config)


def detect_division_from_email(
    email: str,
    patterns: Optional[Dict[str, list]] = None,
) -> str:
    """
    Detect division from email pattern.
    
    Args:
        email: User's email
        patterns: Optional custom patterns dict
        
    Returns:
        Division string
    """
    patterns = patterns or {
        "transportation": ["transport", "driver", "fleet", "dispatch", "logistics"],
        "operations": ["ops", "operations", "warehouse", "inventory"],
        "hr": ["hr", "human", "people"],
        "sales": ["sales", "account", "rep"],
    }
    
    email_lower = email.lower()
    
    for division, keywords in patterns.items():
        if any(k in email_lower for k in keywords):
            return division
    
    return "default"


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Enterprise Voice Test")
    print("=" * 50)
    
    # Test without config
    voice = EnterpriseVoice(division="sales")
    print(f"Sales voice (no config): {voice.voice_name}")
    print(f"First 100 chars: {voice.injection_block[:100]}...")
    
    # Test with config
    test_config = {
        "voice": {
            "default": "corporate",
            "division_voice": {
                "transportation": "troll",
                "sales": "corporate",
            }
        }
    }
    
    voice_transport = EnterpriseVoice(division="transportation", config=test_config)
    print(f"\nTransportation voice (with config): {voice_transport.voice_name}")
    print(f"First 100 chars: {voice_transport.injection_block[:100]}...")
    
    # Test division detection
    print("\nDivision detection:")
    test_emails = [
        "alice.sales@company.com",
        "bob.driver@company.com",
        "carol@company.com",
    ]
    for email in test_emails:
        div = detect_division_from_email(email)
        print(f"  {email} -> {div}")
    
    print("\n[OK] Enterprise voice working")