"""
AFML Quant Pipeline - Sanity Check Utility
Uses Gemini with Google Search Grounding to check for alpha decay
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# google-generativeai is an optional extra (requirements-extras.txt).
# Importing this module must not fail when it is absent.
try:
    import google.generativeai as genai

    HAS_GENAI = True
except ImportError:  # pragma: no cover - exercised only without the extra
    genai = None
    HAS_GENAI = False

load_dotenv()

if HAS_GENAI:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def sanity_check_strategy(
    strategy_name: str,
    lookback_months: int = 24
) -> Dict[str, Any]:
    """
    Check if a trading strategy has suffered from alpha decay.
    
    Uses Gemini with Google Search grounding to find:
    - Academic papers about the strategy
    - Quant discussions and forum posts
    - Recent performance data
    
    Args:
        strategy_name: Name of the strategy (e.g., 'Triple Barrier Method')
        lookback_months: How far back to search (default: 24 months)
    
    Returns:
        Dict with:
        - 'strategy': Strategy name
        - 'alpha_decay_detected': bool
        - 'consensus': Summary of findings
        - 'risk_level': 'low', 'medium', 'high'
        - 'citations': List of sources
        - 'recommendations': List of recommendations
    """
    # Configure grounded model
    if not HAS_GENAI:
        raise RuntimeError(
            "google-generativeai is not installed. "
            "Install requirements-extras.txt to use the sanity check."
        )

    google_search_tool = genai.Tool(
        function_declarations=[],
        google_search_retrieval=genai.GoogleSearchRetrieval()
    )
    
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=[google_search_tool],
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            top_p=0.95,
            max_output_tokens=4096,
        )
    )
    
    # Construct the research prompt
    cutoff_date = datetime.now() - timedelta(days=lookback_months * 30)
    
    prompt = f"""Research the trading strategy "{strategy_name}" focusing on:

1. **Alpha Decay**: Has this strategy's edge diminished since {cutoff_date.strftime('%B %Y')}?
   - Look for academic papers, SSRN publications
   - Check quantitative finance forums (Quantopian, Wilmott, etc.)
   - Search for backtesting results and live performance data

2. **Crowding**: Is this strategy now "crowded" (too many traders using it)?
   - Check for mentions in popular trading books or courses
   - Look for hedge fund disclosures using similar approaches

3. **Market Regime Changes**: Have market conditions changed unfavorably?
   - Volatility regime changes
   - Regulatory changes affecting the strategy

4. **Recent Research**: Any new papers or discussions in the last {lookback_months} months?

Provide your findings in this format:

ALPHA_DECAY: [YES/NO/UNCERTAIN]
RISK_LEVEL: [LOW/MEDIUM/HIGH]

CONSENSUS:
[2-3 paragraph summary of findings]

KEY FINDINGS:
- [Finding 1]
- [Finding 2]
...

RECOMMENDATIONS:
- [Recommendation 1]
- [Recommendation 2]
...

CITATIONS:
- [Source 1]
- [Source 2]
..."""

    try:
        response = model.generate_content(prompt)
        
        # Parse the response
        result = _parse_sanity_check_response(response.text, strategy_name)
        
        # Extract citations from grounding metadata if available
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding = candidate.grounding_metadata
                if hasattr(grounding, 'grounding_chunks'):
                    for chunk in grounding.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            result['citations'].append({
                                'title': getattr(chunk.web, 'title', 'Unknown'),
                                'uri': getattr(chunk.web, 'uri', '')
                            })
        
        return result
        
    except Exception as e:
        return {
            'strategy': strategy_name,
            'alpha_decay_detected': None,
            'consensus': f"Error performing sanity check: {e}",
            'risk_level': 'unknown',
            'citations': [],
            'recommendations': ['Unable to complete analysis. Try again later.'],
            'error': str(e)
        }


def _parse_sanity_check_response(content: str, strategy_name: str) -> Dict[str, Any]:
    """Parse the structured response from the model."""
    result = {
        'strategy': strategy_name,
        'alpha_decay_detected': False,
        'consensus': '',
        'risk_level': 'unknown',
        'key_findings': [],
        'citations': [],
        'recommendations': []
    }
    
    lines = content.split('\n')
    current_section = None
    section_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith('ALPHA_DECAY:'):
            decay_value = line_stripped.replace('ALPHA_DECAY:', '').strip().upper()
            result['alpha_decay_detected'] = decay_value == 'YES'
            
        elif line_stripped.startswith('RISK_LEVEL:'):
            result['risk_level'] = line_stripped.replace('RISK_LEVEL:', '').strip().lower()
            
        elif line_stripped.startswith('CONSENSUS:'):
            if current_section and section_content:
                result[current_section] = '\n'.join(section_content).strip()
            current_section = 'consensus'
            section_content = []
            
        elif line_stripped.startswith('KEY FINDINGS:'):
            if current_section and section_content:
                result[current_section] = '\n'.join(section_content).strip()
            current_section = 'key_findings'
            section_content = []
            
        elif line_stripped.startswith('RECOMMENDATIONS:'):
            if current_section and section_content:
                if current_section == 'key_findings':
                    result[current_section] = section_content
                else:
                    result[current_section] = '\n'.join(section_content).strip()
            current_section = 'recommendations'
            section_content = []
            
        elif line_stripped.startswith('CITATIONS:'):
            if current_section and section_content:
                if current_section in ['key_findings', 'recommendations']:
                    result[current_section] = section_content
                else:
                    result[current_section] = '\n'.join(section_content).strip()
            current_section = 'citations'
            section_content = []
            
        elif current_section:
            if line_stripped.startswith('- '):
                section_content.append(line_stripped[2:])
            elif line_stripped:
                section_content.append(line_stripped)
    
    # Don't forget the last section
    if current_section and section_content:
        if current_section in ['key_findings', 'recommendations', 'citations']:
            result[current_section] = section_content
        else:
            result[current_section] = '\n'.join(section_content).strip()
    
    return result


def check_multiple_strategies(strategies: List[str]) -> List[Dict[str, Any]]:
    """
    Check multiple strategies for alpha decay.
    
    Args:
        strategies: List of strategy names
    
    Returns:
        List of sanity check results
    """
    results = []
    
    for strategy in strategies:
        print(f"Checking: {strategy}...")
        result = sanity_check_strategy(strategy)
        results.append(result)
    
    return results


def format_sanity_report(result: Dict[str, Any]) -> str:
    """Format a sanity check result as a readable report."""
    decay_status = "YES" if result['alpha_decay_detected'] else "NO"
    
    risk_icons = {
        'low': '[low]',
        'medium': '[medium]',
        'high': '[high]',
        'unknown': '[unknown]'
    }
    risk_icon = risk_icons.get(result['risk_level'], '[unknown]')
    
    report = f"""
# Sanity Check Report: {result['strategy']}

## Summary
- **Alpha Decay Detected:** {decay_status}
- **Risk Level:** {risk_icon} {result['risk_level'].upper()}

## Consensus
{result['consensus']}

## Key Findings
"""
    
    for finding in result.get('key_findings', []):
        report += f"- {finding}\n"
    
    report += "\n## Recommendations\n"
    for rec in result.get('recommendations', []):
        report += f"- {rec}\n"
    
    if result.get('citations'):
        report += "\n## Citations\n"
        for citation in result['citations']:
            if isinstance(citation, dict):
                report += f"- [{citation.get('title', 'Source')}]({citation.get('uri', '')})\n"
            else:
                report += f"- {citation}\n"
    
    return report


if __name__ == "__main__":
    print("Testing Sanity Check Utility...")
    print("=" * 50)
    
    # Test with Triple Barrier Method
    result = sanity_check_strategy("Triple Barrier Method")
    
    print(format_sanity_report(result))
