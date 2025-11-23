#!/usr/bin/env python3
"""
run_finops.py - Complete FinOps Cloud Plan Generator
Your scenario: Web app with 10,000 users using local GPT4All

Usage:
    python run_finops.py
"""

import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# ============================================================================
# CONFIGURATION - Update these to match your setup
# ============================================================================

GPT4ALL_CONFIG = {
    "model_name": "Llama-3.2-3B-Instruct-Q4_0",
    "model_dir": r"C:\Users\OrCon\AppData\Local\nomic.ai\GPT4All",
    "max_tokens": 2048,
    "temperature": 0.2,
    "allow_download": False
}

USER_SCENARIO = {
    "requirements": "Web application with REST API backend, PostgreSQL database, Redis cache, S3 storage for static assets",
    "users_total": 10,
    "concurrency_pct": 0.05,  # 5% = 500 concurrent users
    "latency_requirements": "< 200ms p99",
    "availability_target": "99.9%"
}

# ============================================================================
# GPT4All Integration
# ============================================================================

try:
    from gpt4all import GPT4All
    HAS_GPT4ALL = True
except ImportError:
    print("⚠️  GPT4All not installed. Run: pip install gpt4all")
    HAS_GPT4ALL = False


class GPT4AllService:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self._init_model()
    
    def _init_model(self):
        if not HAS_GPT4ALL:
            raise RuntimeError("GPT4All not available")
        
        print(f"🔄 Loading model: {self.config['model_name']}")
        print(f"   Path: {self.config['model_dir']}")
        
        self.model = GPT4All(
            model_name=self.config['model_name'],
            model_path=self.config['model_dir'],
            allow_download=self.config['allow_download']
        )
        
        print("✅ Model loaded!")
    
    def generate(self, prompt: str, max_tokens: int = None) -> str:
        max_tokens = max_tokens or self.config['max_tokens']
        
        with self.model.chat_session():
            response = self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temp=self.config['temperature']
            )
        
        return response.strip()


# ============================================================================
# Cloud Plan Data Structures
# ============================================================================

@dataclass
class CloudPlan:
    plan_name: str
    summary: str
    assumptions: Dict
    components: List[Dict]
    est_monthly_cost_usd: float
    cost_category: str
    scalability: str
    risk: str
    rollback_recommendation: str
    notes: str
    provenance: List[Dict]


# ============================================================================
# Cloud Plan Agent
# ============================================================================

class CloudPlanAgent:
    def __init__(self, llm_service: GPT4AllService):
        self.llm_service = llm_service
        self.prompt_template = self._get_prompt_template()
    
    def _get_prompt_template(self) -> str:
        return """You are a cloud architecture expert. Generate a cloud infrastructure plan.

REQUIREMENTS:
{requirements}

SCALE:
- Total Users: {users_total:,}
- Peak Concurrent: {peak_concurrent} ({concurrency_pct}%)
- Latency Target: {latency}
- Availability Target: {availability}

PLAN TYPE: {plan_type}

{guidance}

OUTPUT: Return a valid JSON object with these exact fields:
{{
  "plan_name": "string",
  "summary": "brief description",
  "assumptions": {{"users_total": {users_total}, "concurrency_pct": {concurrency_pct}, "peak_concurrent": {peak_concurrent}}},
  "components": [
    {{"component_type": "LoadBalancer", "description": "...", "count_or_size": "1", "notes": "..."}}
  ],
  "est_monthly_cost_usd": 1000,
  "cost_category": "{cost_category}",
  "scalability": "medium",
  "risk": "medium",
  "rollback_recommendation": "...",
  "notes": "...",
  "provenance": []
}}

Return ONLY the JSON. No extra text."""
    
    def _get_guidance(self, plan_type: str) -> str:
        guides = {
            "standard": """STANDARD PLAN REQUIREMENTS:
- Use managed services (RDS, ElastiCache, LoadBalancer)
- 3-6 medium app servers with autoscaling
- Multi-AZ database with read replica
- Balanced cost vs. reliability
- Target: $800-1500/month""",
            
            "cost-optimized": """COST-OPTIMIZED PLAN REQUIREMENTS:
- Maximize serverless components
- Use spot instances where possible
- Single-AZ acceptable
- Aggressive caching
- Target: $400-800/month""",
            
            "scalable": """SCALABLE PLAN REQUIREMENTS:
- Multi-region capable architecture
- Large autoscaling groups (4-20 instances)
- Clustered database with replicas
- Full observability stack
- Target: $1800-3000/month"""
        }
        return guides.get(plan_type, "")
    
    def generate_plan(self, scenario: Dict, plan_type: str) -> CloudPlan:
        print(f"\n📋 Generating {plan_type.upper()} plan...")
        
        # Build prompt
        peak = int(scenario['users_total'] * scenario['concurrency_pct'])
        cost_cat = plan_type if plan_type != 'cost-optimized' else 'cost-optimized'
        
        prompt = self.prompt_template.format(
            requirements=scenario['requirements'],
            users_total=scenario['users_total'],
            peak_concurrent=peak,
            concurrency_pct=scenario['concurrency_pct'] * 100,
            latency=scenario.get('latency_requirements', 'N/A'),
            availability=scenario.get('availability_target', 'N/A'),
            plan_type=plan_type.upper(),
            guidance=self._get_guidance(plan_type),
            cost_category=cost_cat
        )
        
        # Call LLM
        print(f"   🤖 Calling LLM...")
        start = time.time()
        
        try:
            response = self.llm_service.generate(prompt)
            elapsed = time.time() - start
            
            print(f"   ✓ Generated in {elapsed:.1f}s ({len(response)} chars)")
            
            # Parse JSON
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            
            parsed = json.loads(cleaned.strip())
            
            # Create CloudPlan
            plan = CloudPlan(
                plan_name=parsed.get('plan_name', f'{plan_type} Plan'),
                summary=parsed.get('summary', ''),
                assumptions=parsed.get('assumptions', {}),
                components=parsed.get('components', []),
                est_monthly_cost_usd=float(parsed.get('est_monthly_cost_usd', 0)),
                cost_category=parsed.get('cost_category', plan_type),
                scalability=parsed.get('scalability', 'medium'),
                risk=parsed.get('risk', 'medium'),
                rollback_recommendation=parsed.get('rollback_recommendation', ''),
                notes=parsed.get('notes', ''),
                provenance=parsed.get('provenance', [])
            )
            
            print(f"   ✅ {plan.plan_name} - ${plan.est_monthly_cost_usd:,.0f}/mo")
            return plan
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print(f"   Response preview: {response[:200]}...")
            
            # Return fallback
            return self._fallback_plan(plan_type, scenario)
    
    def _fallback_plan(self, plan_type: str, scenario: Dict) -> CloudPlan:
        costs = {"standard": 1100, "cost-optimized": 600, "scalable": 2200}
        
        return CloudPlan(
            plan_name=f"{plan_type.title()} (Fallback)",
            summary="Generated fallback due to LLM error",
            assumptions={
                "users_total": scenario['users_total'],
                "concurrency_pct": scenario['concurrency_pct'],
                "peak_concurrent": int(scenario['users_total'] * scenario['concurrency_pct'])
            },
            components=[],
            est_monthly_cost_usd=costs.get(plan_type, 0),
            cost_category=plan_type,
            scalability="medium",
            risk="high",
            rollback_recommendation="None",
            notes="LLM failed - manual review needed",
            provenance=[]
        )


# ============================================================================
# Main Execution
# ============================================================================

def print_banner():
    print("\n" + "="*80)
    print("🏗️  AI FINOPS CLOUD PLAN GENERATOR")
    print("="*80)
    print(f"\n📊 Scenario: {USER_SCENARIO['requirements']}")
    print(f"👥 Users: {USER_SCENARIO['users_total']:,}")
    print(f"⚡ Peak Concurrent: {int(USER_SCENARIO['users_total'] * USER_SCENARIO['concurrency_pct']):,}")
    print()


def print_comparison(plans: Dict[str, CloudPlan]):
    print("\n" + "="*80)
    print("📊 PLAN COMPARISON")
    print("="*80)
    
    for plan_type, plan in plans.items():
        print(f"\n🎯 {plan.plan_name}")
        print(f"   💰 Cost: ${plan.est_monthly_cost_usd:,.2f}/month")
        print(f"   📈 Scalability: {plan.scalability}")
        print(f"   ⚠️  Risk: {plan.risk}")
        print(f"   🔧 Components: {len(plan.components)}")
        
        if plan.components:
            print(f"\n   Components:")
            for comp in plan.components[:5]:
                print(f"      • {comp['component_type']}: {comp['count_or_size']}")
            if len(plan.components) > 5:
                print(f"      ... and {len(plan.components) - 5} more")


def save_results(plans: Dict[str, CloudPlan], scenario: Dict):
    output_path = Path("outputs/reports/cloud_plans_finops.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "request_id": f"req_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scenario": scenario,
        "plans": {
            k: {
                "plan_name": p.plan_name,
                "summary": p.summary,
                "assumptions": p.assumptions,
                "components": p.components,
                "est_monthly_cost_usd": p.est_monthly_cost_usd,
                "cost_category": p.cost_category,
                "scalability": p.scalability,
                "risk": p.risk,
                "rollback_recommendation": p.rollback_recommendation,
                "notes": p.notes,
                "provenance": p.provenance
            }
            for k, p in plans.items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\n💾 Results saved to: {output_path}")


def main():
    print_banner()
    
    # Initialize LLM
    print("🔧 Initializing GPT4All...")
    try:
        llm = GPT4AllService(GPT4ALL_CONFIG)
    except Exception as e:
        print(f"\n❌ Failed to initialize GPT4All: {e}")
        print("\n💡 Make sure:")
        print("   1. gpt4all is installed: pip install gpt4all")
        print("   2. Model path is correct in GPT4ALL_CONFIG")
        print("   3. Model file exists")
        sys.exit(1)
    
    # Create agent
    agent = CloudPlanAgent(llm)
    
    # Generate plans
    print("\n🚀 Generating cloud architecture plans...")
    plans = {}
    
    for plan_type in ["standard", "cost-optimized", "scalable"]:
        plans[plan_type] = agent.generate_plan(USER_SCENARIO, plan_type)
        time.sleep(0.5)  # Brief pause between generations
    
    # Display results
    print_comparison(plans)
    
    # Save results
    save_results(plans, USER_SCENARIO)
    
    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()