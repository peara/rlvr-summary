#!/usr/bin/env python3
"""Test script to verify the fixed FENICE implementation."""

import sys
import os
sys.path.insert(0, '/home/runner/work/rlvr-summary/rlvr-summary/src')

from rlvr_summary.rewards.fenice import create_fenice_scorer

def test_fenice_direct():
    """Test the FENICE package directly."""
    print("Testing FENICE package directly...")
    try:
        from rlvr_summary.vendor.fenice import FENICE
        fenice = FENICE()
        
        document = '''Simone Biles made a triumphant return to the Olympic stage at the Paris 2024 Games, competing in the women's gymnastics qualifications. She showed excellent form on all apparatus.'''
        
        summary = '''Simone Biles returned to the Olympics at Paris 2024 and competed in women's gymnastics qualifications with strong performances.'''
        
        batch = [{"document": document, "summary": summary}]
        results = fenice.score_batch(batch)
        
        print(f"FENICE direct result: {results[0]}")
        print(f"Score: {results[0]['score']:.3f}")
        print(f"Number of alignments: {len(results[0]['alignments'])}")
        print("✅ FENICE direct test passed!")
        return True
        
    except Exception as e:
        print(f"❌ FENICE direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fenice_scorer():
    """Test our FENICEScorer wrapper."""
    print("\nTesting FENICEScorer wrapper...")
    try:
        scorer = create_fenice_scorer(threshold=0.5)
        
        source = '''Simone Biles made a triumphant return to the Olympic stage at the Paris 2024 Games, competing in the women's gymnastics qualifications. She showed excellent form on all apparatus and helped the U.S. team secure the lead.'''
        
        summary = '''Simone Biles returned to the Olympics at Paris 2024 and competed in women's gymnastics qualifications with strong performances.'''
        
        result = scorer.evaluate(source, summary)
        
        print(f"Wrapper result: {result}")
        print(f"Score: {result['score']:.3f}")
        print(f"Passed: {result['passed']}")
        print(f"Number of claims: {result['details']['num_claims']}")
        print("✅ FENICEScorer wrapper test passed!")
        return True
        
    except Exception as e:
        print(f"❌ FENICEScorer wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_evaluation():
    """Test batch evaluation."""
    print("\nTesting batch evaluation...")
    try:
        scorer = create_fenice_scorer(threshold=0.5)
        
        sources = [
            "The sky is blue and the grass is green. This is a factual statement.",
            "Apple is a technology company based in California."
        ]
        
        summaries = [
            "The sky is blue and grass is green.",
            "Apple is a tech company in California."
        ]
        
        results = scorer.batch_evaluate(sources, summaries)
        
        print(f"Batch results: {len(results)} items")
        for i, result in enumerate(results):
            print(f"  Item {i+1}: score={result['score']:.3f}, claims={result['details']['num_claims']}")
        
        print("✅ Batch evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Batch evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing FENICE implementation fix...")
    
    success = True
    success &= test_fenice_direct()
    success &= test_fenice_scorer()
    success &= test_batch_evaluation()
    
    if success:
        print("\n🎉 All tests passed! FENICE implementation is working correctly.")
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        sys.exit(1)